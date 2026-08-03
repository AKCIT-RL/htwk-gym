"""Multi-motion soccer policy evaluation harness (E5 gate).

Layers (per the project evaluation plan):
  1. task metrics: goals/ep, out-of-bounds/ep, falls/ep, episode length,
     time to first ball touch, best ball->goal progress;
  2. kick events: ball speed gain while a foot is in contact;
  3. style coverage: nearest-clip classification of rollout disc obs
     against per-clip demo windows (detects mode collapse / averaging);
  4. (visual layer is the interactive viewer, not this script).

Run inside the mimickit-isaacgym container from the MimicKit root:
    python3.8 /workspace/htwk-gym/scripts/evaluate_soccer_policy.py \
        --model_file output/mcwamp_g1_soccer_smoke_seed1/model.pt \
        [--num_envs 32] [--episodes 64]
"""

import argparse
import os
import sys

MIMICKIT_ROOT = "/workspace/MimicKit"
sys.path.insert(0, os.path.join(MIMICKIT_ROOT, "mimickit"))
os.chdir(MIMICKIT_ROOT)

import isaacgym  # noqa: F401  (must precede torch)
import numpy as np
import torch

import envs.env_builder as env_builder
import learning.agent_builder as agent_builder
import util.mp_util as mp_util
import util.util as util

KICK_SPEED_GAIN = 0.5    # m/s ball speed gain in foot contact = kick event
STYLE_DEMOS_PER_CLIP = 512
STYLE_SUBSAMPLE = 5      # keep every k-th rollout disc obs window


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_file", required=True)
    p.add_argument("--env_config", default="data/envs/mcwamp_g1_soccer_env.yaml")
    p.add_argument("--agent_config", default="data/agents/mcwamp_g1_soccer_agent.yaml")
    p.add_argument("--engine_config", default="data/engines/isaac_gym_engine.yaml")
    p.add_argument("--num_envs", type=int, default=32)
    p.add_argument("--episodes", type=int, default=64)
    p.add_argument("--rand_seed", type=int, default=1)
    p.add_argument("--no_perturb", action="store_true",
                   help="disable random ball perturbations (clean eval: goals "
                        "and progress are attributable to the policy only)")
    return p.parse_args()


def build(args, device):
    env = env_builder.build_env(env_file=args.env_config,
                                engine_file=args.engine_config,
                                num_envs=args.num_envs, device=device,
                                visualize=False)
    agent = agent_builder.build_agent(args.agent_config, env, device)
    agent.load(args.model_file)

    if (args.no_perturb):
        # push perturbation triggers beyond any episode length
        env._ball_perturb_time_min = 1.0e9
        env._ball_perturb_time_max = 1.0e9
        env._ball_perturb_times[:] = 1.0e9
    return env, agent


def fetch_clip_demos(env, agent, device):
    """Normalized demo disc-obs windows per motion clip."""
    motion_lib = env._motion_lib
    num_motions = motion_lib.get_num_motions()
    window = env.get_num_disc_obs_steps() * env._engine.get_timestep()

    demos = []
    names = []
    for m in range(num_motions):
        ids = torch.full([STYLE_DEMOS_PER_CLIP], m, device=device, dtype=torch.long)
        length = motion_lib.get_motion_length(ids)
        # keep the full history window inside the clip
        lo = torch.clamp(torch.full_like(length, window), max=length)
        t = lo + (length - lo) * torch.rand_like(length)
        disc_obs = env._compute_disc_obs_demo(ids, t)
        demos.append(agent._disc_obs_norm.normalize(disc_obs))
        names.append(os.path.basename(str(motion_lib._motion_files[m]))
                     if hasattr(motion_lib, "_motion_files") else "clip_{:d}".format(m))
    return demos, names


def nearest_clip_fractions(rollout_obs, demos):
    """Fraction of rollout windows whose nearest demo window is from each clip."""
    n_clips = len(demos)
    counts = torch.zeros(n_clips)
    for chunk in torch.split(rollout_obs, 256):
        best_d = None
        best_c = None
        for c, demo in enumerate(demos):
            d = torch.cdist(chunk, demo).min(dim=-1).values
            if best_d is None:
                best_d, best_c = d, torch.full_like(d, c)
            else:
                closer = d < best_d
                best_d = torch.where(closer, d, best_d)
                best_c = torch.where(closer, torch.full_like(best_c, c), best_c)
        for c in range(n_clips):
            counts[c] += (best_c == c).sum().item()
    total = counts.sum().clamp_min(1.0)
    return (counts / total).tolist()


def main():
    args = parse_args()
    device = "cuda:0"
    mp_util.init(0, 1, device, int(np.random.randint(6000, 7000)))
    util.set_rand_seed(args.rand_seed)

    env, agent = build(args, device)
    agent.eval()
    import learning.base_agent as base_agent
    agent.set_mode(base_agent.AgentMode.TEST)

    num_envs = args.num_envs
    dt = env._engine.get_timestep()
    ball_id = env._get_ball_id()
    char_id = env._get_char_id()
    foot_ids = env._foot_body_ids

    demos, clip_names = fetch_clip_demos(env, agent, device)

    # per-env episode accumulators
    ep_goals = torch.zeros(num_envs, device=device)
    ep_oob = torch.zeros(num_envs, device=device)
    ep_kicks = torch.zeros(num_envs, device=device)
    ep_first_touch = torch.full([num_envs], -1.0, device=device)
    ep_best_progress = torch.zeros(num_envs, device=device)
    ep_start_goal_dist = torch.zeros(num_envs, device=device)

    # completed-episode stats
    fin = {"goals": [], "oob": [], "kicks": [], "len": [], "first_touch": [],
           "progress": [], "falls": 0, "episodes": 0}

    style_windows = []

    def init_episode_stats(env_ids):
        ball_pos = env._get_ball_pos()[env_ids]
        goal_pos = env._goal_pos[env_ids]
        d = torch.linalg.norm(goal_pos - ball_pos[:, 0:2], dim=-1)
        ep_goals[env_ids] = 0.0
        ep_oob[env_ids] = 0.0
        ep_kicks[env_ids] = 0.0
        ep_first_touch[env_ids] = -1.0
        ep_best_progress[env_ids] = 0.0
        ep_start_goal_dist[env_ids] = d

    obs, info = env.reset()
    init_episode_stats(torch.arange(num_envs, device=device, dtype=torch.long))
    prev_ball_speed = torch.linalg.norm(
        env._engine.get_root_vel(ball_id)[:, 0:2], dim=-1)

    step_i = 0
    with torch.no_grad():
        while fin["episodes"] < args.episodes:
            a, _ = agent._decide_action(obs, info)
            obs, r, done, info = env.step(a)

            # --- events (post-step, pre-soft-reset flags cached by the env) --
            ep_goals += env._goal_scored_buf.float()
            ep_oob += env._ball_oob_buf.float()

            # ball touch / kick detection
            ball_pos = env._get_ball_pos()
            ball_speed = torch.linalg.norm(
                env._engine.get_root_vel(ball_id)[:, 0:2], dim=-1)
            body_pos = env._engine.get_body_pos(char_id)
            in_contact = torch.zeros(num_envs, device=device, dtype=torch.bool)
            for i in range(len(foot_ids)):
                foot_pos = body_pos[:, foot_ids[i], :]
                d = torch.linalg.norm(foot_pos - ball_pos, dim=-1)
                in_contact |= d < env._ball_contact_dist
            first = in_contact & (ep_first_touch < 0.0)
            ep_first_touch[first] = env._time_buf[first]
            kick = in_contact & ((ball_speed - prev_ball_speed) > KICK_SPEED_GAIN)
            ep_kicks += kick.float()
            prev_ball_speed = ball_speed

            # ball->goal progress (best over the episode)
            goal_d = torch.linalg.norm(env._goal_pos - ball_pos[:, 0:2], dim=-1)
            ep_best_progress = torch.maximum(ep_best_progress,
                                             ep_start_goal_dist - goal_d)

            # style windows
            if (step_i % STYLE_SUBSAMPLE == 0):
                style_windows.append(
                    agent._disc_obs_norm.normalize(info["disc_obs"]).cpu())
            step_i += 1

            # --- episode boundaries ------------------------------------------
            done_ids = (done != 0).nonzero(as_tuple=False).flatten()
            if len(done_ids) > 0:
                for e in done_ids.tolist():
                    fin["episodes"] += 1
                    fin["goals"].append(ep_goals[e].item())
                    fin["oob"].append(ep_oob[e].item())
                    fin["kicks"].append(ep_kicks[e].item())
                    fin["len"].append(env._time_buf[e].item())
                    fin["progress"].append(ep_best_progress[e].item())
                    if ep_first_touch[e].item() >= 0.0:
                        fin["first_touch"].append(ep_first_touch[e].item())
                    if done[e].item() == 1:  # DoneFlags.FAIL
                        fin["falls"] += 1
                obs_r, info_r = env.reset(done_ids)
                obs, info = obs_r, info_r
                init_episode_stats(done_ids)
                prev_ball_speed = torch.linalg.norm(
                    env._engine.get_root_vel(ball_id)[:, 0:2], dim=-1)

    # --------------------------- report ---------------------------------------
    n = fin["episodes"]

    def stat(key):
        v = fin[key]
        return (float(np.mean(v)), float(np.std(v))) if len(v) > 0 else (float("nan"), float("nan"))

    print("")
    print("================ SOCCER POLICY EVALUATION ================")
    print("model: {:s}".format(args.model_file))
    print("episodes: {:d}   sim dt: {:.4f}s".format(n, dt))
    print("")
    print("-- task metrics (mean +- std / episode) --")
    print("goals:          {:.3f} +- {:.3f}".format(*stat("goals")))
    print("goal rate:      {:.1%} of episodes with >=1 goal".format(
        float(np.mean([g > 0 for g in fin["goals"]]))))
    print("out of bounds:  {:.3f} +- {:.3f}".format(*stat("oob")))
    print("falls:          {:.1%} of episodes".format(fin["falls"] / max(n, 1)))
    print("episode length: {:.1f} +- {:.1f} s".format(*stat("len")))
    print("first touch:    {:.1f} +- {:.1f} s ({:d}/{:d} episodes touched the ball)".format(
        *stat("first_touch"), len(fin["first_touch"]), n))
    print("best progress:  {:.2f} +- {:.2f} m (ball toward goal)".format(*stat("progress")))
    print("")
    print("-- kick events (contact + ball speed gain > {:.1f} m/s) --".format(KICK_SPEED_GAIN))
    print("kicks/episode:  {:.3f} +- {:.3f}".format(*stat("kicks")))
    print("")
    print("-- style coverage (nearest demo clip per rollout window) --")
    rollout_obs = torch.cat(style_windows, dim=0)
    fracs = nearest_clip_fractions(rollout_obs, [d.cpu() for d in demos])
    for name, f in zip(clip_names, fracs):
        print("{:30s} {:.1%}".format(name, f))
    print("==========================================================")
    return


if __name__ == "__main__":
    main()
