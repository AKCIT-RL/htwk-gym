"""Multi-motion soccer policy evaluation harness (E5 gate).

Layers (per the project evaluation plan):
  1. task metrics: goals/ep, out-of-bounds/ep, falls/ep, episode length,
     time to first ball touch, best ball->goal progress;
  2. kick events: ball speed gain while a foot is in contact;
  2b. kick impact metrics (T1 kicking-env style): angular error toward the
      goal at impact, impact speed, ball state at impact, engage latency,
      kicking foot, and per-kick outcome (goal <= 5 s / OOB <= 2 s /
      fall <= 1 s after the kick);
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
BALL_STILL_SPEED = 0.1   # m/s: below this the ball counts as stationary
ENGAGE_DIST = 1.0        # m: ball "within reach" for the engage-latency timer
OUTCOME_GOAL_S = 5.0     # kick -> goal attribution window
OUTCOME_OOB_S = 2.0      # kick -> out-of-bounds attribution window
OUTCOME_FALL_S = 1.0     # kick -> fall attribution window
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

    # kick impact stats (one entry per kick event)
    kick_stats = {"speed": [], "ang_err": [], "on_still": [], "foot": [],
                  "latency": []}
    # per-env in-episode event log for kick outcome attribution
    ep_events = [{"kicks": [], "goals": [], "oobs": []} for _ in range(num_envs)]
    outcomes = {"total": 0, "goal": 0, "oob": 0, "fall": 0}
    # ball-within-reach-and-still timer (engage latency)
    reach_still_t = torch.zeros(num_envs, device=device)

    def close_episode_events(e, fall_time):
        ev = ep_events[e]
        for kt in ev["kicks"]:
            outcomes["total"] += 1
            if any(kt < gt <= kt + OUTCOME_GOAL_S for gt in ev["goals"]):
                outcomes["goal"] += 1
            if any(kt < ot <= kt + OUTCOME_OOB_S for ot in ev["oobs"]):
                outcomes["oob"] += 1
            if fall_time is not None and kt < fall_time <= kt + OUTCOME_FALL_S:
                outcomes["fall"] += 1
        ep_events[e] = {"kicks": [], "goals": [], "oobs": []}

    style_windows = []

    # --- Frente H: ball-placement bookkeeping -----------------------------
    # Every ball placement (episode start / soft reset after goal or OOB)
    # opens a "trial": we bin its field region (3x3 grid, x oriented toward
    # the goal) and the robot->ball approach angle, then score whether that
    # placement ended in a goal and how long the first kick took.
    GRID = 3
    hl = 0.5 * env._field_length
    hw = 0.5 * env._field_width
    grid_total = np.zeros([GRID, GRID], dtype=np.int64)
    grid_goals = np.zeros([GRID, GRID], dtype=np.int64)
    ANGLE_BINS = [0.0, 45.0, 90.0, 135.0, 180.0]
    kick_time_by_angle = [[] for _ in range(len(ANGLE_BINS) - 1)]
    plc_cell = torch.zeros(num_envs, 2, dtype=torch.long)          # (row, col)
    plc_angle = torch.zeros(num_envs)                              # deg
    plc_time = torch.zeros(num_envs, device=device)
    plc_kicked = torch.zeros(num_envs, dtype=torch.bool)

    def record_placements(env_ids):
        """Open a trial for each env in env_ids using the CURRENT (fresh)
        ball position. Call right after a reset/soft reset."""
        ball = env._get_ball_pos()[env_ids, 0:2]
        local = ball - env._field_offset[env_ids]
        gdir = env._goal_dir[env_ids]
        x_goal = (local * gdir).sum(dim=-1)          # + toward the goal
        col = torch.clamp(((x_goal + hl) / (2.0 * hl) * GRID).long(), 0, GRID - 1)
        row = torch.clamp(((local[:, 1] + hw) / (2.0 * hw) * GRID).long(), 0, GRID - 1)
        plc_cell[env_ids] = torch.stack([row.cpu(), col.cpu()], dim=-1)
        # approach angle: between (ball - robot) and (goal - ball); 0 deg =
        # robot straight behind the ball relative to the goal
        root = env._engine.get_root_pos(char_id)[env_ids, 0:2]
        v_rb = ball - root
        v_bg = env._goal_pos[env_ids] - ball
        v_rb = v_rb / torch.clamp_min(torch.linalg.norm(v_rb, dim=-1, keepdim=True), 1e-6)
        v_bg = v_bg / torch.clamp_min(torch.linalg.norm(v_bg, dim=-1, keepdim=True), 1e-6)
        ang = torch.rad2deg(torch.acos(torch.clamp((v_rb * v_bg).sum(dim=-1), -1.0, 1.0)))
        plc_angle[env_ids] = ang.cpu()
        plc_time[env_ids] = env._time_buf[env_ids]
        plc_kicked[env_ids] = False

    def close_placements(env_ids, scored):
        """Close trials: scored is a bool tensor aligned with env_ids."""
        for j, e in enumerate(env_ids.tolist()):
            r, c = plc_cell[e, 0].item(), plc_cell[e, 1].item()
            grid_total[r, c] += 1
            if bool(scored[j]):
                grid_goals[r, c] += 1

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
    record_placements(torch.arange(num_envs, device=device, dtype=torch.long))
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
            ball_vel = env._engine.get_root_vel(ball_id)[:, 0:2]
            ball_speed = torch.linalg.norm(ball_vel, dim=-1)
            body_pos = env._engine.get_body_pos(char_id)
            in_contact = torch.zeros(num_envs, device=device, dtype=torch.bool)
            foot_dists = []
            for i in range(len(foot_ids)):
                foot_pos = body_pos[:, foot_ids[i], :]
                d = torch.linalg.norm(foot_pos - ball_pos, dim=-1)
                foot_dists.append(d)
                in_contact |= d < env._ball_contact_dist
            first = in_contact & (ep_first_touch < 0.0)
            ep_first_touch[first] = env._time_buf[first]
            kick = in_contact & ((ball_speed - prev_ball_speed) > KICK_SPEED_GAIN)
            ep_kicks += kick.float()

            # --- kick impact metrics (T1 style, measured at the kick step) --
            kick_ids = kick.nonzero(as_tuple=False).flatten()
            if len(kick_ids) > 0:
                to_goal = env._goal_pos[kick_ids] - ball_pos[kick_ids, 0:2]
                to_goal = to_goal / torch.clamp_min(
                    torch.linalg.norm(to_goal, dim=-1, keepdim=True), 1e-6)
                v = ball_vel[kick_ids]
                v_norm = torch.clamp_min(torch.linalg.norm(v, dim=-1), 1e-6)
                cos_err = torch.clamp((v * to_goal).sum(dim=-1) / v_norm, -1.0, 1.0)
                ang_err = torch.rad2deg(torch.acos(cos_err))
                nearest_foot = torch.stack(foot_dists, dim=-1)[kick_ids].argmin(dim=-1)
                on_still = prev_ball_speed[kick_ids] < BALL_STILL_SPEED
                lat = reach_still_t[kick_ids]  # pre-update: excludes this step
                for j, e in enumerate(kick_ids.tolist()):
                    kick_stats["speed"].append(ball_speed[e].item())
                    kick_stats["ang_err"].append(ang_err[j].item())
                    kick_stats["on_still"].append(bool(on_still[j].item()))
                    kick_stats["foot"].append(int(nearest_foot[j].item()))
                    if lat[j].item() > 0.0:
                        kick_stats["latency"].append(lat[j].item())
                    ep_events[e]["kicks"].append(env._time_buf[e].item())
                    # Frente H: first kick of the current placement
                    if not bool(plc_kicked[e]):
                        plc_kicked[e] = True
                        t_kick = env._time_buf[e].item() - plc_time[e].item()
                        ang = plc_angle[e].item()
                        b = min(int(ang / 45.0), len(ANGLE_BINS) - 2)
                        kick_time_by_angle[b].append(t_kick)

            # goal / OOB event times for outcome attribution
            for e in env._goal_scored_buf.nonzero(as_tuple=False).flatten().tolist():
                ep_events[e]["goals"].append(env._time_buf[e].item())
            for e in env._ball_oob_buf.nonzero(as_tuple=False).flatten().tolist():
                ep_events[e]["oobs"].append(env._time_buf[e].item())

            # Frente H: goal/OOB soft-resets close the trial and (the env has
            # already teleported the ball) open the next one
            soft = torch.logical_or(env._goal_scored_buf, env._ball_oob_buf)
            soft_ids = soft.nonzero(as_tuple=False).flatten()
            if len(soft_ids) > 0:
                close_placements(soft_ids, env._goal_scored_buf[soft_ids].cpu())
                record_placements(soft_ids)

            # engage-latency timer: ball within reach and stationary
            root_pos = env._engine.get_root_pos(char_id)
            near_still = (torch.linalg.norm(
                ball_pos[:, 0:2] - root_pos[:, 0:2], dim=-1) < ENGAGE_DIST) \
                & (ball_speed < BALL_STILL_SPEED)
            reach_still_t = torch.where(near_still, reach_still_t + dt,
                                        torch.zeros_like(reach_still_t))
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
                    fall_time = None
                    if done[e].item() == 1:  # DoneFlags.FAIL
                        fin["falls"] += 1
                        fall_time = env._time_buf[e].item()
                    close_episode_events(e, fall_time)
                obs_r, info_r = env.reset(done_ids)
                obs, info = obs_r, info_r
                init_episode_stats(done_ids)
                # Frente H: episode end closes the open trial without a goal
                close_placements(done_ids, torch.zeros(len(done_ids), dtype=torch.bool))
                record_placements(done_ids)
                reach_still_t[done_ids] = 0.0
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
    nk = len(kick_stats["speed"])
    if nk > 0:
        speeds = np.asarray(kick_stats["speed"])
        angs = np.asarray(kick_stats["ang_err"])
        print("")
        print("-- kick impact metrics ({:d} kicks) --".format(nk))
        print("impact speed:   {:.2f} +- {:.2f} m/s (p90 {:.2f})".format(
            speeds.mean(), speeds.std(), np.percentile(speeds, 90)))
        print("speed buckets:  <1: {:.1%}   1-3: {:.1%}   >3 m/s: {:.1%}".format(
            float((speeds < 1).mean()), float(((speeds >= 1) & (speeds <= 3)).mean()),
            float((speeds > 3).mean())))
        print("angular error:  {:.1f} +- {:.1f} deg   <=20deg: {:.1%}   <=45deg: {:.1%}".format(
            angs.mean(), angs.std(), float((angs <= 20).mean()), float((angs <= 45).mean())))
        print("ball at impact: still {:.1%} / rolling {:.1%}".format(
            float(np.mean(kick_stats["on_still"])), 1.0 - float(np.mean(kick_stats["on_still"]))))
        feet = np.asarray(kick_stats["foot"])
        print("kicking foot:   left {:.1%} / right {:.1%}".format(
            float((feet == 0).mean()), float((feet == 1).mean())))
        if len(kick_stats["latency"]) > 0:
            print("engage latency: {:.2f} +- {:.2f} s (ball still & <{:.1f} m -> kick; {:d} kicks)".format(
                float(np.mean(kick_stats["latency"])), float(np.std(kick_stats["latency"])),
                ENGAGE_DIST, len(kick_stats["latency"])))
        if outcomes["total"] > 0:
            t = outcomes["total"]
            print("kick outcomes:  goal<={:.0f}s: {:.1%}   OOB<={:.0f}s: {:.1%}   fall<={:.0f}s: {:.1%}   (of {:d} kicks)".format(
                OUTCOME_GOAL_S, outcomes["goal"] / t, OUTCOME_OOB_S, outcomes["oob"] / t,
                OUTCOME_FALL_S, outcomes["fall"] / t, t))
    print("")
    print("-- style coverage (nearest demo clip per rollout window) --")
    rollout_obs = torch.cat(style_windows, dim=0)
    fracs = nearest_clip_fractions(rollout_obs, [d.cpu() for d in demos])
    for name, f in zip(clip_names, fracs):
        print("{:30s} {:.1%}".format(name, f))

    # ------------------- Frente H: paper-style breakdowns ---------------------
    print("")
    print("-- goal rate by ball-placement region (3x3 field grid) --")
    print("   cols: far from goal -> near goal | rows: y- -> y+ | goal%/trials")
    for r in range(GRID):
        cells = []
        for c in range(GRID):
            t = grid_total[r, c]
            rate = grid_goals[r, c] / t if t > 0 else float("nan")
            cells.append("{:5.1%}/{:<4d}".format(rate, t) if t > 0 else "   -  /0   ")
        print("   " + "  ".join(cells))
    total_trials = int(grid_total.sum())
    print("   overall: {:.1%} of {:d} ball placements ended in a goal".format(
        grid_goals.sum() / max(total_trials, 1), total_trials))
    print("")
    print("-- time to first kick vs approach angle at placement --")
    print("   (angle between robot->ball and ball->goal; 0 deg = straight behind)")
    for b in range(len(ANGLE_BINS) - 1):
        v = kick_time_by_angle[b]
        if len(v) > 0:
            print("   {:3.0f}-{:3.0f} deg: {:5.2f} +- {:4.2f} s  ({:d} placements)".format(
                ANGLE_BINS[b], ANGLE_BINS[b + 1], float(np.mean(v)), float(np.std(v)), len(v)))
        else:
            print("   {:3.0f}-{:3.0f} deg:   (no kicked placements)".format(
                ANGLE_BINS[b], ANGLE_BINS[b + 1]))
    print("==========================================================")
    return


if __name__ == "__main__":
    main()
