"""GPU gate for the Frente E virtual-perception pipeline (paper section 9).

Runs inside the mimickit-isaacgym docker image:
  1. env builds with perception on; obs finite; contract unchanged (249 dims).
  2. mask drops to 0 within ~period+latency after teleporting the ball far
     behind the robot (FOV + range dropout).
  3. perceived ball obs are noisy: std of (perceived - true) matches
     sigma = 0.124 d + 0.149 within tolerance at a known distance.
  4. latency: after teleporting the ball to a new spot (in view), the
     perceived obs keeps the OLD position for at least ~latency seconds.
  5. capture frequency: perceived obs changes at ~25 Hz, not every step.
  6. perception disabled -> mask constant 1 and ball obs exact.
"""

import os
import sys

MIMICKIT_ROOT = "/workspace/MimicKit"
sys.path.insert(0, os.path.join(MIMICKIT_ROOT, "mimickit"))
os.chdir(MIMICKIT_ROOT)

import isaacgym  # noqa: F401  (must precede torch)
import numpy as np
import torch

import envs.env_builder as env_builder

DEVICE = "cuda:0"
NUM_ENVS = 16
ENV_CFG = "data/envs/mcwamp_g1_soccer_env_e.yaml"
ENV_CFG_OFF = "data/envs/mcwamp_g1_soccer_env_c.yaml"
ENGINE_CFG = "data/engines/isaac_gym_engine_uneven.yaml"

CHAR_OBS = 237
STEER = 5
BALL_XY = slice(CHAR_OBS + STEER, CHAR_OBS + STEER + 2)   # local ball (x, y)
MASK = CHAR_OBS + STEER + 6                               # ball mask slot

failures = []


def check(name, ok, detail=""):
    tag = "PASS" if ok else "FAIL"
    print("{:s} {:s}{:s}".format(tag, name, " | " + detail if detail else ""))
    if (not ok):
        failures.append(name)


def freeze_task(env):
    """Disable soft events (perturb/push/OOB ball reset) but KEEP the
    perception tick, so the probes below control the ball position."""
    env._update_ball_perturb = lambda: None
    env._update_char_push = lambda: None
    env._reset_ball = lambda env_ids, near=True: None  # OOB soft reset off
    env._ball_perturb_times[:] = 1.0e9
    env._char_push_times[:] = 1.0e9


def steps(env, n):
    adim = env.get_action_space().shape[0]
    obs = None
    for _ in range(n):
        obs, r, done, _ = env.step(torch.zeros([env.get_num_envs(), adim], device=DEVICE))
    return obs


def pin_ball(env, pos_xy):
    """Teleport the ball of every env to a fixed world xy (zero velocity)."""
    ball_id = env._get_ball_id()
    ids = torch.arange(NUM_ENVS, device=DEVICE)
    bp = env._engine.get_root_pos(ball_id)
    bp[:, 0:2] = pos_xy
    bp[:, 2] = env._ball_radius
    env._engine.set_root_pos(ids, ball_id, bp)
    env._engine.set_root_vel(ids, ball_id, torch.zeros([NUM_ENVS, 3], device=DEVICE))


def drive(env, n, dt, record=False):
    """Advance ONLY the perception state machine (no physics): robots with
    zero actions fall over within a couple of seconds, which would corrupt
    the statistical probes below."""
    trace = []
    for _ in range(n):
        env._time_buf += dt
        env._update_perception()
        if (record):
            trace.append(env._percep_ball_pos.clone())
    return trace


def main():
    env = env_builder.build_env(env_file=ENV_CFG, engine_file=ENGINE_CFG,
                                num_envs=NUM_ENVS, device=DEVICE, visualize=False)
    obs, _ = env.reset()
    freeze_task(env)
    dt = env._engine.get_timestep()

    # --- 1. build + contract -------------------------------------------------
    check("env builds with virtual perception on", torch.all(torch.isfinite(obs)).item())
    check("obs contract unchanged (249 dims)", obs.shape[1] == 249,
          "got {:d}".format(obs.shape[1]))
    check("mask starts at 1 after reset", torch.all(obs[:, MASK] == 1.0).item())

    # --- 2. integration: real steps stay finite, mask mirrors valid flag ------
    adim = env.get_action_space().shape[0]
    for _ in range(10):
        obs, r, done, _ = env.step(torch.zeros([NUM_ENVS, adim], device=DEVICE))
    check("obs finite after 10 real steps", torch.all(torch.isfinite(obs)).item())
    check("obs mask mirrors the perception valid flag",
          torch.equal(obs[:, MASK], env._percep_ball_valid.float()))

    # --- 3. statistical probes on the state machine (no physics) --------------
    env.reset()
    char_id = env._get_char_id()
    root0 = env._engine.get_root_pos(char_id)[:, 0:2].clone()

    # noise + frequency: ball pinned 3 m ahead of each robot's heading so it
    # is inside the FOV regardless of the random spawn yaw
    root_rot = env._engine.get_root_rot(char_id)
    import util.torch_util as torch_util
    heading = torch_util.calc_heading(root_rot)
    fwd = torch.stack([torch.cos(heading), torch.sin(heading)], dim=-1)
    tgt = root0 + 3.0 * fwd
    pin_ball(env, tgt)
    trace = drive(env, 90, dt, record=True)  # 3 s
    changes = 0.0
    for a, b in zip(trace[10:-1], trace[11:]):
        changes += (torch.linalg.norm(b - a, dim=-1) > 1e-6).float().sum().item()
    rate = changes / (NUM_ENVS * (len(trace) - 11) * dt)
    # effective delivered-valid rate = 25.36 Hz captures x 90% detection,
    # minus occasional same-step delivery collisions from latency jitter
    check("valid measurements delivered at ~20 Hz", 15.0 <= rate <= 26.0,
          "{:.1f} Hz (25.36 x 0.9 = 22.8 nominal)".format(rate))

    err = torch.stack(trace[10:], dim=0) - tgt.unsqueeze(0)
    std = err.reshape(-1, 2).std(dim=0).mean().item()
    expect = 0.124 * 3.0 + 0.149  # 0.521
    check("noise std matches 0.124*d+0.149 at d=3", 0.7 * expect <= std <= 1.4 * expect,
          "std {:.3f} vs {:.3f}".format(std, expect))
    check("detections stay valid at d=3 in FOV",
          env._percep_ball_valid.float().mean().item() >= 0.5,
          "valid {:.0%}".format(env._percep_ball_valid.float().mean().item()))

    # latency: move the pinned ball +2 m sideways (still in view); for the
    # first ~66 ms (< latency 116 ms) every delivery still comes from
    # captures of the OLD spot, so the perceived pos must stay closer to the
    # old target than to the new one (noise-robust criterion)
    tgt2 = tgt + 2.0 * torch.stack([-fwd[:, 1], fwd[:, 0]], dim=-1)
    pin_ball(env, tgt2)
    drive(env, 2, dt)  # 66 ms < latency 116 ms
    d_old = torch.linalg.norm(env._percep_ball_pos - tgt, dim=-1)
    d_new = torch.linalg.norm(env._percep_ball_pos - tgt2, dim=-1)
    still_old = d_old < d_new
    check("latency holds the old measurement for ~2 steps",
          still_old.float().mean().item() >= 0.9,
          "held {:.0%}".format(still_old.float().mean().item()))
    drive(env, 10, dt)  # ~330 ms >> latency
    caught = torch.linalg.norm(env._percep_ball_pos - tgt2, dim=-1) < 1.5
    check("perception catches up after the latency",
          caught.float().mean().item() >= 0.9,
          "caught {:.0%}".format(caught.float().mean().item()))

    # --- 4. dropout: ball pinned 20 m behind the robot (out of range) ---------
    far = root0 - 20.0 * fwd
    pin_ball(env, far)
    drive(env, 12, dt)
    masked = ~env._percep_ball_valid
    check("mask drops when ball is far behind",
          masked.float().mean().item() >= 0.9,
          "masked {:.0%}".format(masked.float().mean().item()))
    # ...and the held position is the LAST valid one, not the far ball
    held_near = torch.linalg.norm(env._percep_ball_pos - tgt2, dim=-1) < 3.0
    check("zero-order hold keeps the last valid position",
          held_near.float().mean().item() >= 0.9,
          "held {:.0%}".format(held_near.float().mean().item()))

    # --- 6. perception off: exact obs, mask 1 --------------------------------
    env_off = env_builder.build_env(env_file=ENV_CFG_OFF, engine_file=ENGINE_CFG,
                                    num_envs=NUM_ENVS, device=DEVICE, visualize=False)
    obs_off, _ = env_off.reset()
    check("perception off -> mask constant 1", torch.all(obs_off[:, MASK] == 1.0).item())
    ball_pos = env_off._get_ball_pos()
    root_pos = env_off._engine.get_root_pos(env_off._get_char_id())
    dist_true = torch.linalg.norm(ball_pos[:, 0:2] - root_pos[:, 0:2], dim=-1)
    dist_obs = torch.linalg.norm(obs_off[:, BALL_XY], dim=-1)
    check("perception off -> ball obs exact",
          torch.allclose(dist_true, dist_obs, atol=1e-3),
          "max err {:.4f}".format((dist_true - dist_obs).abs().max().item()))

    print("=" * 58)
    if (failures):
        print("FAILURES: {}".format(failures))
        sys.exit(1)
    print("ALL SOCCER PERCEPTION (FRENTE E) CHECKS PASSED")


if (__name__ == "__main__"):
    main()
