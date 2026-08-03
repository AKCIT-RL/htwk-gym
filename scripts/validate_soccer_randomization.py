"""GPU validation for Frente C (domain randomization) + Frente D (steer anneal).

Run inside the mimickit-isaacgym container from the MimicKit root:
    python3.8 /workspace/htwk-gym/scripts/validate_soccer_randomization.py

Asserts (no eyeballing):
  1. env builds on the uneven trimesh ground with the C config; extent injected.
  2. per-env ball mass/friction/restitution actually differ and stay in range.
  3. per-env robot mass differs across envs (base mass + CoM randomization).
  4. steering anneal: step schedule zeroes the steering obs block; disabled
     schedule leaves it alive.
  5. random-action rollout on bumps: finite obs/rewards, ball stays near the
     ground, robots do not fall through the mesh.
  6. ball drop settles at z ~ radius +- bump amplitude.
  7. robot velocity push fires, changes the root velocity and re-arms.
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

NUM_ENVS = 8
DEVICE = "cuda:0"
ROLLOUT_STEPS = 300


def check(cond, msg):
    assert cond, msg
    print("PASS " + msg)


def main():
    env = env_builder.build_env(env_file="data/envs/mcwamp_g1_soccer_env_c.yaml",
                                engine_file="data/engines/isaac_gym_engine_uneven.yaml",
                                num_envs=NUM_ENVS, device=DEVICE, visualize=False)
    obs, _ = env.reset()
    check(torch.all(torch.isfinite(obs)).item(), "env builds on uneven ground, obs finite")

    amp = env._ground_z_offset
    check(amp > 0.0, "spawn z clearance set from ground.random_height ({:.3f} m)".format(amp))

    # --- 2. ball randomization ------------------------------------------------
    ball_id = env._get_ball_id()
    ball_masses = np.array([env._engine.calc_obj_mass(i, ball_id) for i in range(NUM_ENVS)])
    ratio = ball_masses.max() / ball_masses.min()
    check(ratio > 1.02, "ball masses differ across envs (ratio {:.3f})".format(ratio))
    check(ratio <= 1.3 / 0.7 + 1e-3,
          "ball mass spread within [0.7,1.3] scaling (ratio {:.3f})".format(ratio))

    gym = env._engine._gym
    frictions, restitutions = [], []
    for i in range(NUM_ENVS):
        props = gym.get_actor_rigid_shape_properties(env._engine.get_env(i), ball_id)
        frictions.append(props[0].friction)
        restitutions.append(props[0].restitution)
    frictions, restitutions = np.array(frictions), np.array(restitutions)
    check(frictions.std() > 1e-3 and np.all(frictions >= 0.2 - 1e-4)
          and np.all(frictions <= 1.2 + 1e-4),
          "ball friction randomized in [0.2, 1.2] ({:.2f}..{:.2f})".format(
              frictions.min(), frictions.max()))
    check(restitutions.std() > 1e-3 and np.all(restitutions >= 0.1 - 1e-4)
          and np.all(restitutions <= 0.9 + 1e-4),
          "ball restitution randomized in [0.1, 0.9] ({:.2f}..{:.2f})".format(
              restitutions.min(), restitutions.max()))

    # --- 3. robot randomization -------------------------------------------------
    char_id = env._get_char_id()
    char_masses = np.array([env._engine.calc_obj_mass(i, char_id) for i in range(NUM_ENVS)])
    cratio = char_masses.max() / char_masses.min()
    check(cratio > 1.01, "robot masses differ across envs (ratio {:.3f})".format(cratio))
    check(cratio < 1.6, "robot mass spread sane (ratio {:.3f})".format(cratio))
    cfr = []
    for i in range(NUM_ENVS):
        props = gym.get_actor_rigid_shape_properties(env._engine.get_env(i), char_id)
        cfr.append(props[0].friction)
    cfr = np.array(cfr)
    check(cfr.std() > 1e-3 and np.all(cfr >= 0.1 - 1e-4) and np.all(cfr <= 2.0 + 1e-4),
          "robot friction randomized in [0.1, 2.0] ({:.2f}..{:.2f})".format(cfr.min(), cfr.max()))

    # --- 4. steering anneal -----------------------------------------------------
    import envs.smp_env as smp_env
    char_obs_dim = smp_env.SMPEnv._compute_obs(env).shape[-1]
    obs_full = env._compute_obs()
    steer_block = obs_full[:, char_obs_dim:char_obs_dim + 5]
    check(bool((steer_block.abs().sum() > 0).item()),
          "steering obs alive with anneal disabled")
    env._steer_anneal_start_samples = 0.0
    env._steer_anneal_end_samples = 0.0
    obs_zero = env._compute_obs()
    steer_zero = obs_zero[:, char_obs_dim:char_obs_dim + 5]
    check(bool((steer_zero.abs().max() == 0).item()),
          "steering obs zeroed by the step schedule (eval config)")
    env._steer_anneal_start_samples = -1.0
    env._steer_anneal_end_samples = -1.0

    # --- 5. rollout on bumps ------------------------------------------------------
    env.reset()
    off = env._field_offset
    action_dim = env.get_action_space().shape[0]
    min_root_z = 10.0
    for i in range(ROLLOUT_STEPS):
        a = 2.0 * torch.rand([NUM_ENVS, action_dim], device=DEVICE) - 1.0
        obs, r, done, _ = env.step(a)
        assert torch.all(torch.isfinite(obs)).item(), "NaN/inf obs at step {:d}".format(i)
        assert torch.all(torch.isfinite(r)).item(), "NaN/inf reward at step {:d}".format(i)
        ball_pos = env._get_ball_pos()
        assert torch.all(ball_pos[:, 2] > -0.05 - amp).item(), \
            "ball below ground at step {:d}".format(i)
        assert torch.all(ball_pos[:, 2] < 3.0).item(), "ball flying at step {:d}".format(i)
        assert torch.all(torch.abs(ball_pos[:, 0:2] - off) < 20.0).item(), \
            "ball escaped at step {:d}".format(i)
        root_z = env._engine.get_root_pos(char_id)[:, 2]
        min_root_z = min(min_root_z, root_z.min().item())
        assert torch.all(root_z > -0.05 - amp).item(), \
            "robot fell through the mesh at step {:d}".format(i)
        done_ids = (done != 0).nonzero(as_tuple=False).flatten()
        if len(done_ids) > 0:
            env.reset(done_ids)
    check(True, "random rollout {:d} steps clean on bumps (min root z {:.3f})".format(
        ROLLOUT_STEPS, min_root_z))

    # --- 6. ball drop on bumps -----------------------------------------------------
    env.reset()
    env_ids = torch.arange(NUM_ENVS, device=DEVICE, dtype=torch.long)
    drop_pos = torch.zeros([NUM_ENVS, 3], device=DEVICE)
    drop_pos[:, 0] = 3.0
    drop_pos[:, 2] = 1.0
    drop_pos[:, 0:2] += off
    zero3 = torch.zeros([NUM_ENVS, 3], device=DEVICE)
    quat0 = torch.zeros([NUM_ENVS, 4], device=DEVICE)
    quat0[:, 3] = 1.0
    env._engine.set_root_pos(env_ids, ball_id, drop_pos)
    env._engine.set_root_rot(env_ids, ball_id, quat0)
    env._engine.set_root_vel(env_ids, ball_id, zero3)
    env._engine.set_root_ang_vel(env_ids, ball_id, zero3)
    zero_a = torch.zeros([NUM_ENVS, action_dim], device=DEVICE)
    for _ in range(150):  # 5 s (random restitution can bounce longer)
        env.step(zero_a)
    ball_z = env._get_ball_pos()[:, 2]
    check(torch.all(torch.abs(ball_z - env._ball_radius) < 0.05 + amp).item(),
          "dropped ball settled at z ~ r +- bumps (z = {:.3f}..{:.3f})".format(
              ball_z.min().item(), ball_z.max().item()))

    # --- 7. robot push ---------------------------------------------------------------
    env.reset()
    vel_std_orig = env._char_push_vel_std
    env._char_push_vel_std = 5.0  # exaggerate so dynamics noise cannot mask it
    env._char_push_times[:] = 0.0  # force an immediate trigger
    env.step(zero_a)  # trigger + queued root-state write
    env.step(zero_a)  # applied
    planar_speed = torch.linalg.norm(env._engine.get_root_vel(char_id)[:, 0:2], dim=-1)
    check(planar_speed.median().item() > 1.0,
          "push changed root velocity (median planar speed {:.2f} m/s)".format(
              planar_speed.median().item()))
    check(bool((env._char_push_times > env._time_buf).all().item()),
          "push timers re-armed into the future")
    env._char_push_vel_std = vel_std_orig

    print("ALL SOCCER RANDOMIZATION C+D CHECKS PASSED")
    return


if __name__ == "__main__":
    main()
