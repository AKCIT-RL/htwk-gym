"""GPU validation for TaskSoccerEnv (E2 gate): build, physics, obs, resets.

Run inside the mimickit-isaacgym container from the MimicKit root:
    python3.8 /workspace/htwk-gym/scripts/validate_soccer_env.py

Asserts (no eyeballing):
  1. env builds; obs = char obs + 7 task dims; disc obs dims unchanged.
  2. random-action rollout: no NaN/inf in obs/reward; ball stays near the
     ground and inside a sane bounding box.
  3. ball drop test: ball released above the ground settles at z ~ radius.
  4. resets place the ball inside the spawn region and away from the robot.
"""

import os
import sys

MIMICKIT_ROOT = "/workspace/MimicKit"
sys.path.insert(0, os.path.join(MIMICKIT_ROOT, "mimickit"))
os.chdir(MIMICKIT_ROOT)

import isaacgym  # noqa: F401  (must precede torch)
import torch

import envs.env_builder as env_builder
import envs.smp_env as smp_env

NUM_ENVS = 8
DEVICE = "cuda:0"
ROLLOUT_STEPS = 300
SOCCER_TASK_OBS_DIM = 12  # steer cmd (dir 2, speed 1, face 2), ball xy, goal xy, goal dir (cos, sin), ball mask


def build_env(env_file):
    env = env_builder.build_env(env_file=env_file,
                                engine_file="data/engines/isaac_gym_engine.yaml",
                                num_envs=NUM_ENVS, device=DEVICE, visualize=False)
    return env


def check(cond, msg):
    assert cond, msg
    print("PASS " + msg)


def main():
    steering_env = None  # only built if needed for the obs-dim comparison
    env = build_env("data/envs/mcwamp_g1_soccer_env.yaml")
    obs, _ = env.reset()

    # --- 1. dimensions -----------------------------------------------------
    char_obs_dim = smp_env.SMPEnv._compute_obs(env).shape[-1]
    task_dim = obs.shape[-1] - char_obs_dim
    check(task_dim == SOCCER_TASK_OBS_DIM,
          "task obs dims == {:d} (got {:d})".format(SOCCER_TASK_OBS_DIM, task_dim))

    disc_obs = env._disc_obs_buf
    check(disc_obs.shape[0] == NUM_ENVS,
          "disc obs shape sane: {}".format(tuple(disc_obs.shape)))
    check(torch.all(torch.isfinite(disc_obs)).item(), "disc obs finite")

    # --- 2. random-action rollout ------------------------------------------
    off = env._field_offset  # [N, 2] per-env field centers
    if NUM_ENVS > 1:
        min_pitch = torch.cdist(off, off).masked_fill(
            torch.eye(NUM_ENVS, device=DEVICE, dtype=torch.bool), float("inf")).min().item()
        check(min_pitch >= env._field_width + 2.0 * env._field_sep - 1e-3,
              "per-env fields laid out on a grid (min pitch {:.1f} m)".format(min_pitch))
    action_space = env.get_action_space()
    action_dim = action_space.shape[0]
    max_ball_speed = 0.0
    for i in range(ROLLOUT_STEPS):
        a = 2.0 * torch.rand([NUM_ENVS, action_dim], device=DEVICE) - 1.0
        obs, r, done, _ = env.step(a)
        assert torch.all(torch.isfinite(obs)).item(), "NaN/inf obs at step {:d}".format(i)
        assert torch.all(torch.isfinite(r)).item(), "NaN/inf reward at step {:d}".format(i)

        ball_pos = env._get_ball_pos()
        assert torch.all(ball_pos[:, 2] > -0.05).item(), "ball below ground at step {:d}".format(i)
        assert torch.all(ball_pos[:, 2] < 3.0).item(), "ball flying at step {:d}".format(i)
        assert torch.all(torch.abs(ball_pos[:, 0:2] - off) < 20.0).item(), "ball escaped at step {:d}".format(i)

        ball_vel = env._engine.get_root_vel(env._get_ball_id())
        max_ball_speed = max(max_ball_speed, torch.linalg.norm(ball_vel, dim=-1).max().item())

        done_ids = (done != 0).nonzero(as_tuple=False).flatten()
        if len(done_ids) > 0:
            env.reset(done_ids)
    check(True, "random rollout {:d} steps clean (max ball speed {:.2f} m/s)".format(
        ROLLOUT_STEPS, max_ball_speed))

    # --- 3. ball drop test ---------------------------------------------------
    env.reset()
    env_ids = torch.arange(NUM_ENVS, device=DEVICE, dtype=torch.long)
    ball_id = env._get_ball_id()
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
    for _ in range(90):  # 3 s at 30 Hz
        env.step(zero_a)
    ball_pos = env._get_ball_pos()
    ball_z = ball_pos[:, 2]
    check(torch.all(torch.abs(ball_z - env._ball_radius) < 0.05).item(),
          "dropped ball settled at z ~ r (z = {:.3f}..{:.3f})".format(
              ball_z.min().item(), ball_z.max().item()))
    ball_speed = torch.linalg.norm(env._engine.get_root_vel(ball_id), dim=-1)
    print("INFO ball speed 3 s after drop: {:.3f}..{:.3f} m/s".format(
        ball_speed.min().item(), ball_speed.max().item()))

    # --- 4. rolling behavior (E1 risk: no rolling resistance) ---------------
    env.reset()
    roll_pos = torch.zeros([NUM_ENVS, 3], device=DEVICE)
    roll_pos[:, 0] = -3.0
    roll_pos[:, 2] = env._ball_radius
    roll_pos[:, 0:2] += off
    roll_vel = torch.zeros([NUM_ENVS, 3], device=DEVICE)
    roll_vel[:, 0] = 3.0
    env._engine.set_root_pos(env_ids, ball_id, roll_pos)
    env._engine.set_root_rot(env_ids, ball_id, quat0)
    env._engine.set_root_vel(env_ids, ball_id, roll_vel)
    env._engine.set_root_ang_vel(env_ids, ball_id, zero3)
    max_moving_time = torch.zeros([NUM_ENVS], device=DEVICE)
    for _ in range(150):  # 5 s
        env.step(zero_a)
        max_moving_time = torch.maximum(max_moving_time, env._ball_moving_time)
    roll_speed = torch.linalg.norm(env._engine.get_root_vel(ball_id)[:, 0:2], dim=-1)
    roll_dist = (env._get_ball_pos()[:, 0] - off[:, 0]).max().item() - (-3.0)
    print("INFO ball speed 5 s after 3 m/s push: {:.3f}..{:.3f} m/s, dist {:.2f} m".format(
        roll_speed.min().item(), roll_speed.max().item(), roll_dist))
    check(roll_dist > 1.0, "pushed ball actually rolled ({:.2f} m)".format(roll_dist))

    # ball motion timers: moving-time armed while the ball rolled (episode
    # resets can legitimately zero the live timer, so track the running max);
    # the cleared-on-reset half is checked after test 5's reset
    check(bool((max_moving_time > 0.0).all().item()),
          "ball motion timers armed during the push (max {:.2f} s)".format(
              max_moving_time.max().item()))

    # --- 5. robot-ball collision ---------------------------------------------
    # Deterministic contact test: teleport the ball into overlap with the
    # robot's pelvis. With collisions on, depenetration kicks the ball out
    # (lateral velocity and/or blocked fall). Without collisions the ball is
    # in free fall: v_xy = 0 and v_z = -g*t after t seconds.
    env.reset()
    check(bool((env._ball_moving_time == 0.0).all().item())
          and bool((env._ball_still_time == 0.0).all().item()),
          "ball motion timers cleared on reset")
    char_id = env._get_char_id()
    root_pos = env._engine.get_root_pos(char_id)
    overlap = root_pos.clone()  # ball center = pelvis center
    env._engine.set_root_pos(env_ids, ball_id, overlap)
    env._engine.set_root_rot(env_ids, ball_id, quat0)
    env._engine.set_root_vel(env_ids, ball_id, zero3)
    env._engine.set_root_ang_vel(env_ids, ball_id, zero3)
    dt = env._engine.get_timestep()
    n_steps = 3
    for _ in range(n_steps):
        env.step(zero_a)
    vel = env._engine.get_root_vel(ball_id)
    free_fall_vz = -9.81 * n_steps * dt
    lateral_speed = torch.linalg.norm(vel[:, 0:2], dim=-1)
    deflected = (lateral_speed > 0.3) | (vel[:, 2] > 0.5 * free_fall_vz)
    check(bool(deflected.all().item()),
          "ball overlapping robot got depenetrated in all envs ({:d}/{:d})".format(
              int(deflected.sum().item()), NUM_ENVS))

    # --- 6. reset placement --------------------------------------------------
    for _ in range(5):
        env.reset()
        ball_pos = env._get_ball_pos()
        half_x = 0.5 * env._field_length - env._spawn_margin
        half_y = 0.5 * env._field_width - env._spawn_margin
        # the min-dist push can move the ball slightly past the margin box
        slack = env._ball_spawn_min_dist
        assert torch.all(torch.abs(ball_pos[:, 0] - off[:, 0]) <= half_x + slack).item(), "ball reset outside x range"
        assert torch.all(torch.abs(ball_pos[:, 1] - off[:, 1]) <= half_y + slack).item(), "ball reset outside y range"

        char_id = env._get_char_id()
        root_pos = env._engine.get_root_pos(char_id)
        d = torch.linalg.norm(ball_pos[:, 0:2] - root_pos[:, 0:2], dim=-1)
        assert torch.all(d > 0.9 * env._ball_spawn_min_dist).item(), \
            "ball reset too close to robot (d_min = {:.3f})".format(d.min().item())
        # rigid body state must be recomputed after the random placement:
        # stale key-body positions leak the (huge) field grid offsets into
        # the char obs and destroy the obs normalizer
        reset_obs = env._compute_obs()
        assert reset_obs.abs().max().item() < 50.0, \
            "reset obs blowup (max {:.1f}): stale rigid body state?".format(
                reset_obs.abs().max().item())
    check(True, "reset placement respects spawn region and min robot distance")
    check(True, "reset obs bounded (no stale key-body positions)")

    # near spawn: ball starts within reach of the robot (T1-style curriculum)
    if (env._ball_spawn_near):
        max_near = (env._ball_spawn_front_max ** 2 + env._ball_spawn_lateral ** 2) ** 0.5
        check(bool((d <= max_near + 1e-3).all().item()),
              "ball spawns near the robot (d_max = {:.2f} m <= {:.2f})".format(
                  d.max().item(), max_near))

    # steering command slots: unit target dir, speed in [0, max], zero near ball
    steer = reset_obs[:, char_obs_dim:char_obs_dim + 5]
    dir_norm = torch.linalg.norm(steer[:, 0:2], dim=-1)
    speed = steer[:, 2]
    exp_speed = torch.clamp(d - env._steer_stop_dist, min=0.0, max=env._steer_speed_max)
    check(bool(torch.all(torch.abs(dir_norm - 1.0) < 1e-3).item())
          and bool(torch.all(torch.abs(speed - exp_speed) < 1e-3).item()),
          "steering slots consistent (unit dir, speed = clamp(dist - stop, 0, max))")

    # ======================= E3: episode logic ===============================

    # --- 7. robot placement randomization ------------------------------------
    env.reset()
    char_id = env._get_char_id()
    root_pos = env._engine.get_root_pos(char_id)
    spread = (root_pos[:, 0:2] - off).std(dim=0)
    check(bool((spread > 0.5).all().item()),
          "robot spawns scattered over the field (std xy = {:.2f}, {:.2f} m)".format(
              spread[0].item(), spread[1].item()))
    # heading spread via root rot z component
    root_rot = env._engine.get_root_rot(char_id)
    check(root_rot[:, 2].std().item() > 0.2,
          "robot headings randomized (quat z std = {:.2f})".format(root_rot[:, 2].std().item()))

    # --- 8. goal event: episode ends (SUCC), robot kept, ball respawned ------
    env.reset()
    ball_id = env._get_ball_id()
    # park the robot away from the ball path
    park = env._engine.get_root_pos(char_id).clone()
    park[:, 0] = -3.0 + off[:, 0]
    park[:, 1] = off[:, 1]
    env._engine.set_root_pos(env_ids, char_id, park)

    shot_pos = torch.zeros([NUM_ENVS, 3], device=DEVICE)
    shot_pos[:, 0] = 6.5
    shot_pos[:, 2] = env._ball_radius
    shot_pos[:, 0:2] += off
    shot_vel = torch.zeros([NUM_ENVS, 3], device=DEVICE)
    shot_vel[:, 0] = 3.0
    env._engine.set_root_pos(env_ids, ball_id, shot_pos)
    env._engine.set_root_rot(env_ids, ball_id, quat0)
    env._engine.set_root_vel(env_ids, ball_id, shot_vel)
    env._engine.set_root_ang_vel(env_ids, ball_id, zero3)
    env._prev_ball_pos[:] = shot_pos

    goal_seen = torch.zeros([NUM_ENVS], device=DEVICE, dtype=torch.bool)
    goal_step_reward = torch.zeros([NUM_ENVS], device=DEVICE)
    done_at_goal = torch.zeros([NUM_ENVS], device=DEVICE, dtype=torch.long)
    soft_at_goal = torch.zeros([NUM_ENVS], device=DEVICE, dtype=torch.bool)
    max_root_moved = 0.0
    for _ in range(40):
        _, r, done, _ = env.step(zero_a)
        new_goals = env._goal_scored_buf & ~goal_seen
        goal_step_reward[new_goals] = r[new_goals]
        done_at_goal[new_goals] = done[new_goals].long()
        soft_at_goal[new_goals] = env._soft_done_buf[new_goals]
        goal_seen |= env._goal_scored_buf
        done_ids = (done != 0).nonzero(as_tuple=False).flatten()
        if len(done_ids) > 0:
            pose_before = env._engine.get_root_pos(char_id)[done_ids].clone()
            env.reset(done_ids)
            pose_after = env._engine.get_root_pos(char_id)[done_ids]
            moved = torch.linalg.norm(pose_after - pose_before, dim=-1).max().item()
            max_root_moved = max(max_root_moved, moved)
            t_after = env._time_buf[done_ids]
            check(bool((t_after == 0.0).all().item()),
                  "episode clock restarted on the ball-event reset")
        if bool(goal_seen.all().item()):
            break
    check(bool(goal_seen.all().item()), "goal detected in all envs")
    check(bool((done_at_goal == 2).all().item()),  # DoneFlags.SUCC
          "goal terminates the episode with SUCC (paper 4.1)")
    check(bool(soft_at_goal.all().item()),
          "goal done flagged as soft (ball-only reset)")
    check(max_root_moved < 1e-4,
          "robot state untouched by the ball-event reset (moved {:.2e} m)".format(
              max_root_moved))
    ball_pos = env._get_ball_pos()
    inside = (torch.abs(ball_pos[:, 0] - off[:, 0]) < 0.5 * env._field_length) & \
             (torch.abs(ball_pos[:, 1] - off[:, 1]) < 0.5 * env._field_width)
    check(bool(inside.all().item()), "ball respawned inside the field after goal")
    check(bool((goal_step_reward > 5.0).all().item()),
          "goal step paid the terminal goal reward (min {:.1f})".format(
              goal_step_reward.min().item()))

    # --- 9. out of bounds: episode ends (FAIL), no goal, robot kept ----------
    env.reset()
    park = env._engine.get_root_pos(char_id).clone()
    park[:, 0] = off[:, 0]
    park[:, 1] = off[:, 1]
    env._engine.set_root_pos(env_ids, char_id, park)
    out_pos = torch.zeros([NUM_ENVS, 3], device=DEVICE)
    out_pos[:, 1] = 4.3
    out_pos[:, 2] = env._ball_radius
    out_pos[:, 0:2] += off
    out_vel = torch.zeros([NUM_ENVS, 3], device=DEVICE)
    out_vel[:, 1] = 3.0
    env._engine.set_root_pos(env_ids, ball_id, out_pos)
    env._engine.set_root_rot(env_ids, ball_id, quat0)
    env._engine.set_root_vel(env_ids, ball_id, out_vel)
    env._engine.set_root_ang_vel(env_ids, ball_id, zero3)
    env._prev_ball_pos[:] = out_pos

    oob_seen = torch.zeros([NUM_ENVS], device=DEVICE, dtype=torch.bool)
    goal_flagged = torch.zeros([NUM_ENVS], device=DEVICE, dtype=torch.bool)
    done_at_oob = torch.zeros([NUM_ENVS], device=DEVICE, dtype=torch.long)
    soft_at_oob = torch.zeros([NUM_ENVS], device=DEVICE, dtype=torch.bool)
    for _ in range(40):
        _, _, done, _ = env.step(zero_a)
        new_oob = env._ball_oob_buf & ~oob_seen
        done_at_oob[new_oob] = done[new_oob].long()
        soft_at_oob[new_oob] = env._soft_done_buf[new_oob]
        oob_seen |= env._ball_oob_buf
        goal_flagged |= env._goal_scored_buf
        done_ids = (done != 0).nonzero(as_tuple=False).flatten()
        if len(done_ids) > 0:
            env.reset(done_ids)
        if bool(oob_seen.all().item()):
            break
    check(bool(oob_seen.all().item()), "sideline out detected in all envs")
    check(not bool(goal_flagged.any().item()), "sideline out never counted as goal")
    check(bool((done_at_oob == 1).all().item()),  # DoneFlags.FAIL
          "ball out terminates the episode with FAIL (paper 4.1)")
    check(bool(soft_at_oob.all().item()),
          "out done flagged as soft (ball-only reset)")
    ball_pos = env._get_ball_pos()
    inside = (torch.abs(ball_pos[:, 0] - off[:, 0]) < 0.5 * env._field_length) & \
             (torch.abs(ball_pos[:, 1] - off[:, 1]) < 0.5 * env._field_width)
    check(bool(inside.all().item()), "ball respawned inside the field after out")

    # --- 10. ball perturbations ----------------------------------------------
    env.reset()
    pos_before = env._get_ball_pos().clone()
    env._ball_perturb_times[:] = -1.0
    env.step(zero_a)
    pos_after = env._get_ball_pos()
    vel_after = env._engine.get_root_vel(ball_id)
    moved = torch.linalg.norm(pos_after[:, 0:2] - pos_before[:, 0:2], dim=-1) > 0.05
    pushed = torch.linalg.norm(vel_after[:, 0:2], dim=-1) > 0.5 * env._ball_perturb_speed_min
    check(bool((moved | pushed).all().item()),
          "forced perturbation teleported or pushed the ball in all envs")
    check(bool((env._ball_perturb_times > env._time_buf).all().item()),
          "perturbation timers resampled into the future")

    # --- 11. fall termination (FAIL) ------------------------------------------
    env.reset()
    fail_seen = torch.zeros([NUM_ENVS], device=DEVICE, dtype=torch.bool)
    for _ in range(120):  # 4 s of violent random actions
        a = 2.0 * torch.rand([NUM_ENVS, action_dim], device=DEVICE) - 1.0
        _, _, done, _ = env.step(a)
        fail_seen |= (done == 1)  # DoneFlags.FAIL
        done_ids = (done != 0).nonzero(as_tuple=False).flatten()
        if len(done_ids) > 0:
            env.reset(done_ids)
    check(bool(fail_seen.any().item()),
          "fall termination fires under violent random actions ({:d}/{:d} envs)".format(
              int(fail_seen.sum().item()), NUM_ENVS))

    # --- 12. time limit (TIME) -------------------------------------------------
    env.reset()
    dt = env._engine.get_timestep()
    env._timestep_buf[:] = int(60.0 / dt) + 1
    _, _, done, _ = env.step(zero_a)
    check(bool((done == 3).all().item()),  # DoneFlags.TIME
          "time limit terminates the episode at 60 s")

    # ======================= E4: aux reward stream ============================

    # --- 13. aux reward exposed and finite ------------------------------------
    env.reset()
    aux_seen_nonzero = False
    for i in range(60):
        a = 0.2 * (2.0 * torch.rand([NUM_ENVS, action_dim], device=DEVICE) - 1.0)
        _, _, done, info = env.step(a)
        assert "aux_reward" in info, "aux_reward missing from step info"
        aux = info["aux_reward"]
        assert torch.all(torch.isfinite(aux)).item(), "NaN/inf aux reward at step {:d}".format(i)
        aux_seen_nonzero = aux_seen_nonzero or bool((aux.abs() > 1e-6).any().item())
        done_ids = (done != 0).nonzero(as_tuple=False).flatten()
        if len(done_ids) > 0:
            env.reset(done_ids)
    check(aux_seen_nonzero, "aux reward stream alive and finite over 60 random steps")

    # --- 14. survival baseline -------------------------------------------------
    # right after a reset with mild actions, aux ~ survival_w + small penalties
    env.reset()
    _, _, _, info = env.step(zero_a)
    aux = info["aux_reward"]
    check(bool((aux <= env._reward_survival_w + 25.0).all().item())
          and bool((aux > env._reward_survival_w - 50.0).all().item()),
          "first-step aux reward near the survival baseline ({:.2f}..{:.2f})".format(
              aux.min().item(), aux.max().item()))

    # --- 15. fall termination pays the aux penalty ----------------------------
    env.reset()
    got_term_penalty = False
    for _ in range(120):
        a = 2.0 * torch.rand([NUM_ENVS, action_dim], device=DEVICE) - 1.0
        _, _, done, info = env.step(a)
        fail_mask = (done == 1)
        if bool(fail_mask.any().item()):
            aux_fail = info["aux_reward"][fail_mask]
            got_term_penalty = bool((aux_fail < 0.5 * env._reward_termination_w).all().item())
            break
    check(got_term_penalty, "fall step pays the aux termination penalty")

    # --- 16. stagnation penalty ------------------------------------------------
    env.reset()
    char_id = env._get_char_id()
    # RSI can give the root up to ~3 m/s from the motion clip, which can move
    # it past the stagnation threshold in a single step -- zero it out so the
    # robot genuinely does not move.
    zero_vel = torch.zeros([NUM_ENVS, 3], device=DEVICE)
    env._engine.set_root_vel(env_ids, char_id, zero_vel)
    env._engine.set_root_ang_vel(env_ids, char_id, zero_vel)
    root_pos_now = env._engine.get_root_pos(char_id).clone()
    env._stagnation_anchor_pos[:] = root_pos_now
    env._stagnation_anchor_time[:] = env._time_buf - 2.0  # window already elapsed
    _, _, _, info = env.step(zero_a)
    aux = info["aux_reward"]
    check(bool((aux < 0.5 * env._reward_stagnation_w).all().item()),
          "stagnation penalty fires when the robot has not moved for the window")
    check(bool((env._stagnation_anchor_time >= 0.0).all().item()),
          "stagnation anchor re-armed after firing")

    print("ALL SOCCER ENV E2+E3+E4 CHECKS PASSED")
    return


if __name__ == "__main__":
    main()
