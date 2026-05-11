"""Kick policy evaluator — multi-scenario, multi-metric.

Runs a trained kick policy through several evaluation scenarios, each varying
one factor while keeping others fixed. Reuses the env's built-in metrics
(`kick_detected`, `angular_error_at_kick`, `z_error_at_kick`,
`ball_crossed_ref`, `ball_ref_y_error`) and adds extended ones:

  * ball_speed_at_kick (m/s)
  * ball_travel_m (final ball displacement, XY)
  * steps_to_kick / time_to_kick_s
  * energy (sum of ||action||^2 over the rollout)
  * foot_kicked ("left"/"right", whichever foot is closest at kick step)
  * fell_before_kick / fell_after_kick / timed_out

Scenarios (--scenarios; "all" runs every one):
  * angles       — target angle varies (the original use case)
  * ball_pos     — initial ball (dx, dy) in robot frame varies
  * robot_yaw    — initial robot yaw vs. ball-forward direction varies
  * ball_vel     — initial ball rolling velocity (toward/perp/away) varies
  * distance     — kick target distance varies
  * disturb_push — external base push magnitude (N) mid-episode varies

Outputs (under <output_dir> or eval_results/<task>/<timestamp>/):
  <scenario>/attempts.csv
  <scenario>/summary.json
  all_scenarios_summary.json

Usage:
    python evaluate_kick.py --task T1/Kicking_Movement_Bica \\
        --checkpoint logs/T1/.../model_5000.pth \\
        --scenarios all --num_envs 60
"""

import isaacgym  # must be imported before torch

import argparse
import csv
import json
import os
import sys
import time
import types

import numpy as np
import torch


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
ALL_SCENARIOS = ["angles", "ball_pos", "robot_yaw", "ball_vel",
                 "distance", "disturb_push"]


def _parse_eval_args():
    p = argparse.ArgumentParser(
        add_help=False,
        description="Kick policy multi-scenario evaluator.",
    )
    p.add_argument("--scenarios", type=str, default="all",
                   help="Comma list or 'all'. Options: " + ",".join(ALL_SCENARIOS))
    p.add_argument("--num_envs", type=int, default=60,
                   help="Parallel envs shared across all scenarios.")
    p.add_argument("--max_steps", type=int, default=600,
                   help="Safety cap on rollout steps per scenario.")
    p.add_argument("--hit_angle_deg", type=float, default=20.0)
    p.add_argument("--hit_lateral_m", type=float, default=0.5)
    p.add_argument("--output_dir", type=str, default=None,
                   help="Default: eval_results/<task>/<timestamp>/")
    # angles
    p.add_argument("--angles_min_deg", type=float, default=-15.0)
    p.add_argument("--angles_max_deg", type=float, default=15.0)
    p.add_argument("--angles_n", type=int, default=6)
    # ball_pos
    p.add_argument("--ball_pos_dx_range", type=str, default="0.20,0.45")
    p.add_argument("--ball_pos_dy_range", type=str, default="-0.15,0.15")
    p.add_argument("--ball_pos_nx", type=int, default=4)
    p.add_argument("--ball_pos_ny", type=int, default=3)
    # robot_yaw
    p.add_argument("--robot_yaw_min_deg", type=float, default=-30.0)
    p.add_argument("--robot_yaw_max_deg", type=float, default=30.0)
    p.add_argument("--robot_yaw_n", type=int, default=5)
    # ball_vel
    p.add_argument("--ball_vel_speeds_mps", type=str, default="0.1,0.3,0.5",
                   help="Comma list of rolling ball speeds (m/s).")
    p.add_argument("--ball_vel_directions", type=str,
                   default="toward,perpendicular,away",
                   help="Subset of {toward, perpendicular, away}.")
    # distance
    p.add_argument("--distance_values_m", type=str, default="2,4,6,8,10")
    # disturb_push
    p.add_argument("--disturb_push_values_N", type=str, default="0,25,50,75,100")
    p.add_argument("--disturb_push_step", type=int, default=50,
                   help="Episode step at which push starts.")
    p.add_argument("--disturb_push_duration_steps", type=int, default=5)
    p.add_argument("--eval_help", action="store_true")
    args, remaining = p.parse_known_args()
    if args.eval_help:
        p.print_help()
        sys.exit(0)
    return args, remaining


EVAL_ARGS, REMAINING = _parse_eval_args()

if EVAL_ARGS.scenarios.strip().lower() == "all":
    SCENARIOS = list(ALL_SCENARIOS)
else:
    SCENARIOS = [s.strip() for s in EVAL_ARGS.scenarios.split(",") if s.strip()]
    for s in SCENARIOS:
        if s not in ALL_SCENARIOS:
            print(f"[eval] unknown scenario '{s}'. Valid: {ALL_SCENARIOS}",
                  file=sys.stderr)
            sys.exit(1)

TOTAL_ENVS = EVAL_ARGS.num_envs


def _ensure_arg(flag, value, argv):
    """Append --flag value if --flag not already present."""
    if not any(a == flag or a.startswith(flag + "=") for a in argv):
        argv.extend([flag, str(value)])


_ensure_arg("--num_envs", TOTAL_ENVS, REMAINING)
_ensure_arg("--headless", "True", REMAINING)
sys.argv = [sys.argv[0]] + REMAINING


# ----------------------------------------------------------------------------
# Build Runner (post isaacgym import).
# ----------------------------------------------------------------------------
from utils.runner import Runner  # noqa: E402

print(f"[eval] scenarios={SCENARIOS}  total_envs={TOTAL_ENVS}")
runner = Runner(test=True)
env = runner.env
device = env.device

# Disable per-step camera rendering — too slow under software Vulkan (lavapipe).
runner.cfg["viewer"]["record_video"] = False
env.cfg["viewer"]["record_video"] = False

from isaacgym import gymapi, gymtorch  # noqa: E402
from isaacgym.torch_utils import get_euler_xyz, quat_from_euler_xyz  # noqa: E402

assert env.num_envs == TOTAL_ENVS, \
    f"env.num_envs ({env.num_envs}) != requested ({TOTAL_ENVS})"

CONTROL_DT = float(env.dt)
print(f"[eval] control dt = {CONTROL_DT:.4f} s")

ORIG_RESAMPLE_KICK_TARGET = env._resample_kick_target
ORIG_PUSH_ROBOTS = getattr(env, "_push_robots", None)


def _disable_push(self):
    """No-op replacement for env._push_robots — keeps domain randomization quiet."""
    if hasattr(self, "pushing_forces"):
        self.pushing_forces.zero_()
    if hasattr(self, "pushing_torques"):
        self.pushing_torques.zero_()
    self.gym.apply_rigid_body_force_tensors(
        self.sim,
        gymtorch.unwrap_tensor(self.pushing_forces),
        gymtorch.unwrap_tensor(self.pushing_torques),
        gymapi.LOCAL_SPACE,
    )


def _restore_originals():
    env._resample_kick_target = ORIG_RESAMPLE_KICK_TARGET
    if ORIG_PUSH_ROBOTS is not None:
        env._push_robots = ORIG_PUSH_ROBOTS


def _push_root_states():
    env.gym.set_actor_root_state_tensor(
        env.sim, gymtorch.unwrap_tensor(env.root_states))


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def _parse_range(spec):
    a, b = (float(x) for x in spec.split(","))
    return a, b


def _parse_floats(spec):
    return [float(x) for x in spec.split(",") if x.strip()]


def _parse_strs(spec):
    return [x.strip() for x in spec.split(",") if x.strip()]


def _balanced_conditions(n_envs, k):
    """Return length-n_envs int array with values 0..k-1, balanced blocks."""
    return ((np.arange(n_envs) * k) // n_envs).astype(int)


def _build_target_patch(target_angles_rad, target_distances):
    """Closure for env._resample_kick_target using per-env angles and distance."""
    z_value = float(getattr(env, "ball_radius", 0.05))
    if not torch.is_tensor(target_angles_rad):
        target_angles_rad = torch.tensor(
            target_angles_rad, device=device, dtype=torch.float)
    if not torch.is_tensor(target_distances):
        target_distances = torch.full(
            (TOTAL_ENVS,), float(target_distances),
            device=device, dtype=torch.float)

    def _patch(self, env_ids):
        if isinstance(env_ids, torch.Tensor):
            idx = env_ids
        else:
            idx = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        angles = target_angles_rad[idx]
        robot_yaw = get_euler_xyz(self.root_states[idx, 0, 3:7])[2]
        target_angle_world = robot_yaw + angles
        ball_xy = self.root_states[idx, 1, 0:2]
        dist = target_distances[idx]
        self.kick_target_pos_world[idx, 0] = ball_xy[:, 0] + dist * torch.cos(target_angle_world)
        self.kick_target_pos_world[idx, 1] = ball_xy[:, 1] + dist * torch.sin(target_angle_world)
        self.kick_target_pos_world[idx, 2] = z_value

    return _patch


# ----------------------------------------------------------------------------
# Scenario builders
# ----------------------------------------------------------------------------
def _scenario_angles():
    k = EVAL_ARGS.angles_n
    angles_deg = np.linspace(EVAL_ARGS.angles_min_deg,
                             EVAL_ARGS.angles_max_deg, k)
    cond = _balanced_conditions(TOTAL_ENVS, k)
    labels = [f"angle={a:+.1f}deg" for a in angles_deg]
    dist_yaml = env.cfg.get("rewards", {}).get("kick_target_ref_distance", 8.0)

    def pre_reset(env_):
        angles_rad = torch.tensor(
            [angles_deg[c] * (np.pi / 180.0) for c in cond],
            device=device, dtype=torch.float)
        env_._resample_kick_target = types.MethodType(
            _build_target_patch(angles_rad, dist_yaml), env_)
        env_._push_robots = types.MethodType(_disable_push, env_)

    def post_reset(env_):
        env_._resample_kick_target(
            torch.arange(TOTAL_ENVS, device=device, dtype=torch.long))

    return dict(name="angles",
                description=f"Target angle from {EVAL_ARGS.angles_min_deg} to "
                            f"{EVAL_ARGS.angles_max_deg} deg in {k} steps.",
                condition_labels=labels, condition_per_env=cond,
                pre_reset_setup=pre_reset, post_reset_setup=post_reset)


def _scenario_ball_pos():
    dx_lo, dx_hi = _parse_range(EVAL_ARGS.ball_pos_dx_range)
    dy_lo, dy_hi = _parse_range(EVAL_ARGS.ball_pos_dy_range)
    nx, ny = EVAL_ARGS.ball_pos_nx, EVAL_ARGS.ball_pos_ny
    pairs = [(float(x), float(y))
             for x in np.linspace(dx_lo, dx_hi, nx)
             for y in np.linspace(dy_lo, dy_hi, ny)]
    k = len(pairs)
    cond = _balanced_conditions(TOTAL_ENVS, k)
    labels = [f"dx={x:+.2f},dy={y:+.2f}" for (x, y) in pairs]
    dist_yaml = env.cfg.get("rewards", {}).get("kick_target_ref_distance", 8.0)

    def pre_reset(env_):
        angles_rad = torch.zeros(TOTAL_ENVS, device=device, dtype=torch.float)
        env_._resample_kick_target = types.MethodType(
            _build_target_patch(angles_rad, dist_yaml), env_)
        env_._push_robots = types.MethodType(_disable_push, env_)

    def post_reset(env_):
        ball_radius = float(getattr(env_, "ball_radius", 0.05))
        dx_t = torch.tensor([pairs[c][0] for c in cond], device=device, dtype=torch.float)
        dy_t = torch.tensor([pairs[c][1] for c in cond], device=device, dtype=torch.float)
        yaws = get_euler_xyz(env_.root_states[:, 0, 3:7])[2]
        cy, sy = torch.cos(yaws), torch.sin(yaws)
        rx = env_.root_states[:, 0, 0]
        ry = env_.root_states[:, 0, 1]
        env_.root_states[:, 1, 0] = rx + dx_t * cy - dy_t * sy
        env_.root_states[:, 1, 1] = ry + dx_t * sy + dy_t * cy
        env_.root_states[:, 1, 2] = ball_radius
        env_.root_states[:, 1, 7:13] = 0.0
        _push_root_states()
        env_._resample_kick_target(
            torch.arange(TOTAL_ENVS, device=device, dtype=torch.long))

    return dict(name="ball_pos",
                description=f"Initial ball xy on {nx}x{ny} grid (robot frame).",
                condition_labels=labels, condition_per_env=cond,
                pre_reset_setup=pre_reset, post_reset_setup=post_reset)


def _scenario_robot_yaw():
    k = EVAL_ARGS.robot_yaw_n
    yaws_deg = np.linspace(EVAL_ARGS.robot_yaw_min_deg,
                           EVAL_ARGS.robot_yaw_max_deg, k)
    cond = _balanced_conditions(TOTAL_ENVS, k)
    labels = [f"yaw={y:+.1f}deg" for y in yaws_deg]
    dist_yaml = env.cfg.get("rewards", {}).get("kick_target_ref_distance", 8.0)

    def pre_reset(env_):
        angles_rad = torch.zeros(TOTAL_ENVS, device=device, dtype=torch.float)
        env_._resample_kick_target = types.MethodType(
            _build_target_patch(angles_rad, dist_yaml), env_)
        env_._push_robots = types.MethodType(_disable_push, env_)

    def post_reset(env_):
        yaws_rad = torch.tensor(
            [yaws_deg[c] * (np.pi / 180.0) for c in cond],
            device=device, dtype=torch.float)
        roll = torch.zeros_like(yaws_rad)
        pitch = torch.zeros_like(yaws_rad)
        quat = quat_from_euler_xyz(roll, pitch, yaws_rad)
        env_.root_states[:, 0, 3:7] = quat
        _push_root_states()
        env_._resample_kick_target(
            torch.arange(TOTAL_ENVS, device=device, dtype=torch.long))

    return dict(name="robot_yaw",
                description="Initial robot yaw with ball at default front pose.",
                condition_labels=labels, condition_per_env=cond,
                pre_reset_setup=pre_reset, post_reset_setup=post_reset)


def _scenario_ball_vel():
    speeds = _parse_floats(EVAL_ARGS.ball_vel_speeds_mps)
    dirs = _parse_strs(EVAL_ARGS.ball_vel_directions)
    valid_dirs = {"toward", "perpendicular", "away"}
    for d in dirs:
        if d not in valid_dirs:
            raise ValueError(f"Invalid ball_vel direction '{d}'. Valid: {valid_dirs}")
    pairs = [(s, d) for d in dirs for s in speeds]
    k = len(pairs)
    cond = _balanced_conditions(TOTAL_ENVS, k)
    labels = [f"{d}@{s:.2f}mps" for (s, d) in pairs]
    dist_yaml = env.cfg.get("rewards", {}).get("kick_target_ref_distance", 8.0)

    def pre_reset(env_):
        angles_rad = torch.zeros(TOTAL_ENVS, device=device, dtype=torch.float)
        env_._resample_kick_target = types.MethodType(
            _build_target_patch(angles_rad, dist_yaml), env_)
        env_._push_robots = types.MethodType(_disable_push, env_)

    def post_reset(env_):
        rx = env_.root_states[:, 0, 0]
        ry = env_.root_states[:, 0, 1]
        bx = env_.root_states[:, 1, 0]
        by = env_.root_states[:, 1, 1]
        # toward = (robot - ball) normalized; away = -toward; perp = 90° rot.
        dx = rx - bx
        dy = ry - by
        norm = torch.sqrt(dx * dx + dy * dy).clamp_min(1e-6)
        ux = dx / norm
        uy = dy / norm
        vx_new = torch.zeros(TOTAL_ENVS, device=device, dtype=torch.float)
        vy_new = torch.zeros(TOTAL_ENVS, device=device, dtype=torch.float)
        for i in range(TOTAL_ENVS):
            speed, direction = pairs[cond[i]]
            if direction == "toward":
                vx_new[i] = ux[i] * speed
                vy_new[i] = uy[i] * speed
            elif direction == "away":
                vx_new[i] = -ux[i] * speed
                vy_new[i] = -uy[i] * speed
            else:  # perpendicular
                vx_new[i] = -uy[i] * speed
                vy_new[i] = ux[i] * speed
        env_.root_states[:, 1, 7] = vx_new
        env_.root_states[:, 1, 8] = vy_new
        env_.root_states[:, 1, 9] = 0.0
        _push_root_states()
        env_._resample_kick_target(
            torch.arange(TOTAL_ENVS, device=device, dtype=torch.long))

    return dict(name="ball_vel",
                description="Initial ball rolling velocity (toward/perp/away).",
                condition_labels=labels, condition_per_env=cond,
                pre_reset_setup=pre_reset, post_reset_setup=post_reset)


def _scenario_distance():
    values = _parse_floats(EVAL_ARGS.distance_values_m)
    k = len(values)
    cond = _balanced_conditions(TOTAL_ENVS, k)
    labels = [f"dist={v:.1f}m" for v in values]

    def pre_reset(env_):
        angles_rad = torch.zeros(TOTAL_ENVS, device=device, dtype=torch.float)
        dist_per_env = torch.tensor(
            [values[c] for c in cond], device=device, dtype=torch.float)
        env_._resample_kick_target = types.MethodType(
            _build_target_patch(angles_rad, dist_per_env), env_)
        env_._push_robots = types.MethodType(_disable_push, env_)

    def post_reset(env_):
        env_._resample_kick_target(
            torch.arange(TOTAL_ENVS, device=device, dtype=torch.long))

    return dict(name="distance",
                description="Kick target distance.",
                condition_labels=labels, condition_per_env=cond,
                pre_reset_setup=pre_reset, post_reset_setup=post_reset)


def _scenario_disturb_push():
    forces = _parse_floats(EVAL_ARGS.disturb_push_values_N)
    k = len(forces)
    cond = _balanced_conditions(TOTAL_ENVS, k)
    labels = [f"push={f:.0f}N" for f in forces]
    dist_yaml = env.cfg.get("rewards", {}).get("kick_target_ref_distance", 8.0)
    push_step = EVAL_ARGS.disturb_push_step
    push_dur = EVAL_ARGS.disturb_push_duration_steps
    forces_per_env = torch.tensor(
        [forces[c] for c in cond], device=device, dtype=torch.float)

    def pre_reset(env_):
        angles_rad = torch.zeros(TOTAL_ENVS, device=device, dtype=torch.float)
        env_._resample_kick_target = types.MethodType(
            _build_target_patch(angles_rad, dist_yaml), env_)

        base_idx = getattr(env_, "base_indice", 0)

        def _custom_push(self):
            if not hasattr(self, "pushing_forces"):
                return
            self.pushing_forces.zero_()
            self.pushing_torques.zero_()
            within = (self.episode_length_buf >= push_step) & \
                     (self.episode_length_buf < push_step + push_dur)
            if within.any():
                # Apply along +x in body-local frame.
                self.pushing_forces[within, base_idx, 0] = forces_per_env[within]
            self.gym.apply_rigid_body_force_tensors(
                self.sim,
                gymtorch.unwrap_tensor(self.pushing_forces),
                gymtorch.unwrap_tensor(self.pushing_torques),
                gymapi.LOCAL_SPACE,
            )

        env_._push_robots = types.MethodType(_custom_push, env_)

    def post_reset(env_):
        env_._resample_kick_target(
            torch.arange(TOTAL_ENVS, device=device, dtype=torch.long))

    return dict(name="disturb_push",
                description=f"+x body push (N) applied at episode step "
                            f"[{push_step}, {push_step + push_dur}).",
                condition_labels=labels, condition_per_env=cond,
                pre_reset_setup=pre_reset, post_reset_setup=post_reset)


SCENARIO_BUILDERS = {
    "angles": _scenario_angles,
    "ball_pos": _scenario_ball_pos,
    "robot_yaw": _scenario_robot_yaw,
    "ball_vel": _scenario_ball_vel,
    "distance": _scenario_distance,
    "disturb_push": _scenario_disturb_push,
}


# ----------------------------------------------------------------------------
# Rollout / metrics
# ----------------------------------------------------------------------------
def run_rollout(scenario):
    name = scenario["name"]
    print(f"\n[eval] === Scenario: {name} ({scenario['description']}) ===")
    print(f"[eval] {len(scenario['condition_labels'])} conditions: "
          f"{scenario['condition_labels']}")

    _restore_originals()
    scenario["pre_reset_setup"](env)

    obs, _ = env.reset()
    obs = obs.to(runner.device)
    scenario["post_reset_setup"](env)

    N = TOTAL_ENVS
    recorded_kick = np.zeros(N, dtype=bool)
    recorded_cross = np.zeros(N, dtype=bool)
    finalized = np.zeros(N, dtype=bool)
    timed_out = np.zeros(N, dtype=bool)
    fell_before_kick = np.zeros(N, dtype=bool)
    fell_after_kick = np.zeros(N, dtype=bool)

    angular_err = np.full(N, np.nan, dtype=np.float32)
    z_err = np.full(N, np.nan, dtype=np.float32)
    lateral_err = np.full(N, np.nan, dtype=np.float32)
    ball_speed_at_kick = np.full(N, np.nan, dtype=np.float32)
    steps_to_kick = np.full(N, -1, dtype=np.int32)
    energy = np.zeros(N, dtype=np.float32)
    foot_kicked = np.full(N, "", dtype=object)

    ball_pos_initial = env.root_states[:, 1, 0:3].detach().cpu().numpy().copy()
    ball_pos_final = ball_pos_initial.copy()

    t0 = time.time()
    with torch.no_grad():
        for step in range(EVAL_ARGS.max_steps):
            dist = runner.model.act(obs)
            act = dist.loc
            energy += torch.sum(act ** 2, dim=-1).detach().cpu().numpy()

            obs, _, done, _ = env.step(act)
            obs = obs.to(runner.device)

            kick_now = env.kick_detected.detach().cpu().numpy()
            cross_now = env.ball_crossed_ref.detach().cpu().numpy()
            ang_now = env.angular_error_at_kick.detach().cpu().numpy()
            zerr_now = env.z_error_at_kick.detach().cpu().numpy()
            lat_now = env.ball_ref_y_error.detach().cpu().numpy()
            done_now = done.detach().cpu().numpy().astype(bool)
            timeout_now = (env.time_out_buf.detach().cpu().numpy().astype(bool)
                           if hasattr(env, "time_out_buf")
                           else np.zeros(N, dtype=bool))

            new_kicks = kick_now & ~recorded_kick & ~finalized
            if new_kicks.any():
                angular_err[new_kicks] = ang_now[new_kicks]
                z_err[new_kicks] = zerr_now[new_kicks]
                bvel = env.root_states[:, 1, 7:10].detach().cpu().numpy()
                ball_speed_at_kick[new_kicks] = np.linalg.norm(
                    bvel[new_kicks], axis=-1)
                if hasattr(env, "feet_pos"):
                    feet_pos = env.feet_pos.detach().cpu().numpy()  # (N, 2, 3)
                    bpos = env.root_states[:, 1, 0:3].detach().cpu().numpy()
                    d_left = np.linalg.norm(feet_pos[:, 0, :] - bpos, axis=-1)
                    d_right = np.linalg.norm(feet_pos[:, 1, :] - bpos, axis=-1)
                    for i in np.where(new_kicks)[0]:
                        foot_kicked[i] = "left" if d_left[i] <= d_right[i] else "right"
                steps_to_kick[new_kicks] = step
                recorded_kick[new_kicks] = True

            new_cross = cross_now & ~recorded_cross & ~finalized
            if new_cross.any():
                lateral_err[new_cross] = lat_now[new_cross]
                bpos = env.root_states[:, 1, 0:3].detach().cpu().numpy()
                ball_pos_final[new_cross] = bpos[new_cross]
                recorded_cross[new_cross] = True

            newly_done = done_now & ~finalized
            if newly_done.any():
                # Capture ball pos at done for envs that didn't cross.
                if (newly_done & ~recorded_cross).any():
                    bpos = env.root_states[:, 1, 0:3].detach().cpu().numpy()
                    pick = newly_done & ~recorded_cross
                    ball_pos_final[pick] = bpos[pick]
                this_timeout = newly_done & timeout_now
                this_fall = newly_done & ~timeout_now
                timed_out[this_timeout] = True
                fell_before_kick[this_fall & ~recorded_kick] = True
                fell_after_kick[this_fall & recorded_kick] = True
                finalized[newly_done] = True

            complete = finalized | (recorded_kick & recorded_cross)
            if complete.all():
                print(f"[eval]   all envs complete at step {step + 1}")
                break

            if (step + 1) % 200 == 0:
                print(f"[eval]   step={step + 1} "
                      f"kicks={int(recorded_kick.sum())}/{N} "
                      f"cross={int(recorded_cross.sum())}/{N} "
                      f"finalized={int(finalized.sum())}/{N}")

    elapsed = time.time() - t0
    print(f"[eval]   rollout finished in {elapsed:.1f}s")

    # Capture final ball pos for envs that neither crossed nor finalized.
    not_captured = ~finalized & ~recorded_cross
    if not_captured.any():
        bpos = env.root_states[:, 1, 0:3].detach().cpu().numpy()
        ball_pos_final[not_captured] = bpos[not_captured]

    ball_travel_m = np.linalg.norm(
        ball_pos_final[:, :2] - ball_pos_initial[:, :2], axis=-1)

    hit = (
        recorded_kick
        & recorded_cross
        & (np.nan_to_num(angular_err, nan=1e9) < EVAL_ARGS.hit_angle_deg)
        & (np.abs(np.nan_to_num(lateral_err, nan=1e9)) < EVAL_ARGS.hit_lateral_m)
    )

    return dict(
        elapsed=elapsed,
        condition_per_env=scenario["condition_per_env"],
        condition_labels=scenario["condition_labels"],
        recorded_kick=recorded_kick, recorded_cross=recorded_cross,
        finalized=finalized, timed_out=timed_out,
        fell_before_kick=fell_before_kick, fell_after_kick=fell_after_kick,
        angular_err=angular_err, z_err=z_err, lateral_err=lateral_err,
        ball_speed_at_kick=ball_speed_at_kick, ball_travel_m=ball_travel_m,
        steps_to_kick=steps_to_kick, energy=energy, foot_kicked=foot_kicked,
        hit=hit,
    )


# ----------------------------------------------------------------------------
# Aggregation / output
# ----------------------------------------------------------------------------
def _safe_stats(values, mask):
    sel = values[mask]
    if sel.dtype.kind == "f":
        sel = sel[~np.isnan(sel)]
    if sel.size == 0:
        return {"count": 0, "mean": None, "std": None}
    return {"count": int(sel.size),
            "mean": float(sel.mean()),
            "std": float(sel.std())}


def _time_to_kick_stats(steps_to_kick, mask):
    idx = np.where(mask & (steps_to_kick >= 0))[0]
    if idx.size == 0:
        return {"count": 0, "mean": None, "std": None}
    tvals = steps_to_kick[idx].astype(np.float64) * CONTROL_DT
    return {"count": int(tvals.size),
            "mean": float(tvals.mean()),
            "std": float(tvals.std())}


def aggregate(result):
    cond = result["condition_per_env"]
    labels = result["condition_labels"]
    N = len(cond)
    per_cond = []
    for c_idx, lab in enumerate(labels):
        m = cond == c_idx
        n = int(m.sum())
        if n == 0:
            continue
        foot_counts = {
            "left": int(((result["foot_kicked"] == "left") & m).sum()),
            "right": int(((result["foot_kicked"] == "right") & m).sum()),
            "unknown": int(((result["foot_kicked"] == "") & m
                            & result["recorded_kick"]).sum()),
        }
        per_cond.append({
            "condition_idx": c_idx,
            "condition_label": lab,
            "n_attempts": n,
            "n_kicks_detected": int(result["recorded_kick"][m].sum()),
            "n_crossed_ref": int(result["recorded_cross"][m].sum()),
            "n_timed_out": int(result["timed_out"][m].sum()),
            "n_fell_before_kick": int(result["fell_before_kick"][m].sum()),
            "n_fell_after_kick": int(result["fell_after_kick"][m].sum()),
            "n_hits": int(result["hit"][m].sum()),
            "hit_rate": float(result["hit"][m].mean()),
            "angular_error_deg": _safe_stats(result["angular_err"],
                                             m & result["recorded_kick"]),
            "z_error_deg": _safe_stats(result["z_err"],
                                       m & result["recorded_kick"]),
            "lateral_error_m": _safe_stats(result["lateral_err"],
                                           m & result["recorded_cross"]),
            "ball_speed_at_kick_mps": _safe_stats(result["ball_speed_at_kick"],
                                                  m & result["recorded_kick"]),
            "ball_travel_m": _safe_stats(result["ball_travel_m"], m),
            "time_to_kick_s": _time_to_kick_stats(result["steps_to_kick"], m),
            "energy_act_sq": _safe_stats(result["energy"], m),
            "foot_kicked_counts": foot_counts,
        })

    all_mask = np.ones(N, dtype=bool)
    overall = {
        "n_attempts": N,
        "n_kicks_detected": int(result["recorded_kick"].sum()),
        "n_crossed_ref": int(result["recorded_cross"].sum()),
        "n_timed_out": int(result["timed_out"].sum()),
        "n_fell_before_kick": int(result["fell_before_kick"].sum()),
        "n_fell_after_kick": int(result["fell_after_kick"].sum()),
        "n_hits": int(result["hit"].sum()),
        "hit_rate": float(result["hit"].mean()),
        "angular_error_deg": _safe_stats(result["angular_err"], result["recorded_kick"]),
        "z_error_deg": _safe_stats(result["z_err"], result["recorded_kick"]),
        "lateral_error_m": _safe_stats(result["lateral_err"], result["recorded_cross"]),
        "ball_speed_at_kick_mps": _safe_stats(result["ball_speed_at_kick"],
                                              result["recorded_kick"]),
        "ball_travel_m": _safe_stats(result["ball_travel_m"], all_mask),
        "time_to_kick_s": _time_to_kick_stats(result["steps_to_kick"], all_mask),
        "energy_act_sq": _safe_stats(result["energy"], all_mask),
        "foot_kicked_counts": {
            "left": int((result["foot_kicked"] == "left").sum()),
            "right": int((result["foot_kicked"] == "right").sum()),
            "unknown": int(((result["foot_kicked"] == "")
                            & result["recorded_kick"]).sum()),
        },
    }
    return per_cond, overall


def _fmt(stats):
    if stats["count"] == 0 or stats["mean"] is None:
        return "    n/a   "
    return f"{stats['mean']:6.2f}±{stats['std']:5.2f}"


def write_scenario_artifacts(scenario, result, per_cond, overall, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "attempts.csv")
    cond = result["condition_per_env"]
    labels = result["condition_labels"]
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "env_idx", "condition_idx", "condition_label",
            "kick_detected", "ball_crossed_ref",
            "timed_out", "fell_before_kick", "fell_after_kick",
            "angular_error_deg", "z_error_deg", "lateral_error_m",
            "ball_speed_at_kick_mps", "ball_travel_m",
            "steps_to_kick", "time_to_kick_s",
            "energy_act_sq", "foot_kicked", "hit",
        ])
        for i in range(len(cond)):
            stk = int(result["steps_to_kick"][i])
            stk_s = "" if stk < 0 else str(stk)
            tk_s = "" if stk < 0 else f"{stk * CONTROL_DT:.3f}"
            w.writerow([
                i, int(cond[i]), labels[cond[i]],
                int(result["recorded_kick"][i]), int(result["recorded_cross"][i]),
                int(result["timed_out"][i]),
                int(result["fell_before_kick"][i]),
                int(result["fell_after_kick"][i]),
                "" if np.isnan(result["angular_err"][i]) else f"{result['angular_err'][i]:.4f}",
                "" if np.isnan(result["z_err"][i]) else f"{result['z_err'][i]:.4f}",
                "" if np.isnan(result["lateral_err"][i]) else f"{result['lateral_err'][i]:.4f}",
                "" if np.isnan(result["ball_speed_at_kick"][i]) else f"{result['ball_speed_at_kick'][i]:.4f}",
                f"{result['ball_travel_m'][i]:.4f}",
                stk_s, tk_s,
                f"{result['energy'][i]:.4f}",
                str(result["foot_kicked"][i]),
                int(result["hit"][i]),
            ])
    summary = {
        "scenario": scenario["name"],
        "description": scenario["description"],
        "per_condition": per_cond,
        "overall": overall,
        "elapsed_s": result["elapsed"],
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def print_scenario_table(scenario, per_cond, overall):
    print()
    print(f"--- Scenario: {scenario['name']} ---")
    hdr = (f"{'idx':>3} {'condition':>22} {'n':>3} "
           f"{'hit%':>5} {'kick%':>5} {'fall%':>5} "
           f"{'angErr°':>13} {'latErr_m':>13} "
           f"{'bSpd_mps':>13} {'travel_m':>13} {'t_kick_s':>13} {'L/R':>9}")
    print(hdr)
    print("-" * len(hdr))
    for row in per_cond:
        n = row["n_attempts"]
        fall_count = row["n_fell_before_kick"] + row["n_fell_after_kick"]
        fc = row["foot_kicked_counts"]
        lr = f"{fc['left']}/{fc['right']}"
        print(f"{row['condition_idx']:>3d} {row['condition_label']:>22} "
              f"{n:>3d} "
              f"{100 * row['hit_rate']:>4.0f}% "
              f"{100 * row['n_kicks_detected'] / n:>4.0f}% "
              f"{100 * fall_count / n:>4.0f}% "
              f"{_fmt(row['angular_error_deg']):>13} "
              f"{_fmt(row['lateral_error_m']):>13} "
              f"{_fmt(row['ball_speed_at_kick_mps']):>13} "
              f"{_fmt(row['ball_travel_m']):>13} "
              f"{_fmt(row['time_to_kick_s']):>13} "
              f"{lr:>9}")
    print("-" * len(hdr))
    n = overall["n_attempts"]
    fall_count = overall["n_fell_before_kick"] + overall["n_fell_after_kick"]
    fc = overall["foot_kicked_counts"]
    lr = f"{fc['left']}/{fc['right']}"
    print(f"{'-':>3} {'OVERALL':>22} "
          f"{n:>3d} "
          f"{100 * overall['hit_rate']:>4.0f}% "
          f"{100 * overall['n_kicks_detected'] / n:>4.0f}% "
          f"{100 * fall_count / n:>4.0f}% "
          f"{_fmt(overall['angular_error_deg']):>13} "
          f"{_fmt(overall['lateral_error_m']):>13} "
          f"{_fmt(overall['ball_speed_at_kick_mps']):>13} "
          f"{_fmt(overall['ball_travel_m']):>13} "
          f"{_fmt(overall['time_to_kick_s']):>13} "
          f"{lr:>9}")


# ----------------------------------------------------------------------------
# Main loop
# ----------------------------------------------------------------------------
task_name = runner.cfg["basic"]["task"].replace("/", "_")
checkpoint = runner.cfg["basic"].get("checkpoint", "")
if EVAL_ARGS.output_dir is None:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_root = os.path.join("eval_results", task_name, timestamp)
else:
    out_root = EVAL_ARGS.output_dir
os.makedirs(out_root, exist_ok=True)
print(f"[eval] output_dir = {out_root}")

all_summaries = {}
for sname in SCENARIOS:
    scenario = SCENARIO_BUILDERS[sname]()
    result = run_rollout(scenario)
    per_cond, overall = aggregate(result)
    write_scenario_artifacts(
        scenario, result, per_cond, overall, os.path.join(out_root, sname))
    print_scenario_table(scenario, per_cond, overall)
    all_summaries[sname] = {
        "description": scenario["description"],
        "overall": overall,
        "per_condition": per_cond,
    }

top = {
    "task": runner.cfg["basic"]["task"],
    "checkpoint": checkpoint,
    "num_envs": TOTAL_ENVS,
    "max_steps": EVAL_ARGS.max_steps,
    "hit_thresholds": {"angle_deg": EVAL_ARGS.hit_angle_deg,
                       "lateral_m": EVAL_ARGS.hit_lateral_m},
    "scenarios": all_summaries,
}
top_path = os.path.join(out_root, "all_scenarios_summary.json")
with open(top_path, "w") as f:
    json.dump(top, f, indent=2)

print(f"\n[eval] all scenarios done. Top-level summary: {top_path}")
