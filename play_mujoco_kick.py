from typing import Optional
"""Play or evaluate a trained kick policy inside MuJoCo.

Two modes:

* **Viewer mode (default)** — opens `mujoco.viewer` and lets you drive the kick
  target interactively. Use it for qualitative debugging / demos.
* **Evaluate mode (`--evaluate`)** — runs the same multi-scenario benchmark as
  `evaluate_kick.py` (which uses Isaac Gym), but inside MuJoCo, serially. Writes
  the same artifact layout: `<output_dir>/<scenario>/{attempts.csv,summary.json}`
  + `all_scenarios_summary.json`.

The 52-dim observation matches `_compute_observations()` in
`envs/T1/kicking_movement_bica.py`. A ball body is injected into the locomotion
MJCF (`resources/T1/T1_locomotion.xml`) at runtime — the original file is
untouched.

Viewer keys (focus on the MuJoCo window):
    R   reset robot + ball   B   reset ball only
    [/] rotate target ∓/±10°  -/= dist ∓/±1 m   P  print state

Examples:
    python play_mujoco_kick.py --task T1/Kicking_Movement_Chapa
    python play_mujoco_kick.py --task T1/Kicking_Movement_Chapa \\
        --evaluate --scenarios angles,distance --num_envs 24 --max_steps 400
"""

import argparse
import csv
import glob
import json
import os
import sys
import time
from typing import Optional

import numpy as np
import torch
import yaml
import mujoco
import mujoco.viewer

from utils.model import ActorCritic


# =============================================================================
# Math helpers
# =============================================================================
def quat_rotate_inverse(q_xyzw, v):
    """Rotate vector v by the inverse of quaternion q (xyzw). Pure numpy."""
    q_w = q_xyzw[3]
    q_vec = q_xyzw[:3]
    a = v * (2.0 * q_w**2 - 1.0)
    b = np.cross(q_vec, v) * (q_w * 2.0)
    c = q_vec * (np.dot(q_vec, v) * 2.0)
    return a - b + c


def quat_rotate(q_xyzw, v):
    """Rotate v by q (xyzw)."""
    q_w = q_xyzw[3]
    q_vec = q_xyzw[:3]
    a = v * (2.0 * q_w**2 - 1.0)
    b = np.cross(q_vec, v) * (q_w * 2.0)
    c = q_vec * (np.dot(q_vec, v) * 2.0)
    return a + b + c


def quat_to_yaw(q_xyzw):
    x, y, z, w = q_xyzw
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return np.arctan2(siny_cosp, cosy_cosp)


def yaw_to_quat_wxyz(yaw):
    return np.array([np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)], dtype=np.float32)


def yaw_to_quat_xyzw(yaw):
    return np.array([0.0, 0.0, np.sin(yaw / 2.0), np.cos(yaw / 2.0)], dtype=np.float32)


# =============================================================================
# MJCF augmentation: inject a ball body
# =============================================================================
BALL_BODY_TEMPLATE = """
        <body name="ball" pos="{x} {y} {z}">
            <freejoint name="ball_freejoint"/>
            <inertial pos="0 0 0" mass="{mass}" diaginertia="{ixx} {ixx} {ixx}"/>
            <geom name="ball_geom" type="sphere" size="{r}"
                  rgba="1 0 0 1" friction="{mu} {mu_roll} 0.0001"
                  solref="0.005 1" condim="6"/>
        </body>
"""


def build_kick_xml(base_xml_path: str, ball_cfg: dict, out_path: str) -> str:
    with open(base_xml_path, "r", encoding="utf-8") as f:
        xml_text = f.read()

    init_pos = ball_cfg.get("init_pos", [0.5, 0.0, 0.075])
    radius = 0.075                       # matches resources/T1/ball.urdf
    mass = 0.2
    inertia = 0.00045
    mu = float(ball_cfg.get("friction", 1.0))
    mu_roll = float(ball_cfg.get("rolling_friction", 1.0)) * 1e-3

    ball_xml = BALL_BODY_TEMPLATE.format(
        x=init_pos[0], y=init_pos[1], z=max(init_pos[2], radius),
        mass=mass, ixx=inertia, r=radius, mu=mu, mu_roll=mu_roll,
    )

    marker = "</worldbody>"
    if marker not in xml_text:
        raise RuntimeError(f"Could not find {marker!r} in {base_xml_path}")
    augmented = xml_text.replace(marker, ball_xml + "    " + marker, 1)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(augmented)
    return out_path


# =============================================================================
# Shared setup
# =============================================================================
def load_cfg(task_name: str) -> dict:
    cfg_file = os.path.join("envs", f"{task_name}.yaml")
    with open(cfg_file, "r", encoding="utf-8") as f:
        return yaml.load(f.read(), Loader=yaml.FullLoader)


def find_checkpoint(task_name: str, override: Optional[str]) -> str:
    if override is not None and os.path.isfile(override):
        return override
    if override is not None:
        raise FileNotFoundError(f"--checkpoint {override} not found")
    task_base = os.path.basename(task_name)
    candidate_pt = os.path.join("deploy", "models", f"{task_base}.pt")
    if os.path.isfile(candidate_pt):
        return candidate_pt
    pths = sorted(
        glob.glob(os.path.join("logs", "T1", "T1", task_base, "**", "model_*.pth"),
                  recursive=True),
        key=os.path.getmtime,
    )
    if pths:
        return pths[-1]
    raise FileNotFoundError(
        "Could not locate a checkpoint. Pass --checkpoint or place one in "
        "deploy/models/<task>.pt or logs/T1/T1/<task>/model_*.pth")


class _JitPolicyWrapper:
    """Adapts a TorchScript actor (obs -> action) to the ActorCritic.act(...).loc API."""

    class _Det:
        def __init__(self, action):
            self.loc = action

    def __init__(self, jit_module):
        self.jit = jit_module

    def act(self, obs):
        out = self.jit(obs)
        # Some exported actors return (mean, ...) tuples.
        if isinstance(out, (tuple, list)):
            out = out[0]
        return _JitPolicyWrapper._Det(out)

    def eval(self):
        self.jit.eval()
        return self


def load_model(cfg: dict, ckpt_path: str):
    # Detect TorchScript archives (zip files starting with PK).
    is_jit = False
    try:
        with open(ckpt_path, "rb") as fh:
            head = fh.read(4)
        if head[:2] == b"PK":
            try:
                jit_mod = torch.jit.load(ckpt_path, map_location="cpu")
                is_jit = True
            except Exception:
                is_jit = False
    except Exception:
        is_jit = False

    if is_jit:
        jit_mod.eval()
        print(f"[info] Loaded TorchScript policy from {ckpt_path}")
        return _JitPolicyWrapper(jit_mod)

    model = ActorCritic(
        cfg["env"]["num_actions"],
        cfg["env"]["num_observations"],
        cfg["env"]["num_privileged_obs"],
    )
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "model" in state:
        model.load_state_dict(state["model"])
    else:
        try:
            model.load_state_dict(state)
        except RuntimeError:
            missing = model.actor.load_state_dict(state, strict=False)
            print(f"[info] Loaded raw state into actor only: {missing}")
    model.eval()
    return model


class SimCtx:
    """Bundles MuJoCo model / data and per-actuator metadata."""

    def __init__(self, cfg: dict):
        # ---- Build XML with ball (writes a sibling file in resources/T1/) -
        base_xml = cfg["asset"]["mujoco_file"]
        ball_cfg = dict(cfg.get("ball", {}))
        if "ball_pos" in cfg.get("init_state", {}):
            ball_cfg["init_pos"] = cfg["init_state"]["ball_pos"]
        runtime_xml = os.path.join(os.path.dirname(base_xml), "_T1_kick_runtime.xml")
        build_kick_xml(base_xml, ball_cfg, runtime_xml)
        self.runtime_xml = runtime_xml
        self.ball_cfg = ball_cfg

        self.mj_model = mujoco.MjModel.from_xml_path(runtime_xml)
        self.mj_model.opt.timestep = cfg["sim"]["dt"]
        self.mj_data = mujoco.MjData(self.mj_model)
        mujoco.mj_resetData(self.mj_model, self.mj_data)

        # ---- Actuator defaults & PD gains --------------------------------
        nu = self.mj_model.nu
        self.default_dof_pos = np.zeros(nu, dtype=np.float32)
        self.dof_stiffness = np.zeros(nu, dtype=np.float32)
        self.dof_damping = np.zeros(nu, dtype=np.float32)
        for i in range(nu):
            act_name = mujoco.mj_id2name(
                self.mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            chosen = None
            for name, val in cfg["init_state"]["default_joint_angles"].items():
                if name != "default" and name in act_name:
                    chosen = val
                    break
            self.default_dof_pos[i] = (
                chosen if chosen is not None
                else cfg["init_state"]["default_joint_angles"].get("default", 0.0)
            )
            gain_set = False
            for name in cfg["control"]["stiffness"].keys():
                if name in act_name:
                    self.dof_stiffness[i] = cfg["control"]["stiffness"][name]
                    self.dof_damping[i] = cfg["control"]["damping"][name]
                    gain_set = True
                    break
            if not gain_set:
                raise ValueError(f"No PD gain defined for actuator {act_name}")

        # ---- Ball joint slice --------------------------------------------
        ball_jnt = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, "ball_freejoint")
        if ball_jnt < 0:
            raise RuntimeError("Ball freejoint not found in compiled model.")
        self.ball_qpos_adr = self.mj_model.jnt_qposadr[ball_jnt]
        self.ball_qvel_adr = self.mj_model.jnt_dofadr[ball_jnt]
        assert self.ball_qpos_adr == self.mj_model.nq - 7, \
            f"Unexpected ball qpos address {self.ball_qpos_adr}"
        self.n_dof = nu
        # Robot trunk body id (assumed first non-world body, named "Trunk").
        self.trunk_body_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "Trunk")
        if self.trunk_body_id < 0:
            # fall back to body 1 (world is 0)
            self.trunk_body_id = 1

    # ---- Slice helpers ---------------------------------------------------
    @property
    def robot_qpos(self):
        return slice(7, 7 + self.n_dof)

    @property
    def robot_qvel(self):
        return slice(6, 6 + self.n_dof)

    # ---- Reset -----------------------------------------------------------
    def reset(self, cfg: dict, reset_robot=True, reset_ball=True,
              robot_yaw=None, ball_pos=None, ball_vel_xy=None):
        d = self.mj_data
        if reset_robot:
            d.qpos[0:3] = np.array(cfg["init_state"]["pos"], dtype=np.float32)
            if robot_yaw is None:
                rot_xyzw = np.array(cfg["init_state"]["rot"], dtype=np.float32)
                d.qpos[3:7] = rot_xyzw[[3, 0, 1, 2]]
            else:
                d.qpos[3:7] = yaw_to_quat_wxyz(robot_yaw)
            d.qpos[self.robot_qpos] = self.default_dof_pos
            d.qvel[0 : 6 + self.n_dof] = 0.0
        if reset_ball:
            if ball_pos is None:
                bp = np.array(self.ball_cfg.get("init_pos",
                              [0.5, 0.0, 0.075]), dtype=np.float32)
            else:
                bp = np.asarray(ball_pos, dtype=np.float32)
            d.qpos[self.ball_qpos_adr : self.ball_qpos_adr + 3] = bp
            d.qpos[self.ball_qpos_adr + 3 : self.ball_qpos_adr + 7] = \
                np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
            d.qvel[self.ball_qvel_adr : self.ball_qvel_adr + 6] = 0.0
            if ball_vel_xy is not None:
                d.qvel[self.ball_qvel_adr + 0] = float(ball_vel_xy[0])
                d.qvel[self.ball_qvel_adr + 1] = float(ball_vel_xy[1])
        # Clear any leftover external force on the trunk.
        d.xfrc_applied[:] = 0.0
        mujoco.mj_forward(self.mj_model, d)


# =============================================================================
# Per-step observation & PD control
# =============================================================================
def read_robot_state(ctx: SimCtx):
    d = ctx.mj_data
    base_pos = d.qpos[0:3].astype(np.float32)
    base_quat_wxyz = d.qpos[3:7].astype(np.float32)
    base_quat_xyzw = base_quat_wxyz[[1, 2, 3, 0]]
    dof_pos = d.qpos[ctx.robot_qpos].astype(np.float32)
    dof_vel = d.qvel[ctx.robot_qvel].astype(np.float32)
    base_ang_vel = d.sensor("angular-velocity").data.astype(np.float32)
    proj_grav = quat_rotate_inverse(
        base_quat_xyzw, np.array([0.0, 0.0, -1.0], dtype=np.float32))
    ball_pos = d.qpos[ctx.ball_qpos_adr : ctx.ball_qpos_adr + 3].astype(np.float32)
    ball_vel = d.qvel[ctx.ball_qvel_adr : ctx.ball_qvel_adr + 3].astype(np.float32)
    return dict(
        base_pos=base_pos, base_quat_xyzw=base_quat_xyzw,
        dof_pos=dof_pos, dof_vel=dof_vel,
        base_ang_vel=base_ang_vel, proj_grav=proj_grav,
        ball_pos=ball_pos, ball_vel=ball_vel,
    )


def build_obs(cfg: dict, ctx: SimCtx, state: dict, prev_actions: np.ndarray,
              target_world: np.ndarray) -> np.ndarray:
    """Build the kick observation. Dispatches by num_observations:
      48 → T1/Kicking layout (no commands, has target_distance slot)
      52 → T1/Kicking_Movement_Bica/Chapa layout (with approach commands)
    """
    norm = cfg["normalization"]
    num_obs = cfg["env"]["num_observations"]
    n_dof = ctx.n_dof

    base_pos = state["base_pos"]
    base_quat_xyzw = state["base_quat_xyzw"]
    ball_pos = state["ball_pos"]

    # Shared: target direction in robot frame and target z relative to ball.
    ball_to_target = target_world[:2] - ball_pos[:2]
    norm_bt = float(np.linalg.norm(ball_to_target)) + 1e-6
    btn = ball_to_target / norm_bt
    yaw = quat_to_yaw(base_quat_xyzw)
    cos_y, sin_y = np.cos(yaw), np.sin(yaw)
    target_dir_robot = np.array(
        [cos_y * btn[0] + sin_y * btn[1],
         -sin_y * btn[0] + cos_y * btn[1]],
        dtype=np.float32,
    )
    target_z_rel = float(target_world[2] - ball_pos[2])
    relative_ball_pos = quat_rotate_inverse(base_quat_xyzw, ball_pos - base_pos)

    obs = np.zeros(num_obs, dtype=np.float32)

    if num_obs == 48:
        # T1/Kicking layout (mirrors envs/T1/kicking.py _compute_observations)
        # [0:3]   proj_grav
        # [3:6]   base_ang_vel
        # [6:8]   relative_ball_pos[:2]
        # [8:10]  target_dir_robot
        # [10]    target_z_rel
        # [11]    kick_target_distance
        # [12:24] dof_pos
        # [24:36] dof_vel
        # [36:48] actions
        kick_target_dist = float(np.linalg.norm(ball_to_target))
        obs[0:3] = state["proj_grav"] * norm["gravity"]
        obs[3:6] = state["base_ang_vel"] * norm["ang_vel"]
        obs[6:8] = relative_ball_pos[:2] * norm["ball_pos"]
        obs[8:10] = target_dir_robot * norm["target_dir"]
        obs[10] = target_z_rel * norm["target_z"]
        obs[11] = kick_target_dist * norm["target_distance"]
        obs[12 : 12 + n_dof] = (state["dof_pos"] - ctx.default_dof_pos) * norm["dof_pos"]
        obs[12 + n_dof : 12 + 2 * n_dof] = state["dof_vel"] * norm["dof_vel"]
        obs[12 + 2 * n_dof : 12 + 3 * n_dof] = prev_actions
    else:
        # T1/Kicking_Movement_Bica/Chapa layout (52 dims, with approach commands)
        # [0:3]   proj_grav
        # [3:6]   base_ang_vel
        # [6:9]   approach commands [vx, vy, wz]
        # [9:11]  zeros (gait slots zeroed)
        # [11:13] relative_ball_pos[:2]
        # [13:15] target_dir_robot
        # [15:16] target_z_rel
        # [16:28] dof_pos
        # [28:40] dof_vel
        # [40:52] actions
        rew = cfg.get("rewards", {})
        stop_dist = float(rew.get("approach_cmd_stop_dist", 0.5))
        max_speed = float(rew.get("approach_cmd_max_speed", 1.0))
        yaw_gain = float(rew.get("approach_cmd_yaw_gain", 0.3))
        to_ball_local = quat_rotate_inverse(base_quat_xyzw, ball_pos - base_pos)
        dist = max(float(np.linalg.norm(to_ball_local[:2])), 1e-6)
        speed = min(dist * (max_speed / stop_dist), max_speed)
        cmd_xy = (to_ball_local[:2] / dist) * speed
        cmd_yaw = np.arctan2(to_ball_local[1], to_ball_local[0]) * yaw_gain
        near = 1.0 if dist < stop_dist else 0.0
        cmd = np.array([cmd_xy[0], cmd_xy[1], cmd_yaw], dtype=np.float32) * (1.0 - near)
        obs[0:3] = state["proj_grav"] * norm["gravity"]
        obs[3:6] = state["base_ang_vel"] * norm["ang_vel"]
        obs[6] = cmd[0] * norm["lin_vel"]
        obs[7] = cmd[1] * norm["lin_vel"]
        obs[8] = cmd[2] * norm["ang_vel"]
        obs[9] = 0.0
        obs[10] = 0.0
        obs[11:13] = relative_ball_pos[:2] * norm["ball_pos"]
        obs[13:15] = target_dir_robot * norm["target_dir"]
        obs[15:16] = target_z_rel * norm["target_z"]
        obs[16 : 16 + n_dof] = (state["dof_pos"] - ctx.default_dof_pos) * norm["dof_pos"]
        obs[16 + n_dof : 16 + 2 * n_dof] = state["dof_vel"] * norm["dof_vel"]
        obs[16 + 2 * n_dof : 16 + 3 * n_dof] = prev_actions
    return obs


def pd_step(ctx: SimCtx, dof_targets: np.ndarray):
    d = ctx.mj_data
    dof_pos = d.qpos[ctx.robot_qpos].astype(np.float32)
    dof_vel = d.qvel[ctx.robot_qvel].astype(np.float32)
    d.ctrl[:] = np.clip(
        ctx.dof_stiffness * (dof_targets - dof_pos) - ctx.dof_damping * dof_vel,
        ctx.mj_model.actuator_ctrlrange[:, 0],
        ctx.mj_model.actuator_ctrlrange[:, 1],
    )
    mujoco.mj_step(ctx.mj_model, d)


# =============================================================================
# Viewer mode
# =============================================================================
def run_viewer(cfg, model, ctx, args):
    num_obs = cfg["env"]["num_observations"]
    num_act = cfg["env"]["num_actions"]
    norm = cfg["normalization"]
    clip_actions = float(norm["clip_actions"])
    action_scale = float(cfg["control"]["action_scale"])
    decimation = int(cfg["control"]["decimation"])

    init_rot_xyzw = np.array(cfg["init_state"]["rot"], dtype=np.float32)
    target_angle_world = quat_to_yaw(init_rot_xyzw) + np.deg2rad(args.target_angle_deg)
    target_distance = float(args.target_distance)
    target_z_rel = float(args.target_z_rel)

    ctx.reset(cfg)
    actions = np.zeros(num_act, dtype=np.float32)
    dof_targets = ctx.default_dof_pos.copy()

    def recompute_target():
        bp = ctx.mj_data.qpos[ctx.ball_qpos_adr : ctx.ball_qpos_adr + 3]
        return np.array([
            bp[0] + target_distance * np.cos(target_angle_world),
            bp[1] + target_distance * np.sin(target_angle_world),
            bp[2] + target_z_rel,
        ], dtype=np.float32)

    def key_callback(keycode):
        nonlocal target_angle_world, target_distance
        try:
            ch = chr(keycode)
        except ValueError:
            return
        if ch in ("R", "r"):
            ctx.reset(cfg)
            actions[:] = 0.0
            dof_targets[:] = ctx.default_dof_pos
            print("[reset] robot+ball")
        elif ch in ("B", "b"):
            ctx.reset(cfg, reset_robot=False, reset_ball=True)
            print("[reset] ball only")
        elif ch == "[":
            target_angle_world -= np.deg2rad(10.0)
            print(f"[target] angle = {np.rad2deg(target_angle_world):+.1f} deg")
        elif ch == "]":
            target_angle_world += np.deg2rad(10.0)
            print(f"[target] angle = {np.rad2deg(target_angle_world):+.1f} deg")
        elif ch == "-":
            target_distance = max(0.5, target_distance - 1.0)
            print(f"[target] distance = {target_distance:.1f} m")
        elif ch == "=":
            target_distance += 1.0
            print(f"[target] distance = {target_distance:.1f} m")
        elif ch in ("P", "p"):
            t = recompute_target()
            bp = ctx.mj_data.qpos[ctx.ball_qpos_adr : ctx.ball_qpos_adr + 3]
            print(f"[state] ball={bp}, target={t}, "
                  f"angle={np.rad2deg(target_angle_world):+.1f}deg, "
                  f"dist={target_distance:.2f}m")

    it = 0
    with mujoco.viewer.launch_passive(
            ctx.mj_model, ctx.mj_data, key_callback=key_callback) as viewer:
        viewer.cam.elevation = -20
        print("Viewer controls: R=reset, B=ball, [/]=target ±10°, -/=dist ∓1m, P=print")
        while viewer.is_running():
            state = read_robot_state(ctx)
            if it % decimation == 0:
                target_w = recompute_target()
                obs = build_obs(cfg, ctx, state, actions, target_w)
                with torch.no_grad():
                    dist_out = model.act(torch.from_numpy(obs).unsqueeze(0))
                    actions[:] = dist_out.loc.squeeze(0).numpy()
                actions[:] = np.clip(actions, -clip_actions, clip_actions)
                dof_targets[:] = ctx.default_dof_pos + action_scale * actions
            pd_step(ctx, dof_targets)
            viewer.cam.lookat[:] = state["base_pos"]
            viewer.sync()
            it += 1


# =============================================================================
# Evaluation mode (mirrors evaluate_kick.py scenarios)
# =============================================================================
ALL_SCENARIOS = ["angles", "ball_pos", "robot_yaw", "ball_vel",
                 "distance", "disturb_push"]


def _parse_range(spec):
    a, b = (float(x) for x in spec.split(","))
    return a, b


def _parse_floats(spec):
    return [float(x) for x in spec.split(",") if x.strip()]


def _parse_strs(spec):
    return [x.strip() for x in spec.split(",") if x.strip()]


def _balanced_conditions(n, k):
    return ((np.arange(n) * k) // n).astype(int)


def _build_scenarios(eval_args, env_cfg):
    """Return list of scenario dicts. Each maps a trial index -> initial setup.

    A scenario has:
        name, description, condition_labels,
        condition_per_trial: list[int] of length num_envs,
        trial_setup: callable(trial_idx, ctx) -> sets up sim.reset(...) state;
                     returns dict {'target_world_xy_angle_deg', 'target_distance_m',
                                   'push_force_N' (optional)} used by the runner.
    """
    n = eval_args.num_envs
    rew = env_cfg.get("rewards", {})
    dist_yaml = float(rew.get("kick_target_ref_distance", 8.0))
    ball_radius = 0.075
    init_ball = env_cfg.get("init_state", {}).get("ball_pos",
                env_cfg.get("ball", {}).get("init_pos", [0.5, 0.0, ball_radius]))
    init_ball = np.array(init_ball, dtype=np.float32)
    init_ball[2] = ball_radius

    scenarios = []

    # ---- angles ---------------------------------------------------------
    if "angles" in eval_args.scenarios:
        k = eval_args.angles_n
        angles_deg = np.linspace(eval_args.angles_min_deg,
                                 eval_args.angles_max_deg, k)
        cond = _balanced_conditions(n, k)
        labels = [f"angle={a:+.1f}deg" for a in angles_deg]

        def trial_setup(i, ctx, _angles_deg=angles_deg, _cond=cond, _ball=init_ball):
            ctx.reset(env_cfg, robot_yaw=0.0, ball_pos=_ball)
            return {"target_angle_world_rad": np.deg2rad(_angles_deg[_cond[i]]),
                    "target_distance_m": dist_yaml,
                    "target_z_world": ball_radius,
                    "push_force_N": 0.0}

        scenarios.append(dict(
            name="angles",
            description=f"Target angle from {eval_args.angles_min_deg} to "
                        f"{eval_args.angles_max_deg} deg in {k} steps.",
            condition_labels=labels, condition_per_trial=cond,
            trial_setup=trial_setup,
        ))

    # ---- ball_pos -------------------------------------------------------
    if "ball_pos" in eval_args.scenarios:
        dx_lo, dx_hi = _parse_range(eval_args.ball_pos_dx_range)
        dy_lo, dy_hi = _parse_range(eval_args.ball_pos_dy_range)
        nx, ny = eval_args.ball_pos_nx, eval_args.ball_pos_ny
        pairs = [(float(x), float(y))
                 for x in np.linspace(dx_lo, dx_hi, nx)
                 for y in np.linspace(dy_lo, dy_hi, ny)]
        k = len(pairs)
        cond = _balanced_conditions(n, k)
        labels = [f"dx={x:+.2f},dy={y:+.2f}" for (x, y) in pairs]

        def trial_setup(i, ctx, _pairs=pairs, _cond=cond):
            dx, dy = _pairs[_cond[i]]
            # Robot at origin facing +x, so robot-frame == world-frame here.
            ball = np.array([dx, dy, ball_radius], dtype=np.float32)
            ctx.reset(env_cfg, robot_yaw=0.0, ball_pos=ball)
            return {"target_angle_world_rad": 0.0,
                    "target_distance_m": dist_yaml,
                    "target_z_world": ball_radius,
                    "push_force_N": 0.0}

        scenarios.append(dict(
            name="ball_pos",
            description=f"Initial ball xy on {nx}x{ny} grid (robot frame).",
            condition_labels=labels, condition_per_trial=cond,
            trial_setup=trial_setup,
        ))

    # ---- robot_yaw ------------------------------------------------------
    if "robot_yaw" in eval_args.scenarios:
        k = eval_args.robot_yaw_n
        yaws_deg = np.linspace(eval_args.robot_yaw_min_deg,
                               eval_args.robot_yaw_max_deg, k)
        cond = _balanced_conditions(n, k)
        labels = [f"yaw={y:+.1f}deg" for y in yaws_deg]

        def trial_setup(i, ctx, _y=yaws_deg, _cond=cond, _ball=init_ball):
            yaw_rad = float(np.deg2rad(_y[_cond[i]]))
            ctx.reset(env_cfg, robot_yaw=yaw_rad, ball_pos=_ball)
            # Target stays in world +x at ref_distance (ball front-facing world).
            return {"target_angle_world_rad": 0.0,
                    "target_distance_m": dist_yaml,
                    "target_z_world": ball_radius,
                    "push_force_N": 0.0}

        scenarios.append(dict(
            name="robot_yaw",
            description="Initial robot yaw with ball at default front pose.",
            condition_labels=labels, condition_per_trial=cond,
            trial_setup=trial_setup,
        ))

    # ---- ball_vel -------------------------------------------------------
    if "ball_vel" in eval_args.scenarios:
        speeds = _parse_floats(eval_args.ball_vel_speeds_mps)
        dirs = _parse_strs(eval_args.ball_vel_directions)
        for d in dirs:
            if d not in {"toward", "perpendicular", "away"}:
                raise ValueError(f"Invalid ball_vel direction '{d}'")
        pairs = [(s, d) for d in dirs for s in speeds]
        k = len(pairs)
        cond = _balanced_conditions(n, k)
        labels = [f"{d}@{s:.2f}mps" for (s, d) in pairs]

        def trial_setup(i, ctx, _pairs=pairs, _cond=cond, _ball=init_ball):
            speed, direction = _pairs[_cond[i]]
            # Robot at origin facing +x, ball ahead. toward = -x (toward robot).
            if direction == "toward":
                vx, vy = -speed, 0.0
            elif direction == "away":
                vx, vy = speed, 0.0
            else:
                vx, vy = 0.0, speed
            ctx.reset(env_cfg, robot_yaw=0.0,
                      ball_pos=_ball, ball_vel_xy=(vx, vy))
            return {"target_angle_world_rad": 0.0,
                    "target_distance_m": dist_yaml,
                    "target_z_world": ball_radius,
                    "push_force_N": 0.0}

        scenarios.append(dict(
            name="ball_vel",
            description="Initial ball rolling velocity (toward/perp/away).",
            condition_labels=labels, condition_per_trial=cond,
            trial_setup=trial_setup,
        ))

    # ---- distance -------------------------------------------------------
    if "distance" in eval_args.scenarios:
        values = _parse_floats(eval_args.distance_values_m)
        k = len(values)
        cond = _balanced_conditions(n, k)
        labels = [f"dist={v:.1f}m" for v in values]

        def trial_setup(i, ctx, _v=values, _cond=cond, _ball=init_ball):
            ctx.reset(env_cfg, robot_yaw=0.0, ball_pos=_ball)
            return {"target_angle_world_rad": 0.0,
                    "target_distance_m": float(_v[_cond[i]]),
                    "target_z_world": ball_radius,
                    "push_force_N": 0.0}

        scenarios.append(dict(
            name="distance",
            description="Kick target distance.",
            condition_labels=labels, condition_per_trial=cond,
            trial_setup=trial_setup,
        ))

    # ---- disturb_push ---------------------------------------------------
    if "disturb_push" in eval_args.scenarios:
        forces = _parse_floats(eval_args.disturb_push_values_N)
        k = len(forces)
        cond = _balanced_conditions(n, k)
        labels = [f"push={f:.0f}N" for f in forces]

        def trial_setup(i, ctx, _f=forces, _cond=cond, _ball=init_ball):
            ctx.reset(env_cfg, robot_yaw=0.0, ball_pos=_ball)
            return {"target_angle_world_rad": 0.0,
                    "target_distance_m": dist_yaml,
                    "target_z_world": ball_radius,
                    "push_force_N": float(_f[_cond[i]])}

        scenarios.append(dict(
            name="disturb_push",
            description=f"+x body push (N) applied at episode step "
                        f"[{eval_args.disturb_push_step}, "
                        f"{eval_args.disturb_push_step + eval_args.disturb_push_duration_steps}).",
            condition_labels=labels, condition_per_trial=cond,
            trial_setup=trial_setup,
        ))

    return scenarios


def _run_one_trial(cfg, ctx, model, setup, eval_args):
    """Run a single rollout. Returns metrics dict for this trial."""
    num_act = cfg["env"]["num_actions"]
    norm = cfg["normalization"]
    clip_actions = float(norm["clip_actions"])
    action_scale = float(cfg["control"]["action_scale"])
    decimation = int(cfg["control"]["decimation"])
    control_dt = float(cfg["sim"]["dt"]) * decimation

    # World-frame target derived from initial robot yaw + scenario delta.
    init_yaw = quat_to_yaw(ctx.mj_data.qpos[3:7][[1, 2, 3, 0]].astype(np.float32))
    target_yaw_world = init_yaw + setup["target_angle_world_rad"]
    target_dist = setup["target_distance_m"]
    target_z_world = setup["target_z_world"]
    ball_init = ctx.mj_data.qpos[ctx.ball_qpos_adr : ctx.ball_qpos_adr + 3].copy()
    target_world = np.array([
        ball_init[0] + target_dist * np.cos(target_yaw_world),
        ball_init[1] + target_dist * np.sin(target_yaw_world),
        target_z_world,
    ], dtype=np.float32)

    # Lateral reference line (world x): base_init_x + ref_x_offset.
    ref_x_offset = float(cfg.get("rewards", {}).get("lateral_error_ref_x", 3.0))
    ref_dist_y = float(cfg.get("rewards", {}).get("kick_target_ref_distance", 8.0))
    base_init_x = float(ctx.mj_data.qpos[0])
    ref_line_x = base_init_x + ref_x_offset

    push_force = setup["push_force_N"]
    push_start = eval_args.disturb_push_step
    push_end = push_start + eval_args.disturb_push_duration_steps

    actions = np.zeros(num_act, dtype=np.float32)
    dof_targets = ctx.default_dof_pos.copy()

    kick_detected = False
    cross_detected = False
    angular_err = np.nan
    z_err = np.nan
    lateral_err = np.nan
    ball_speed_at_kick = np.nan
    foot_kicked = ""
    steps_to_kick = -1
    energy = 0.0
    fell_before = False
    fell_after = False
    timed_out = False
    final_ball_pos = ball_init.copy()

    max_inner = eval_args.max_steps * decimation  # max_steps is control-rate
    for sub_it in range(max_inner):
        # Apply disturbance force on the trunk in local +x at the specified window
        # (converted to world frame).
        d = ctx.mj_data
        if push_force > 0.0:
            ctrl_step = sub_it // decimation
            if push_start <= ctrl_step < push_end:
                base_quat_xyzw = d.qpos[3:7][[1, 2, 3, 0]].astype(np.float32)
                force_world = quat_rotate(
                    base_quat_xyzw, np.array([push_force, 0.0, 0.0], dtype=np.float32))
                d.xfrc_applied[ctx.trunk_body_id, 0:3] = force_world
                d.xfrc_applied[ctx.trunk_body_id, 3:6] = 0.0
            else:
                d.xfrc_applied[ctx.trunk_body_id, :] = 0.0

        state = read_robot_state(ctx)

        # Policy step at control rate.
        if sub_it % decimation == 0:
            obs = build_obs(cfg, ctx, state, actions, target_world)
            with torch.no_grad():
                dist_out = model.act(torch.from_numpy(obs).unsqueeze(0))
                actions[:] = dist_out.loc.squeeze(0).numpy()
            actions[:] = np.clip(actions, -clip_actions, clip_actions)
            dof_targets[:] = ctx.default_dof_pos + action_scale * actions
            energy += float(np.sum(actions * actions))

        pd_step(ctx, dof_targets)

        # ---- Metrics (sampled every step, but conceptually control-rate) -
        ball_pos = d.qpos[ctx.ball_qpos_adr : ctx.ball_qpos_adr + 3].astype(np.float32)
        ball_vel = d.qvel[ctx.ball_qvel_adr : ctx.ball_qvel_adr + 3].astype(np.float32)
        ball_speed = float(np.linalg.norm(ball_vel))

        # Fall detection: base too low or tilted (proj_grav_z > -0.5 ≈ tilt > 60°).
        base_z = float(d.qpos[2])
        proj_gz = float(state["proj_grav"][2])
        fell = (base_z < 0.30) or (proj_gz > -0.5)

        if not kick_detected and ball_speed > 0.5:
            kick_detected = True
            steps_to_kick = sub_it // decimation
            ball_speed_at_kick = ball_speed
            # angular error: between ball velocity and (target - ball).
            tgt_dir = target_world - ball_pos
            tdn = float(np.linalg.norm(tgt_dir)) + 1e-6
            vn = ball_speed + 1e-6
            cos_e = float(np.clip(np.dot(ball_vel, tgt_dir) / (vn * tdn), -1.0, 1.0))
            angular_err = float(np.degrees(np.arccos(cos_e)))
            # z error: |elevation(ball_vel) - elevation(target_dir)|, degrees.
            ball_elev = float(np.arcsin(np.clip(ball_vel[2] / vn, -1.0, 1.0)))
            tgt_dxy = float(np.linalg.norm(tgt_dir[:2])) + 1e-6
            tgt_elev = float(np.arctan2(tgt_dir[2], tgt_dxy))
            z_err = float(np.degrees(abs(ball_elev - tgt_elev)))
            # Foot proxy: closest geom contact body. We use a simple xy-closest.
            # Read foot site/body positions via body names "left_foot_link"
            # / "right_foot_link".
            lf_id = mujoco.mj_name2id(ctx.mj_model, mujoco.mjtObj.mjOBJ_BODY,
                                      "left_foot_link")
            rf_id = mujoco.mj_name2id(ctx.mj_model, mujoco.mjtObj.mjOBJ_BODY,
                                      "right_foot_link")
            if lf_id >= 0 and rf_id >= 0:
                lp = d.xpos[lf_id].astype(np.float32)
                rp = d.xpos[rf_id].astype(np.float32)
                foot_kicked = "left" if np.linalg.norm(lp - ball_pos) <= \
                                       np.linalg.norm(rp - ball_pos) else "right"

        if not cross_detected and ball_pos[0] >= ref_line_x:
            cross_detected = True
            target_y_at_ref = target_world[1] * (ref_x_offset / ref_dist_y)
            lateral_err = float(abs(ball_pos[1] - target_y_at_ref))
            final_ball_pos = ball_pos.copy()

        if fell:
            if kick_detected:
                fell_after = True
            else:
                fell_before = True
            final_ball_pos = ball_pos.copy()
            break

        if kick_detected and cross_detected:
            final_ball_pos = ball_pos.copy()
            break
    else:
        timed_out = True
        final_ball_pos = ctx.mj_data.qpos[
            ctx.ball_qpos_adr : ctx.ball_qpos_adr + 3].astype(np.float32).copy()

    ball_travel = float(np.linalg.norm(final_ball_pos[:2] - ball_init[:2]))
    hit = (
        kick_detected and cross_detected
        and (not np.isnan(angular_err)) and angular_err < eval_args.hit_angle_deg
        and (not np.isnan(lateral_err)) and abs(lateral_err) < eval_args.hit_lateral_m
    )
    return dict(
        kick_detected=kick_detected, cross_detected=cross_detected,
        angular_err=angular_err, z_err=z_err, lateral_err=lateral_err,
        ball_speed_at_kick=ball_speed_at_kick, ball_travel_m=ball_travel,
        steps_to_kick=steps_to_kick, time_to_kick_s=(steps_to_kick * control_dt
                                                    if steps_to_kick >= 0 else np.nan),
        energy=energy, foot_kicked=foot_kicked,
        timed_out=timed_out, fell_before=fell_before, fell_after=fell_after,
        hit=hit,
    )


def _safe_stats(values, mask):
    sel = np.array([v for v, m in zip(values, mask) if m and not (
        isinstance(v, float) and np.isnan(v))])
    if sel.size == 0:
        return {"count": 0, "mean": None, "std": None}
    return {"count": int(sel.size),
            "mean": float(sel.mean()),
            "std": float(sel.std())}


def _aggregate(scenario, trials):
    cond = scenario["condition_per_trial"]
    labels = scenario["condition_labels"]
    n = len(trials)

    def col(key):
        return [t[key] for t in trials]

    kicks = np.array(col("kick_detected"))
    crosses = np.array(col("cross_detected"))
    hits = np.array(col("hit"))
    fells = np.array(col("fell_before")) | np.array(col("fell_after"))
    timed = np.array(col("timed_out"))
    fell_before = np.array(col("fell_before"))
    fell_after = np.array(col("fell_after"))

    per_cond = []
    for c_idx, lab in enumerate(labels):
        m = (cond == c_idx)
        nm = int(m.sum())
        if nm == 0:
            continue
        foot_left = sum(1 for i in range(n) if m[i] and trials[i]["foot_kicked"] == "left")
        foot_right = sum(1 for i in range(n) if m[i] and trials[i]["foot_kicked"] == "right")
        foot_unknown = sum(1 for i in range(n) if m[i] and trials[i]["foot_kicked"] == "" and trials[i]["kick_detected"])
        per_cond.append({
            "condition_idx": c_idx,
            "condition_label": lab,
            "n_attempts": nm,
            "n_kicks_detected": int(kicks[m].sum()),
            "n_crossed_ref": int(crosses[m].sum()),
            "n_timed_out": int(timed[m].sum()),
            "n_fell_before_kick": int(fell_before[m].sum()),
            "n_fell_after_kick": int(fell_after[m].sum()),
            "n_hits": int(hits[m].sum()),
            "hit_rate": float(hits[m].mean()),
            "angular_error_deg": _safe_stats(col("angular_err"),
                                             m & kicks),
            "z_error_deg": _safe_stats(col("z_err"), m & kicks),
            "lateral_error_m": _safe_stats(col("lateral_err"),
                                           m & crosses),
            "ball_speed_at_kick_mps": _safe_stats(col("ball_speed_at_kick"),
                                                  m & kicks),
            "ball_travel_m": _safe_stats(col("ball_travel_m"), m),
            "time_to_kick_s": _safe_stats(col("time_to_kick_s"),
                                          m & kicks),
            "energy_act_sq": _safe_stats(col("energy"), m),
            "foot_kicked_counts": {"left": foot_left, "right": foot_right,
                                   "unknown": foot_unknown},
        })

    all_mask = np.ones(n, dtype=bool)
    overall = {
        "n_attempts": n,
        "n_kicks_detected": int(kicks.sum()),
        "n_crossed_ref": int(crosses.sum()),
        "n_timed_out": int(timed.sum()),
        "n_fell_before_kick": int(fell_before.sum()),
        "n_fell_after_kick": int(fell_after.sum()),
        "n_hits": int(hits.sum()),
        "hit_rate": float(hits.mean()),
        "angular_error_deg": _safe_stats(col("angular_err"), kicks),
        "z_error_deg": _safe_stats(col("z_err"), kicks),
        "lateral_error_m": _safe_stats(col("lateral_err"), crosses),
        "ball_speed_at_kick_mps": _safe_stats(col("ball_speed_at_kick"), kicks),
        "ball_travel_m": _safe_stats(col("ball_travel_m"), all_mask),
        "time_to_kick_s": _safe_stats(col("time_to_kick_s"), kicks),
        "energy_act_sq": _safe_stats(col("energy"), all_mask),
        "foot_kicked_counts": {
            "left": sum(1 for t in trials if t["foot_kicked"] == "left"),
            "right": sum(1 for t in trials if t["foot_kicked"] == "right"),
            "unknown": sum(1 for t in trials if t["foot_kicked"] == "" and t["kick_detected"]),
        },
    }
    return per_cond, overall


def _fmt(stats):
    if stats["count"] == 0 or stats["mean"] is None:
        return "    n/a   "
    return f"{stats['mean']:6.2f}±{stats['std']:5.2f}"


def _write_scenario_artifacts(scenario, trials, per_cond, overall, out_dir,
                              elapsed):
    os.makedirs(out_dir, exist_ok=True)
    cond = scenario["condition_per_trial"]
    labels = scenario["condition_labels"]
    with open(os.path.join(out_dir, "attempts.csv"), "w", newline="") as f:
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
        for i, t in enumerate(trials):
            stk = int(t["steps_to_kick"])
            stk_s = "" if stk < 0 else str(stk)
            tk_s = "" if np.isnan(t["time_to_kick_s"]) else f"{t['time_to_kick_s']:.3f}"
            w.writerow([
                i, int(cond[i]), labels[cond[i]],
                int(t["kick_detected"]), int(t["cross_detected"]),
                int(t["timed_out"]), int(t["fell_before"]),
                int(t["fell_after"]),
                "" if np.isnan(t["angular_err"]) else f"{t['angular_err']:.4f}",
                "" if np.isnan(t["z_err"]) else f"{t['z_err']:.4f}",
                "" if np.isnan(t["lateral_err"]) else f"{t['lateral_err']:.4f}",
                "" if np.isnan(t["ball_speed_at_kick"]) else f"{t['ball_speed_at_kick']:.4f}",
                f"{t['ball_travel_m']:.4f}",
                stk_s, tk_s,
                f"{t['energy']:.4f}",
                t["foot_kicked"],
                int(t["hit"]),
            ])
    summary = {"scenario": scenario["name"],
               "description": scenario["description"],
               "per_condition": per_cond, "overall": overall,
               "elapsed_s": elapsed}
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def _print_scenario_table(scenario, per_cond, overall):
    print()
    print(f"--- Scenario: {scenario['name']} ---")
    hdr = (f"{'idx':>3} {'condition':>22} {'n':>3} "
           f"{'hit%':>5} {'kick%':>5} {'fall%':>5} "
           f"{'angErr°':>13} {'latErr_m':>13} "
           f"{'bSpd_mps':>13} {'travel_m':>13} {'t_kick_s':>13} {'L/R':>9}")
    print(hdr)
    print("-" * len(hdr))
    rows = list(per_cond) + [None]
    for row in rows:
        if row is None:
            row = {**overall, "condition_idx": "-",
                   "condition_label": "OVERALL"}
        n = row["n_attempts"]
        fall_count = row["n_fell_before_kick"] + row["n_fell_after_kick"]
        fc = row["foot_kicked_counts"]
        lr = f"{fc['left']}/{fc['right']}"
        print(f"{str(row['condition_idx']):>3} {row['condition_label']:>22} "
              f"{n:>3d} "
              f"{100 * row['hit_rate']:>4.0f}% "
              f"{100 * row['n_kicks_detected'] / max(n, 1):>4.0f}% "
              f"{100 * fall_count / max(n, 1):>4.0f}% "
              f"{_fmt(row['angular_error_deg']):>13} "
              f"{_fmt(row['lateral_error_m']):>13} "
              f"{_fmt(row['ball_speed_at_kick_mps']):>13} "
              f"{_fmt(row['ball_travel_m']):>13} "
              f"{_fmt(row['time_to_kick_s']):>13} "
              f"{lr:>9}")


def run_evaluation(cfg, model, ctx, eval_args, task_name, ckpt_path):
    # Normalize scenarios arg
    s_arg = eval_args.scenarios.strip().lower()
    if s_arg == "all":
        eval_args.scenarios = list(ALL_SCENARIOS)
    else:
        eval_args.scenarios = [s.strip() for s in s_arg.split(",") if s.strip()]
        for s in eval_args.scenarios:
            if s not in ALL_SCENARIOS:
                print(f"[eval] unknown scenario '{s}'. Valid: {ALL_SCENARIOS}",
                      file=sys.stderr)
                sys.exit(1)
    print(f"[eval] scenarios={eval_args.scenarios}  num_envs={eval_args.num_envs}")

    out_root = eval_args.output_dir
    if out_root is None:
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_root = os.path.join("eval_results",
                                task_name.replace("/", "_") + "_mujoco", ts)
    os.makedirs(out_root, exist_ok=True)
    print(f"[eval] output_dir = {out_root}")

    scenarios = _build_scenarios(eval_args, cfg)
    all_summaries = {}
    for scenario in scenarios:
        print(f"\n[eval] === Scenario: {scenario['name']} "
              f"({scenario['description']}) ===")
        print(f"[eval] {len(scenario['condition_labels'])} conditions: "
              f"{scenario['condition_labels']}")
        t0 = time.time()
        trials = []
        for i in range(eval_args.num_envs):
            setup = scenario["trial_setup"](i, ctx)
            trial = _run_one_trial(cfg, ctx, model, setup, eval_args)
            trials.append(trial)
            if (i + 1) % max(1, eval_args.num_envs // 4) == 0:
                kicked = sum(t["kick_detected"] for t in trials)
                hit = sum(t["hit"] for t in trials)
                print(f"[eval]   trial {i + 1}/{eval_args.num_envs}: "
                      f"kicks={kicked}, hits={hit}")
        elapsed = time.time() - t0
        print(f"[eval]   rollout finished in {elapsed:.1f}s")
        per_cond, overall = _aggregate(scenario, trials)
        _write_scenario_artifacts(scenario, trials, per_cond, overall,
                                  os.path.join(out_root, scenario["name"]),
                                  elapsed)
        _print_scenario_table(scenario, per_cond, overall)
        all_summaries[scenario["name"]] = {
            "description": scenario["description"],
            "overall": overall, "per_condition": per_cond,
        }

    top = {"task": task_name, "checkpoint": ckpt_path,
           "num_envs": eval_args.num_envs, "max_steps": eval_args.max_steps,
           "hit_thresholds": {"angle_deg": eval_args.hit_angle_deg,
                              "lateral_m": eval_args.hit_lateral_m},
           "scenarios": all_summaries, "backend": "mujoco"}
    top_path = os.path.join(out_root, "all_scenarios_summary.json")
    with open(top_path, "w") as f:
        json.dump(top, f, indent=2)
    print(f"\n[eval] all scenarios done. Top-level summary: {top_path}")


# =============================================================================
# CLI
# =============================================================================
def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True, type=str,
                   help="Task name relative to envs/, e.g. T1/Kicking_Movement_Chapa.")
    p.add_argument("--checkpoint", type=str, default=None)

    # Viewer-only args
    p.add_argument("--target-angle-deg", type=float, default=0.0)
    p.add_argument("--target-distance", type=float, default=8.0)
    p.add_argument("--target-z-rel", type=float, default=0.0)

    # Evaluation
    p.add_argument("--evaluate", action="store_true",
                   help="Run multi-scenario benchmark instead of viewer.")
    p.add_argument("--scenarios", type=str, default="all",
                   help="Comma list or 'all'. " + ",".join(ALL_SCENARIOS))
    p.add_argument("--num_envs", type=int, default=60,
                   help="Number of trials per scenario (serial in MuJoCo).")
    p.add_argument("--max_steps", type=int, default=600,
                   help="Max control-rate steps per trial.")
    p.add_argument("--hit_angle_deg", type=float, default=20.0)
    p.add_argument("--hit_lateral_m", type=float, default=0.5)
    p.add_argument("--output_dir", type=str, default=None)
    # Per-scenario knobs (same names as evaluate_kick.py)
    p.add_argument("--angles_min_deg", type=float, default=-15.0)
    p.add_argument("--angles_max_deg", type=float, default=15.0)
    p.add_argument("--angles_n", type=int, default=6)
    p.add_argument("--ball_pos_dx_range", type=str, default="0.20,0.45")
    p.add_argument("--ball_pos_dy_range", type=str, default="-0.15,0.15")
    p.add_argument("--ball_pos_nx", type=int, default=4)
    p.add_argument("--ball_pos_ny", type=int, default=3)
    p.add_argument("--robot_yaw_min_deg", type=float, default=-30.0)
    p.add_argument("--robot_yaw_max_deg", type=float, default=30.0)
    p.add_argument("--robot_yaw_n", type=int, default=5)
    p.add_argument("--ball_vel_speeds_mps", type=str, default="0.1,0.3,0.5")
    p.add_argument("--ball_vel_directions", type=str,
                   default="toward,perpendicular,away")
    p.add_argument("--distance_values_m", type=str, default="2,4,6,8,10")
    p.add_argument("--disturb_push_values_N", type=str, default="0,25,50,75,100")
    p.add_argument("--disturb_push_step", type=int, default=50)
    p.add_argument("--disturb_push_duration_steps", type=int, default=5)
    return p


def main():
    args = build_argparser().parse_args()

    cfg = load_cfg(args.task)
    if cfg["env"]["num_observations"] != 52:
        print(f"[warn] Expected num_observations=52, got "
              f"{cfg['env']['num_observations']}.")
    ckpt_path = find_checkpoint(args.task, args.checkpoint)
    print(f"Loading checkpoint: {ckpt_path}")
    model = load_model(cfg, ckpt_path)
    ctx = SimCtx(cfg)
    print(f"Built runtime MJCF: {ctx.runtime_xml}  "
          f"(nq={ctx.mj_model.nq}, nu={ctx.mj_model.nu})")

    if args.evaluate:
        run_evaluation(cfg, model, ctx, args, args.task, ckpt_path)
    else:
        run_viewer(cfg, model, ctx, args)


if __name__ == "__main__":
    main()
