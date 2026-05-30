"""Play a trained Base Walk Extended policy inside MuJoCo.

Opens a MuJoCo viewer window and lets you drive the robot interactively by
typing velocity commands in the terminal while the simulation runs.

Observation layout (53 dims) mirrors _compute_observations() in
envs/T1/base_walk.py with extended_obs=True:
    [0:3]   projected_gravity        × norm["gravity"]  (= 1.0)
    [3:6]   base_ang_vel             × norm["ang_vel"]   (= 1.0)
    [6:9]   commands [vx, vy, wz]    × norm["lin_vel"]   (= 1.0)
    [9]     cos(2π·gait_process)     · (gait_freq > 0)
    [10]    sin(2π·gait_process)     · (gait_freq > 0)
    [11:17] zeros  (ball/target slots zeroed during walk training)
    [17:29] (dof_pos − default_dof_pos) × norm["dof_pos"] (= 1.0)
    [29:41] dof_vel                  × norm["dof_vel"]   (= 0.1)
    [41:53] prev_actions

Viewer keys (focus on the MuJoCo window):
    W / S    vx  +/− 0.1 m/s
    A / D    wz  +/− 0.1 rad/s  (A = turn left, D = turn right)
    Q / E    vy  +/− 0.1 m/s
    Space    stop (cmd → 0 0 0)
    R        reset robot to initial pose
    P        print current command and robot state

Terminal input:
    Type  <vx> <vy> <wz>  (space or comma separated) and press Enter.
    Examples:
        0.5 0 0       → walk forward 0.5 m/s
        0.5 0 0.3     → walk forward and turn left
        0,0,0         → stop
        0.5           → only vx (vy and wz default to 0)

Examples:
    python play_mujoco_walk.py --task T1/Base_Walk_Extended
    python play_mujoco_walk.py --task T1/Base_Walk_Extended --checkpoint -1
    python play_mujoco_walk.py --task T1/Base_Walk_Extended --command 0.5,0,0
"""

import argparse
import glob
import os
import sys
import threading
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

def quat_rotate_inverse(q_xyzw: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector v by the inverse of quaternion q (xyzw convention)."""
    q_w = q_xyzw[3]
    q_vec = q_xyzw[:3]
    a = v * (2.0 * q_w ** 2 - 1.0)
    b = np.cross(q_vec, v) * (q_w * 2.0)
    c = q_vec * (np.dot(q_vec, v) * 2.0)
    return a - b + c


# =============================================================================
# Config & checkpoint helpers
# =============================================================================

def load_cfg(task_name: str) -> dict:
    cfg_file = os.path.join("envs", f"{task_name}.yaml")
    with open(cfg_file, "r", encoding="utf-8") as f:
        return yaml.load(f.read(), Loader=yaml.FullLoader)


def find_checkpoint(task_name: str, override: Optional[str]) -> str:
    """Resolve checkpoint path.

    Accepts an explicit path, -1 (latest from logs), or None (same as -1).
    """
    if override is not None and override != "-1" and os.path.isfile(override):
        return override

    task_base = os.path.basename(task_name)

    # Explicit path given but file not found.
    if override is not None and override != "-1":
        raise FileNotFoundError(f"Checkpoint not found: {override}")

    # Search in logs (latest by modification time).
    pths = sorted(
        glob.glob(
            os.path.join("logs", "**", task_base, "**", "model_*.pth"),
            recursive=True,
        ),
        key=os.path.getmtime,
    )
    if pths:
        return pths[-1]

    # Fallback: deploy/models directory.
    for candidate in [
        os.path.join("deploy", "models", f"{task_base}.pt"),
        os.path.join("deploy", "models", f"{task_base.lower()}.pt"),
    ]:
        if os.path.isfile(candidate):
            return candidate

    raise FileNotFoundError(
        "No checkpoint found. Pass --checkpoint <path> or ensure a model exists "
        f"under logs/**/{task_base}/model_*.pth"
    )


# =============================================================================
# Model loading
# =============================================================================

class _JitPolicyWrapper:
    """Adapts a TorchScript actor (obs → action) to the ActorCritic.act().loc API."""

    class _Det:
        def __init__(self, action):
            self.loc = action

    def __init__(self, jit_module):
        self.jit = jit_module

    def act(self, obs):
        out = self.jit(obs)
        if isinstance(out, (tuple, list)):
            out = out[0]
        return _JitPolicyWrapper._Det(out)

    def eval(self):
        self.jit.eval()
        return self


def load_model(cfg: dict, ckpt_path: str):
    # Detect TorchScript archives (zip files).
    is_jit = False
    try:
        with open(ckpt_path, "rb") as fh:
            if fh.read(2) == b"PK":
                try:
                    jit_mod = torch.jit.load(ckpt_path, map_location="cpu")
                    is_jit = True
                except Exception:
                    pass
    except Exception:
        pass

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
        model.load_state_dict(state)
    model.eval()
    print(f"[info] Loaded checkpoint: {ckpt_path}")
    return model


# =============================================================================
# Simulation context (no ball)
# =============================================================================

class WalkSimCtx:
    """MuJoCo model/data + per-actuator metadata for the walk policy."""

    def __init__(self, cfg: dict):
        xml_path = cfg["asset"]["mujoco_file"]
        self.mj_model = mujoco.MjModel.from_xml_path(xml_path)
        self.mj_model.opt.timestep = cfg["sim"]["dt"]
        self.mj_data = mujoco.MjData(self.mj_model)
        mujoco.mj_resetData(self.mj_model, self.mj_data)

        nu = self.mj_model.nu
        self.n_dof = nu
        self.default_dof_pos = np.zeros(nu, dtype=np.float32)
        self.dof_stiffness = np.zeros(nu, dtype=np.float32)
        self.dof_damping = np.zeros(nu, dtype=np.float32)

        joint_angles = cfg["init_state"]["default_joint_angles"]
        stiffness_cfg = cfg["control"]["stiffness"]
        damping_cfg = cfg["control"]["damping"]

        for i in range(nu):
            act_name = mujoco.mj_id2name(
                self.mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, i
            )
            # Default joint angle: first matching key wins, else "default".
            chosen = None
            for key, val in joint_angles.items():
                if key != "default" and key in act_name:
                    chosen = float(val)
                    break
            self.default_dof_pos[i] = (
                chosen if chosen is not None else float(joint_angles.get("default", 0.0))
            )
            # PD gains.
            gain_set = False
            for key in stiffness_cfg:
                if key in act_name:
                    self.dof_stiffness[i] = float(stiffness_cfg[key])
                    self.dof_damping[i] = float(damping_cfg[key])
                    gain_set = True
                    break
            if not gain_set:
                raise ValueError(f"No PD gain defined for actuator '{act_name}'")

        # Trunk body id.
        self.trunk_body_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "Trunk"
        )
        if self.trunk_body_id < 0:
            self.trunk_body_id = 1  # fallback: body 1 (world = 0)

        self.cfg = cfg

    @property
    def robot_qpos(self):
        """Slice into mj_data.qpos for the 12 joint positions."""
        return slice(7, 7 + self.n_dof)

    @property
    def robot_qvel(self):
        """Slice into mj_data.qvel for the 12 joint velocities."""
        return slice(6, 6 + self.n_dof)

    def reset(self):
        """Reset robot to the configured initial pose."""
        d = self.mj_data
        cfg = self.cfg
        init = cfg["init_state"]

        d.qpos[0:3] = np.array(init["pos"], dtype=np.float32)
        # init["rot"] is xyzw → MuJoCo wants wxyz.
        rot_xyzw = np.array(init["rot"], dtype=np.float32)
        d.qpos[3:7] = rot_xyzw[[3, 0, 1, 2]]
        d.qpos[self.robot_qpos] = self.default_dof_pos
        d.qvel[:] = 0.0
        d.xfrc_applied[:] = 0.0
        mujoco.mj_forward(self.mj_model, d)


# =============================================================================
# Thread-safe command holder
# =============================================================================

class CommandRef:
    """Shared velocity command updated from stdin thread or key callbacks."""

    def __init__(self, initial: list):
        self._cmd = np.array(initial, dtype=np.float32)
        self._lock = threading.Lock()

    def get(self) -> np.ndarray:
        with self._lock:
            return self._cmd.copy()

    def set(self, vx: float, vy: float, wz: float):
        with self._lock:
            self._cmd[:] = [vx, vy, wz]

    def nudge(self, dvx: float = 0.0, dvy: float = 0.0, dwz: float = 0.0):
        with self._lock:
            self._cmd[0] = float(np.clip(self._cmd[0] + dvx, -1.0, 1.0))
            self._cmd[1] = float(np.clip(self._cmd[1] + dvy, -1.0, 1.0))
            self._cmd[2] = float(np.clip(self._cmd[2] + dwz, -1.0, 1.0))

    def stop(self):
        with self._lock:
            self._cmd[:] = 0.0

    def _fmt(self) -> str:
        c = self._cmd
        return f"vx={c[0]:+.2f}  vy={c[1]:+.2f}  wz={c[2]:+.2f}"

    def print_state(self):
        with self._lock:
            print(f"[CMD] {self._fmt()}", flush=True)


# =============================================================================
# Terminal stdin input thread
# =============================================================================

def _stdin_reader(cmd_ref: CommandRef):
    """Daemon thread: parse lines from stdin and update CommandRef."""
    print(
        "\n[Terminal input] Type  <vx> [vy] [wz]  and press Enter.\n"
        "  Examples:  0.5          →  walk forward\n"
        "             0.5 0 0.3   →  walk forward + turn left\n"
        "             0 0 0        →  stop\n"
        "             0.5,0,-0.3  →  comma-separated also works\n",
        flush=True,
    )
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        parts = line.replace(",", " ").split()
        try:
            vals = [float(p) for p in parts]
        except ValueError:
            print(f"[CMD] Could not parse '{line}'. Use numbers, e.g. '0.5 0 0'",
                  flush=True)
            continue
        if len(vals) == 1:
            cmd_ref.set(vals[0], 0.0, 0.0)
        elif len(vals) == 2:
            cmd_ref.set(vals[0], vals[1], 0.0)
        elif len(vals) >= 3:
            cmd_ref.set(vals[0], vals[1], vals[2])
        else:
            continue
        cmd_ref.print_state()


# =============================================================================
# Per-step robot state & observation
# =============================================================================

def read_robot_state(ctx: WalkSimCtx) -> dict:
    d = ctx.mj_data
    base_quat_wxyz = d.qpos[3:7].astype(np.float32)
    base_quat_xyzw = base_quat_wxyz[[1, 2, 3, 0]]
    proj_grav = quat_rotate_inverse(
        base_quat_xyzw, np.array([0.0, 0.0, -1.0], dtype=np.float32)
    )
    return dict(
        base_pos=d.qpos[0:3].astype(np.float32),
        base_quat_xyzw=base_quat_xyzw,
        proj_grav=proj_grav,
        ang_vel=d.sensor("angular-velocity").data.astype(np.float32),
        dof_pos=d.qpos[ctx.robot_qpos].astype(np.float32),
        dof_vel=d.qvel[ctx.robot_qvel].astype(np.float32),
    )


def build_obs(
    cfg: dict,
    ctx: WalkSimCtx,
    state: dict,
    cmd: np.ndarray,
    gait_process: float,
    gait_freq_active: float,
    prev_actions: np.ndarray,
) -> np.ndarray:
    """Build the 53-dim walk observation matching envs/T1/base_walk.py."""
    norm = cfg["normalization"]
    n_dof = ctx.n_dof
    obs = np.zeros(cfg["env"]["num_observations"], dtype=np.float32)

    gait_active = float(gait_freq_active > 1.0e-8)
    phase = 2.0 * np.pi * gait_process

    # commands_scale matches base_walk.py: [lin_vel, lin_vel, ang_vel]
    cmd_scale = np.array(
        [norm["lin_vel"], norm["lin_vel"], norm["ang_vel"]], dtype=np.float32
    )

    obs[0:3] = state["proj_grav"] * norm["gravity"]
    obs[3:6] = state["ang_vel"] * norm["ang_vel"]
    obs[6:9] = cmd * cmd_scale
    obs[9] = np.cos(phase) * gait_active
    obs[10] = np.sin(phase) * gait_active
    # obs[11:17] = 0  (ball/target slots — already zero-initialised)
    obs[17 : 17 + n_dof] = (state["dof_pos"] - ctx.default_dof_pos) * norm["dof_pos"]
    obs[17 + n_dof : 17 + 2 * n_dof] = state["dof_vel"] * norm["dof_vel"]
    obs[17 + 2 * n_dof : 17 + 3 * n_dof] = prev_actions
    return obs


# =============================================================================
# PD control step
# =============================================================================

def pd_step(ctx: WalkSimCtx, dof_targets: np.ndarray):
    d = ctx.mj_data
    dof_pos = d.qpos[ctx.robot_qpos].astype(np.float32)
    dof_vel = d.qvel[ctx.robot_qvel].astype(np.float32)
    torque = ctx.dof_stiffness * (dof_targets - dof_pos) - ctx.dof_damping * dof_vel
    d.ctrl[:] = np.clip(
        torque,
        ctx.mj_model.actuator_ctrlrange[:, 0],
        ctx.mj_model.actuator_ctrlrange[:, 1],
    )
    mujoco.mj_step(ctx.mj_model, d)


# =============================================================================
# Viewer loop
# =============================================================================

def run_viewer(cfg: dict, model, ctx: WalkSimCtx, cmd_ref: CommandRef, args):
    num_act = cfg["env"]["num_actions"]
    clip_actions = float(cfg["normalization"]["clip_actions"])
    action_scale = float(cfg["control"]["action_scale"])
    decimation = int(cfg["control"]["decimation"])
    control_dt = float(cfg["sim"]["dt"]) * decimation
    gait_freq = float(args.gait_freq)

    ctx.reset()
    prev_actions = np.zeros(num_act, dtype=np.float32)
    dof_targets = ctx.default_dof_pos.copy()
    gait_process = 0.0

    def key_callback(keycode):
        nonlocal dof_targets, prev_actions
        try:
            ch = chr(keycode).upper()
        except (ValueError, OverflowError):
            return

        changed = True
        if ch == "W":
            cmd_ref.nudge(dvx=+0.1)
        elif ch == "S":
            cmd_ref.nudge(dvx=-0.1)
        elif ch == "A":
            cmd_ref.nudge(dwz=+0.1)
        elif ch == "D":
            cmd_ref.nudge(dwz=-0.1)
        elif ch == "Q":
            cmd_ref.nudge(dvy=+0.1)
        elif ch == "E":
            cmd_ref.nudge(dvy=-0.1)
        elif ch == " ":
            cmd_ref.stop()
        elif ch == "R":
            ctx.reset()
            prev_actions[:] = 0.0
            dof_targets[:] = ctx.default_dof_pos
            print("[reset] robot reset to initial pose", flush=True)
            changed = False
        elif ch == "P":
            state = read_robot_state(ctx)
            cmd_ref.print_state()
            print(
                f"[state] base_pos={state['base_pos']}  "
                f"proj_grav={state['proj_grav']}",
                flush=True,
            )
            changed = False
        else:
            changed = False

        if changed:
            cmd_ref.print_state()

    it = 0
    print(
        "\n[Viewer] Keyboard shortcuts (click MuJoCo window first):\n"
        "  W/S  = vx ±0.1    A/D = wz ±0.1    Q/E = vy ±0.1\n"
        "  Space = stop       R = reset         P = print state\n",
        flush=True,
    )

    with mujoco.viewer.launch_passive(
        ctx.mj_model, ctx.mj_data, key_callback=key_callback
    ) as viewer:
        viewer.cam.elevation = -20

        while viewer.is_running():
            state = read_robot_state(ctx)

            if it % decimation == 0:
                cmd = cmd_ref.get()

                # Gait oscillator: freeze when standing still.
                speed = float(np.linalg.norm(cmd))
                gait_freq_active = gait_freq if speed > 1e-3 else 0.0

                obs = build_obs(cfg, ctx, state, cmd, gait_process,
                                gait_freq_active, prev_actions)

                with torch.no_grad():
                    dist_out = model.act(torch.from_numpy(obs).unsqueeze(0))
                    prev_actions = dist_out.loc.squeeze(0).numpy()

                prev_actions = np.clip(prev_actions, -clip_actions, clip_actions)
                dof_targets = ctx.default_dof_pos + action_scale * prev_actions

                # Advance gait phase.
                gait_process = (gait_process + control_dt * gait_freq_active) % 1.0

            pd_step(ctx, dof_targets)
            viewer.cam.lookat[:] = state["base_pos"]
            viewer.sync()
            it += 1


# =============================================================================
# Entry point
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Play a Base Walk Extended policy in MuJoCo."
    )
    parser.add_argument(
        "--task", default="T1/Base_Walk_Extended",
        help="Task path, e.g. T1/Base_Walk_Extended (default: T1/Base_Walk_Extended)",
    )
    parser.add_argument(
        "--checkpoint", default=None,
        help="Path to .pth checkpoint, or -1 to auto-load the latest from logs/.",
    )
    parser.add_argument(
        "--command", default=None,
        help="Initial velocity command as 'vx,vy,wz' m/s (e.g. '0.5,0,0'). "
             "Default: 0 0 0 (standing still).",
    )
    parser.add_argument(
        "--gait_freq", type=float, default=1.5,
        help="Gait oscillator frequency in Hz when the robot is moving. "
             "Training range: [1.0, 2.0]. Default: 1.5",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Initial command.
    initial_cmd = [0.0, 0.0, 0.0]
    if args.command is not None:
        parts = args.command.replace(",", " ").split()
        vals = [float(p) for p in parts]
        initial_cmd = (vals + [0.0, 0.0, 0.0])[:3]

    cfg = load_cfg(args.task)
    ckpt_path = find_checkpoint(args.task, args.checkpoint)
    model = load_model(cfg, ckpt_path)
    ctx = WalkSimCtx(cfg)

    cmd_ref = CommandRef(initial_cmd)
    if any(v != 0.0 for v in initial_cmd):
        cmd_ref.print_state()

    # Start stdin reader thread.
    t = threading.Thread(target=_stdin_reader, args=(cmd_ref,), daemon=True)
    t.start()

    run_viewer(cfg, model, ctx, cmd_ref, args)


if __name__ == "__main__":
    main()
