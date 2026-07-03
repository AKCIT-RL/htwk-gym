"""Drive the pre-trained Base Walk Extended policy on the soccer field (MuJoCo).

MJ.5 sanity check: the walk policy from htwk-gym walking around the full
RoboCup Adult-size field with a Size-5 ball and physical goals. No kicking
intelligence yet — you steer, the ball reacts to contact.

Viewer keys (focus the MuJoCo window):
    W / S    vx +/- 0.1 m/s        A / D    wz +/- 0.1 rad/s
    Q / E    vy +/- 0.1 m/s        Space    stop
    R        reset robot+ball      B        ball to random field position
    K        nudge ball away from robot (fake kick, for goal-detection demo)
    P        print state
Terminal: type "vx vy wz" + Enter (e.g. "0.5 0 0").

Usage:
    .venv/bin/python play_mujoco_soccer_walk.py
    .venv/bin/python play_mujoco_soccer_walk.py --command 0.4,0,0
"""

import argparse
import os
import sys
import threading
import time

import numpy as np
import torch
import yaml
import mujoco

from envs.mujoco.soccer_scene import (
    write_scene, is_goal, is_out_of_bounds,
    FIELD_LENGTH, FIELD_WIDTH, BALL_RADIUS,
)

ROOT = os.path.dirname(os.path.abspath(__file__))
BASE_XML = os.path.join(ROOT, "resources", "T1", "T1_locomotion.xml")
CFG_FILE = os.path.join(ROOT, "configs", "Base_Walk_Extended.yaml")
POLICY_FILE = os.path.join(ROOT, "models", "base_walk_extended.pt")


# ----------------------------------------------------------------- math ----
def quat_rotate_inverse(q_xyzw: np.ndarray, v: np.ndarray) -> np.ndarray:
    q_w = q_xyzw[3]
    q_vec = q_xyzw[:3]
    a = v * (2.0 * q_w ** 2 - 1.0)
    b = np.cross(q_vec, v) * (q_w * 2.0)
    c = q_vec * (np.dot(q_vec, v) * 2.0)
    return a - b + c


# ------------------------------------------------------------- policy ------
class JitPolicy:
    def __init__(self, path: str):
        self.mod = torch.jit.load(path, map_location="cpu")
        self.mod.eval()

    def act(self, obs: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            out = self.mod(torch.from_numpy(obs).unsqueeze(0))
        if isinstance(out, (tuple, list)):
            out = out[0]
        return out.squeeze(0).numpy()


# ---------------------------------------------------------------- sim ------
class SoccerSimCtx:
    """MuJoCo scene (robot + ball + goals) + actuator metadata."""

    def __init__(self, cfg: dict, scene_xml: str):
        self.mj_model = mujoco.MjModel.from_xml_path(scene_xml)
        self.mj_model.opt.timestep = cfg["sim"]["dt"]
        self.mj_data = mujoco.MjData(self.mj_model)
        mujoco.mj_resetData(self.mj_model, self.mj_data)
        self.cfg = cfg

        nu = self.mj_model.nu
        self.n_dof = nu
        self.default_dof_pos = np.zeros(nu, dtype=np.float32)
        self.dof_stiffness = np.zeros(nu, dtype=np.float32)
        self.dof_damping = np.zeros(nu, dtype=np.float32)

        joint_angles = cfg["init_state"]["default_joint_angles"]
        stiffness_cfg = cfg["control"]["stiffness"]
        damping_cfg = cfg["control"]["damping"]
        for i in range(nu):
            name = mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            chosen = None
            for key, val in joint_angles.items():
                if key != "default" and key in name:
                    chosen = float(val)
                    break
            self.default_dof_pos[i] = (
                chosen if chosen is not None else float(joint_angles.get("default", 0.0))
            )
            ok = False
            for key in stiffness_cfg:
                if key in name:
                    self.dof_stiffness[i] = float(stiffness_cfg[key])
                    self.dof_damping[i] = float(damping_cfg[key])
                    ok = True
                    break
            if not ok:
                raise ValueError(f"No PD gain for actuator '{name}'")

        # Ball freejoint address in qpos/qvel.
        jid = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, "ball_freejoint")
        if jid < 0:
            raise ValueError("ball_freejoint not found in scene")
        self.ball_qpos_adr = self.mj_model.jnt_qposadr[jid]
        self.ball_qvel_adr = self.mj_model.jnt_dofadr[jid]

    # Robot free joint occupies qpos[0:7]/qvel[0:6] (robot body defined first).
    @property
    def robot_qpos(self):
        return slice(7, 7 + self.n_dof)

    @property
    def robot_qvel(self):
        return slice(6, 6 + self.n_dof)

    # ------------------------------------------------------------ ball ----
    def ball_pos(self) -> np.ndarray:
        a = self.ball_qpos_adr
        return self.mj_data.qpos[a:a + 3].astype(np.float32)

    def set_ball(self, x: float, y: float, vx: float = 0.0, vy: float = 0.0):
        a, v = self.ball_qpos_adr, self.ball_qvel_adr
        self.mj_data.qpos[a:a + 3] = [x, y, BALL_RADIUS]
        self.mj_data.qpos[a + 3:a + 7] = [1, 0, 0, 0]
        self.mj_data.qvel[v:v + 6] = [vx, vy, 0, 0, 0, 0]
        mujoco.mj_forward(self.mj_model, self.mj_data)

    def reset_ball_random(self, rng: np.random.Generator):
        x = float(rng.uniform(-FIELD_LENGTH / 2 * 0.8, FIELD_LENGTH / 2 * 0.8))
        y = float(rng.uniform(-FIELD_WIDTH / 2 * 0.8, FIELD_WIDTH / 2 * 0.8))
        self.set_ball(x, y)
        print(f"[ball] re-spawned at ({x:+.2f}, {y:+.2f})", flush=True)

    # ----------------------------------------------------------- robot ----
    def reset(self):
        d = self.mj_data
        init = self.cfg["init_state"]
        d.qpos[0:3] = np.array(init["pos"], dtype=np.float32)
        rot_xyzw = np.array(init["rot"], dtype=np.float32)
        d.qpos[3:7] = rot_xyzw[[3, 0, 1, 2]]           # xyzw -> wxyz
        d.qpos[self.robot_qpos] = self.default_dof_pos
        d.qvel[:] = 0.0
        d.xfrc_applied[:] = 0.0
        self.set_ball(1.0, 0.0)
        mujoco.mj_forward(self.mj_model, d)

    def read_state(self) -> dict:
        d = self.mj_data
        q_wxyz = d.qpos[3:7].astype(np.float32)
        q_xyzw = q_wxyz[[1, 2, 3, 0]]
        return dict(
            base_pos=d.qpos[0:3].astype(np.float32),
            base_quat_xyzw=q_xyzw,
            proj_grav=quat_rotate_inverse(q_xyzw, np.array([0.0, 0.0, -1.0], np.float32)),
            ang_vel=d.sensor("angular-velocity").data.astype(np.float32),
            dof_pos=d.qpos[self.robot_qpos].astype(np.float32),
            dof_vel=d.qvel[self.robot_qvel].astype(np.float32),
        )

    def pd_step(self, dof_targets: np.ndarray):
        d = self.mj_data
        dof_pos = d.qpos[self.robot_qpos]
        dof_vel = d.qvel[self.robot_qvel]
        torque = self.dof_stiffness * (dof_targets - dof_pos) - self.dof_damping * dof_vel
        d.ctrl[:] = np.clip(
            torque,
            self.mj_model.actuator_ctrlrange[:, 0],
            self.mj_model.actuator_ctrlrange[:, 1],
        )
        mujoco.mj_step(self.mj_model, d)


# ---------------------------------------------------------------- obs ------
def build_walk_obs(cfg, ctx, state, cmd, gait_process, gait_freq_active, prev_actions):
    """53-dim obs matching envs/T1/base_walk.py with extended_obs=True."""
    norm = cfg["normalization"]
    n = ctx.n_dof
    obs = np.zeros(cfg["env"]["num_observations"], dtype=np.float32)
    gait_active = float(gait_freq_active > 1.0e-8)
    phase = 2.0 * np.pi * gait_process
    cmd_scale = np.array([norm["lin_vel"], norm["lin_vel"], norm["ang_vel"]], np.float32)

    obs[0:3] = state["proj_grav"] * norm["gravity"]
    obs[3:6] = state["ang_vel"] * norm["ang_vel"]
    obs[6:9] = cmd * cmd_scale
    obs[9] = np.cos(phase) * gait_active
    obs[10] = np.sin(phase) * gait_active
    # obs[11:17] = 0 (ball/target slots, zeroed for walk)
    obs[17:17 + n] = (state["dof_pos"] - ctx.default_dof_pos) * norm["dof_pos"]
    obs[17 + n:17 + 2 * n] = state["dof_vel"] * norm["dof_vel"]
    obs[17 + 2 * n:17 + 3 * n] = prev_actions
    return obs


# ------------------------------------------------------------ command ------
class CommandRef:
    def __init__(self, initial):
        self._cmd = np.array(initial, dtype=np.float32)
        self._lock = threading.Lock()

    def get(self):
        with self._lock:
            return self._cmd.copy()

    def set(self, vx, vy, wz):
        with self._lock:
            self._cmd[:] = [vx, vy, wz]

    def nudge(self, dvx=0.0, dvy=0.0, dwz=0.0):
        with self._lock:
            self._cmd[0] = float(np.clip(self._cmd[0] + dvx, -1.0, 1.0))
            self._cmd[1] = float(np.clip(self._cmd[1] + dvy, -1.0, 1.0))
            self._cmd[2] = float(np.clip(self._cmd[2] + dwz, -1.0, 1.0))

    def stop(self):
        with self._lock:
            self._cmd[:] = 0.0

    def print_state(self):
        with self._lock:
            c = self._cmd
            print(f"[CMD] vx={c[0]:+.2f} vy={c[1]:+.2f} wz={c[2]:+.2f}", flush=True)


def _stdin_reader(cmd_ref):
    for line in sys.stdin:
        parts = line.strip().replace(",", " ").split()
        if not parts:
            continue
        try:
            vals = [float(p) for p in parts]
        except ValueError:
            print(f"[CMD] could not parse '{line.strip()}'", flush=True)
            continue
        vals += [0.0] * (3 - len(vals))
        cmd_ref.set(*vals[:3])
        cmd_ref.print_state()


# ---------------------------------------------------------------- main -----
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--command", default="0,0,0", help="initial vx,vy,wz")
    parser.add_argument("--gait_freq", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    with open(CFG_FILE, "r", encoding="utf-8") as f:
        cfg = yaml.load(f.read(), Loader=yaml.FullLoader)

    scene_xml = write_scene(BASE_XML)
    print(f"[scene] generated {scene_xml}")

    ctx = SoccerSimCtx(cfg, scene_xml)
    policy = JitPolicy(POLICY_FILE)
    print(f"[policy] loaded {POLICY_FILE}")

    rng = np.random.default_rng(args.seed)
    cmd_ref = CommandRef([float(x) for x in args.command.split(",")])
    threading.Thread(target=_stdin_reader, args=(cmd_ref,), daemon=True).start()

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
    score = {"east": 0}

    def key_callback(keycode):
        nonlocal prev_actions, dof_targets
        try:
            ch = chr(keycode).upper()
        except (ValueError, OverflowError):
            return
        if ch == "W":
            cmd_ref.nudge(dvx=+0.1); cmd_ref.print_state()
        elif ch == "S":
            cmd_ref.nudge(dvx=-0.1); cmd_ref.print_state()
        elif ch == "A":
            cmd_ref.nudge(dwz=+0.1); cmd_ref.print_state()
        elif ch == "D":
            cmd_ref.nudge(dwz=-0.1); cmd_ref.print_state()
        elif ch == "Q":
            cmd_ref.nudge(dvy=+0.1); cmd_ref.print_state()
        elif ch == "E":
            cmd_ref.nudge(dvy=-0.1); cmd_ref.print_state()
        elif ch == " ":
            cmd_ref.stop(); cmd_ref.print_state()
        elif ch == "R":
            ctx.reset()
            prev_actions[:] = 0.0
            dof_targets[:] = ctx.default_dof_pos
            print("[reset] robot + ball", flush=True)
        elif ch == "B":
            ctx.reset_ball_random(rng)
        elif ch == "K":
            # fake kick: push ball away from the robot toward east goal
            bp = ctx.ball_pos()
            goal = np.array([FIELD_LENGTH / 2, 0.0], np.float32)
            d = goal - bp[:2]
            d = d / (np.linalg.norm(d) + 1e-6) * 4.0
            ctx.set_ball(bp[0], bp[1], vx=float(d[0]), vy=float(d[1]))
            print("[kick] ball pushed toward east goal", flush=True)
        elif ch == "P":
            st = ctx.read_state()
            bp = ctx.ball_pos()
            cmd_ref.print_state()
            print(f"[state] robot={st['base_pos'][:2]} ball={bp[:2]} score={score}", flush=True)

    print(
        "\n[Viewer] W/S vx  A/D wz  Q/E vy  Space stop  R reset  "
        "B ball random  K fake-kick  P print\n", flush=True,
    )

    import mujoco.viewer
    with mujoco.viewer.launch_passive(ctx.mj_model, ctx.mj_data,
                                      key_callback=key_callback) as viewer:
        viewer.cam.elevation = -25
        viewer.cam.distance = 6.0
        it = 0
        while viewer.is_running():
            step_start = time.time()
            state = ctx.read_state()

            if it % decimation == 0:
                cmd = cmd_ref.get()
                speed = float(np.linalg.norm(cmd))
                gait_freq_active = gait_freq if speed > 1e-3 else 0.0
                obs = build_walk_obs(cfg, ctx, state, cmd, gait_process,
                                     gait_freq_active, prev_actions)
                prev_actions = np.clip(policy.act(obs), -clip_actions, clip_actions)
                dof_targets = ctx.default_dof_pos + action_scale * prev_actions
                gait_process = (gait_process + control_dt * gait_freq_active) % 1.0

                # Episode logic demo: goal / out detection with ball-only reset.
                bp = ctx.ball_pos()
                if is_goal(bp[:2], attacking_east=True):
                    score["east"] += 1
                    print(f"[GOAL!] score={score}", flush=True)
                    ctx.reset_ball_random(rng)
                elif is_out_of_bounds(bp[:2]):
                    print("[out] ball out of bounds", flush=True)
                    ctx.reset_ball_random(rng)

            ctx.pd_step(dof_targets)
            viewer.cam.lookat[:] = state["base_pos"]
            viewer.sync()
            it += 1

            # real-time pacing
            leftover = ctx.mj_model.opt.timestep - (time.time() - step_start)
            if leftover > 0:
                time.sleep(leftover)


if __name__ == "__main__":
    main()
