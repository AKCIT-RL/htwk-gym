"""Headless render of the pre-trained Base Walk policy on the soccer field.

Produces an MP4 without opening a viewer (useful on machines without a GUI or
for CI artifacts). Usage:
    .venv/bin/python scripts/render_walk_on_field.py --command 0.5,0,0 --seconds 5
"""
import argparse, os, sys
import numpy as np, yaml, mujoco, imageio

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from envs.mujoco.soccer_scene import write_scene  # noqa: E402
from play_mujoco_soccer_walk import (  # noqa: E402
    SoccerSimCtx, JitPolicy, build_walk_obs, BASE_XML, CFG_FILE, POLICY_FILE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--command", default="0.5,0,0")
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--gait_freq", type=float, default=1.5)
    ap.add_argument("--out", default="videos/walk_on_field.mp4")
    args = ap.parse_args()

    cfg = yaml.load(open(CFG_FILE).read(), Loader=yaml.FullLoader)
    ctx = SoccerSimCtx(cfg, write_scene(BASE_XML))
    pol = JitPolicy(POLICY_FILE)
    dec = int(cfg["control"]["decimation"]); cdt = cfg["sim"]["dt"] * dec
    clip = cfg["normalization"]["clip_actions"]; sc = cfg["control"]["action_scale"]
    cmd = np.array([float(x) for x in args.command.split(",")], np.float32)

    ctx.reset()
    prev = np.zeros(cfg["env"]["num_actions"], np.float32)
    tgt = ctx.default_dof_pos.copy(); gait = 0.0
    ren = mujoco.Renderer(ctx.mj_model, height=480, width=640)
    cam = mujoco.MjvCamera(); cam.distance = 5; cam.elevation = -25; cam.azimuth = 90
    frames = []
    for it in range(int(args.seconds / cfg["sim"]["dt"])):
        st = ctx.read_state()
        if it % dec == 0:
            a = args.gait_freq if np.linalg.norm(cmd) > 1e-3 else 0.0
            obs = build_walk_obs(cfg, ctx, st, cmd, gait, a, prev)
            prev = np.clip(pol.act(obs), -clip, clip)
            tgt = ctx.default_dof_pos + sc * prev
            gait = (gait + cdt * a) % 1.0
        ctx.pd_step(tgt)
        if it % 10 == 0:
            cam.lookat[:] = st["base_pos"]
            ren.update_scene(ctx.mj_data, cam); frames.append(ren.render())
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    imageio.mimsave(args.out, frames, fps=25)
    print(f"[done] {args.out} ({len(frames)} frames); robot_x={st['base_pos'][0]:.2f}")


if __name__ == "__main__":
    main()
