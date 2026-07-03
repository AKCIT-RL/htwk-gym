"""Headless sanity: the pre-trained Base Walk policy stays upright and moves
forward on the soccer field scene (MJ.5 acceptance, no viewer)."""

import os
import sys

import mujoco
import numpy as np
import pytest
import torch
import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from envs.mujoco.soccer_scene import write_scene  # noqa: E402
from play_mujoco_soccer_walk import (  # noqa: E402
    SoccerSimCtx, JitPolicy, build_walk_obs, BASE_XML, CFG_FILE, POLICY_FILE,
)


@pytest.fixture(scope="module")
def setup():
    with open(CFG_FILE, encoding="utf-8") as f:
        cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
    ctx = SoccerSimCtx(cfg, write_scene(BASE_XML))
    policy = JitPolicy(POLICY_FILE)
    return cfg, ctx, policy


def _run(cfg, ctx, policy, cmd, seconds, gait_freq=1.5):
    ctx.reset()
    decimation = int(cfg["control"]["decimation"])
    control_dt = float(cfg["sim"]["dt"]) * decimation
    clip = float(cfg["normalization"]["clip_actions"])
    scale = float(cfg["control"]["action_scale"])
    prev = np.zeros(cfg["env"]["num_actions"], dtype=np.float32)
    targets = ctx.default_dof_pos.copy()
    gait = 0.0
    cmd = np.array(cmd, dtype=np.float32)
    steps = int(seconds / cfg["sim"]["dt"])
    for it in range(steps):
        state = ctx.read_state()
        if state["proj_grav"][2] > -0.5:      # fell over
            return state, it * cfg["sim"]["dt"], False
        if it % decimation == 0:
            active = gait_freq if np.linalg.norm(cmd) > 1e-3 else 0.0
            obs = build_walk_obs(cfg, ctx, state, cmd, gait, active, prev)
            prev = np.clip(policy.act(obs), -clip, clip)
            targets = ctx.default_dof_pos + scale * prev
            gait = (gait + control_dt * active) % 1.0
        ctx.pd_step(targets)
    return ctx.read_state(), seconds, True


def test_stand_still(setup):
    cfg, ctx, policy = setup
    state, t, ok = _run(cfg, ctx, policy, [0, 0, 0], seconds=3.0)
    assert ok, f"robot fell while standing at t={t:.2f}s"


def test_walk_forward(setup):
    cfg, ctx, policy = setup
    state, t, ok = _run(cfg, ctx, policy, [0.4, 0, 0], seconds=8.0)
    assert ok, f"robot fell while walking at t={t:.2f}s"
    dist = float(state["base_pos"][0])
    assert dist > 1.0, f"robot only advanced {dist:.2f} m in 8 s at 0.4 m/s"


def test_walk_does_not_leave_field_frame(setup):
    """Sanity: field plane supports the robot anywhere (no holes)."""
    cfg, ctx, policy = setup
    state, t, ok = _run(cfg, ctx, policy, [0.4, 0, 0.3], seconds=6.0)
    assert ok, f"robot fell while curving at t={t:.2f}s"
