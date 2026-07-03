"""Unit tests for the MuJoCo soccer field scene (run on Mac, no GPU needed)."""

import os
import sys

import mujoco
import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from envs.mujoco.soccer_scene import (  # noqa: E402
    build_scene_xml, write_scene, is_goal, is_out_of_bounds,
    FIELD_LENGTH, FIELD_WIDTH, GOAL_WIDTH, BALL_RADIUS, BALL_MASS,
)

BASE_XML = os.path.join(ROOT, "resources", "T1", "T1_locomotion.xml")


@pytest.fixture(scope="module")
def model():
    path = write_scene(BASE_XML)
    return mujoco.MjModel.from_xml_path(path)


# ------------------------------------------------------------ scene ---------
def test_scene_compiles(model):
    assert model.nu == 12  # 12 leg actuators untouched


def test_scene_has_ball_goals_field(model):
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
             for i in range(model.ngeom)]
    assert "ball_geom" in names
    assert "field" in names
    for g in ("goal_east_post_l", "goal_east_post_r", "goal_east_crossbar",
              "goal_west_post_l", "goal_west_post_r", "goal_west_crossbar"):
        assert g in names, f"missing {g}"


def test_ball_properties(model):
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ball_geom")
    assert model.geom_size[gid][0] == pytest.approx(BALL_RADIUS)
    bid = model.geom_bodyid[gid]
    assert model.body_mass[bid] == pytest.approx(BALL_MASS, rel=1e-3)


def test_goal_posts_positions(model):
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "goal_east")
    assert model.body_pos[gid][0] == pytest.approx(FIELD_LENGTH / 2)


def test_robot_untouched_by_injection():
    """Injection must not alter the robot part of the MJCF."""
    xml = build_scene_xml(BASE_XML)
    with open(BASE_XML, encoding="utf-8") as f:
        base = f.read()
    # every joint definition of the robot survives verbatim
    for line in base.splitlines():
        if "<joint name=" in line:
            assert line.strip() in xml


# ------------------------------------------------------- ball physics -------
def _ball_adr(model):
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "ball_freejoint")
    return model.jnt_qposadr[jid], model.jnt_dofadr[jid]


def _isolate_robot(data, model):
    """Drop the robot far away so it doesn't interact with the ball."""
    data.qpos[0:3] = [-20.0, -20.0, 5.0]


def test_ball_damped_bounce(model):
    """Dropped from 1 m the ball must bounce a little (contact is elastic
    enough to not stick) but not tunnel through the field. MuJoCo's default
    soft contacts give a modest bounce at our 0.002 s timestep; a realistic
    high-restitution bounce is out of scope because the ball almost always
    rolls in soccer. We just assert a sane, non-degenerate contact."""
    data = mujoco.MjData(model)
    qadr, vadr = _ball_adr(model)
    _isolate_robot(data, model)
    data.qpos[qadr:qadr + 3] = [0, 0, 1.0]
    mujoco.mj_forward(model, data)

    max_h_after_bounce = 0.0
    bounced = False
    for _ in range(4000):  # 8 s at dt=0.002
        mujoco.mj_step(model, data)
        z = data.qpos[qadr + 2]
        vz = data.qvel[vadr + 2]
        if not bounced and vz > 0.1:
            bounced = True
        if bounced:
            max_h_after_bounce = max(max_h_after_bounce, z)
    ratio = (max_h_after_bounce - BALL_RADIUS) / (1.0 - BALL_RADIUS)
    assert bounced, "ball never bounced (stuck to the ground)"
    assert 0.02 <= ratio <= 0.30, f"bounce ratio {ratio:.2f} out of MuJoCo range"


def test_ball_rolls_and_stops(model):
    """Kicked at 3 m/s the ball must roll several metres and eventually stop."""
    data = mujoco.MjData(model)
    qadr, vadr = _ball_adr(model)
    _isolate_robot(data, model)
    data.qpos[qadr:qadr + 3] = [-5, 0, BALL_RADIUS]
    data.qvel[vadr] = 3.0
    mujoco.mj_forward(model, data)

    for _ in range(15000):  # 30 s
        mujoco.mj_step(model, data)
    travelled = data.qpos[qadr] - (-5)
    speed = abs(data.qvel[vadr])
    assert travelled > 2.0, f"ball only travelled {travelled:.2f} m"
    assert speed < 0.05, f"ball still moving at {speed:.2f} m/s after 30 s"


def test_ball_stopped_by_goal_post(model):
    """A ball aimed exactly at a post must not pass through it."""
    data = mujoco.MjData(model)
    qadr, vadr = _ball_adr(model)
    _isolate_robot(data, model)
    post_y = GOAL_WIDTH / 2
    data.qpos[qadr:qadr + 3] = [FIELD_LENGTH / 2 - 1.0, post_y, BALL_RADIUS]
    data.qvel[vadr] = 6.0
    mujoco.mj_forward(model, data)
    for _ in range(1000):
        mujoco.mj_step(model, data)
    # ball must have rebounded (x < goal line) instead of tunnelling deep behind
    assert data.qpos[qadr] < FIELD_LENGTH / 2 + 0.3


# ------------------------------------------------------ field geometry ------
def test_is_goal():
    assert is_goal([FIELD_LENGTH / 2 + 0.2, 0.0])
    assert is_goal([FIELD_LENGTH / 2 + 0.2, GOAL_WIDTH / 2 - 0.05])
    assert not is_goal([FIELD_LENGTH / 2 + 0.2, GOAL_WIDTH / 2 + 0.2])  # wide
    assert not is_goal([FIELD_LENGTH / 2 - 0.5, 0.0])                    # in play
    assert is_goal([-(FIELD_LENGTH / 2 + 0.2), 0.0], attacking_east=False)


def test_is_out_of_bounds():
    assert not is_out_of_bounds([0, 0])
    assert is_out_of_bounds([0, FIELD_WIDTH / 2 + 0.3])
    assert is_out_of_bounds([FIELD_LENGTH / 2 + 0.3, GOAL_WIDTH / 2 + 0.5])
    # inside the goal mouth is NOT out (it's a goal)
    assert not is_out_of_bounds([FIELD_LENGTH / 2 + 0.2, 0.0])
