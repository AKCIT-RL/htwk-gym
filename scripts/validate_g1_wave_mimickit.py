#!/usr/bin/env python3
"""Validate the G1 wave asset and MimicKit motion contract on CPU."""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch

WORKSPACE = Path(__file__).resolve().parents[1]
MIMICKIT_ROOT = WORKSPACE / "MimicKit"
sys.path.insert(0, str(MIMICKIT_ROOT / "mimickit"))

from anim.mjcf_char_model import MJCFCharModel  # noqa: E402
from anim.motion import LoopMode, load_motion  # noqa: E402
from anim.motion_lib import MotionLib  # noqa: E402

ASSET = MIMICKIT_ROOT / "data/assets/g1/g1.xml"
MOTION = MIMICKIT_ROOT / "data/motions/g1/g1_wave_clamp.pkl"
EXPECTED_FRAMES = 454
EXPECTED_FPS = 30
EXPECTED_DOFS = 29


def validate_joint_limits() -> np.ndarray:
    root = ET.parse(ASSET).getroot()
    joints = root.findall("./worldbody//joint")
    assert len(joints) == EXPECTED_DOFS
    limits_by_name = {}
    for joint in joints:
        joint_range = joint.get("range")
        assert joint_range is not None, f"Missing range for joint {joint.get('name')}"
        limits = np.fromstring(joint_range, sep=" ")
        assert limits.shape == (2,), f"Invalid range for joint {joint.get('name')}: {joint_range}"
        assert np.isfinite(limits).all()
        assert limits[0] < limits[1], f"Invalid limits for joint {joint.get('name')}: {limits}"
        assert float(joint.get("stiffness", "0")) > 0, f"Missing stiffness for {joint.get('name')}"
        assert float(joint.get("damping", "0")) > 0, f"Missing damping for {joint.get('name')}"
        limits_by_name[joint.get("name")] = limits

    motors = root.findall("./actuator/motor")
    assert len(motors) == EXPECTED_DOFS
    assert [motor.get("joint") for motor in motors] == [joint.get("name") for joint in joints]
    motor_gears = np.array([float(motor.get("gear", "0")) for motor in motors])
    assert np.isfinite(motor_gears).all()
    assert (motor_gears > 0).all()
    return np.stack([limits_by_name[joint.get("name")] for joint in joints])


def main() -> None:
    joint_limits = validate_joint_limits()

    motion = load_motion(MOTION)
    assert motion.loop_mode == LoopMode.CLAMP
    assert motion.fps == EXPECTED_FPS
    assert motion.frames.shape == (EXPECTED_FRAMES, 6 + EXPECTED_DOFS)
    assert motion.frames.dtype == np.float32
    assert np.isfinite(motion.frames).all()
    dof_pos = motion.frames[:, 6:]
    assert np.all(dof_pos >= joint_limits[:, 0] - 1e-5), "Motion violates lower joint limits"
    assert np.all(dof_pos <= joint_limits[:, 1] + 1e-5), "Motion violates upper joint limits"

    character = MJCFCharModel(device="cpu")
    character.load(ASSET)
    assert character.get_dof_size() == EXPECTED_DOFS

    required_bodies = {
        "left_ankle_roll_link",
        "right_ankle_roll_link",
        "head_link",
        "left_wrist_yaw_link",
        "right_wrist_yaw_link",
        "left_knee_link",
        "left_ankle_pitch_link",
        "right_knee_link",
        "right_ankle_pitch_link",
    }
    missing = required_bodies - set(character.get_body_names())
    assert not missing, f"Missing configured bodies: {sorted(missing)}"

    library = MotionLib(str(MOTION), character, "cpu")
    assert library.get_num_motions() == 1
    assert library.get_motion_frame_size() == 6 + EXPECTED_DOFS
    expected_length = (EXPECTED_FRAMES - 1) / EXPECTED_FPS
    assert abs(library.get_total_length() - expected_length) < 1e-5

    motion_ids = torch.zeros(5, dtype=torch.long)
    times = torch.linspace(0.0, expected_length, 5)
    state = library.calc_motion_frame(motion_ids, times)
    for tensor in state:
        assert torch.isfinite(tensor).all()

    print("G1_WAVE_CONTRACT_OK")
    print(f"bodies={character.get_num_joints()} dofs={character.get_dof_size()}")
    print(f"frames={motion.frames.shape} fps={motion.fps} length={library.get_total_length():.6f}s")
    print(f"sample_shapes={[tuple(tensor.shape) for tensor in state]}")


if __name__ == "__main__":
    main()
