#!/usr/bin/env python3
"""Emit deterministic diagnostics for the G1 wave MimicKit motion."""

from __future__ import annotations

import json
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
EPISODE_LENGTH = 10.0
ARM_JOINT_PREFIXES = ("left_shoulder", "left_elbow", "left_wrist", "right_shoulder", "right_elbow", "right_wrist")


def main() -> None:
    motion = load_motion(MOTION)
    assert motion.loop_mode == LoopMode.CLAMP

    root = ET.parse(ASSET).getroot()
    joint_names = [joint.get("name") for joint in root.findall("./worldbody//joint")]
    arm_indices = [i for i, name in enumerate(joint_names) if name.startswith(ARM_JOINT_PREFIXES)]
    assert len(joint_names) == 29
    assert len(arm_indices) == 14

    character = MJCFCharModel(device="cpu")
    character.load(ASSET)
    library = MotionLib(str(MOTION), character, "cpu")

    frame_times = torch.arange(motion.frames.shape[0], dtype=torch.float32) / motion.fps
    motion_ids = torch.zeros(frame_times.shape[0], dtype=torch.long)
    state = library.calc_motion_frame(motion_ids, frame_times)
    root_pos, _, root_vel, _, _, dof_vel = state

    arm_speed = torch.linalg.vector_norm(dof_vel[:, arm_indices], dim=1)
    dynamic_threshold = 0.25
    duration = library.get_total_length()
    clamp_fraction_without_truncation = EPISODE_LENGTH / (2.0 * duration)

    report = {
        "frames": int(motion.frames.shape[0]),
        "fps": int(motion.fps),
        "duration_s": duration,
        "episode_length_s": EPISODE_LENGTH,
        "arm_dofs": [joint_names[i] for i in arm_indices],
        "arm_speed_mean_rad_s": float(arm_speed.mean()),
        "arm_speed_p95_rad_s": float(torch.quantile(arm_speed, 0.95)),
        "arm_dynamic_fraction": float((arm_speed > dynamic_threshold).float().mean()),
        "root_horizontal_drift_m": float(torch.linalg.vector_norm(root_pos[-1, :2] - root_pos[0, :2])),
        "root_speed_p95_m_s": float(torch.quantile(torch.linalg.vector_norm(root_vel, dim=1), 0.95)),
        "clamp_fraction_without_truncated_resets": clamp_fraction_without_truncation,
        "reset_time_max_s_with_truncation": duration - EPISODE_LENGTH,
        "finite": all(torch.isfinite(tensor).all().item() for tensor in state),
    }

    assert report["finite"]
    assert report["reset_time_max_s_with_truncation"] >= 0.0
    assert report["arm_dynamic_fraction"] > 0.01, "Motion has no measurable arm activity"
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
