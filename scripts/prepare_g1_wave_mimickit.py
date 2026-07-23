#!/usr/bin/env python3
"""Prepare the validated Xsens/GMR G1 wave clip for MimicKit."""

from __future__ import annotations

import argparse
import pickle
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch
from scipy import signal

WORKSPACE = Path(__file__).resolve().parents[1]
MIMICKIT_ROOT = WORKSPACE / "MimicKit"
sys.path.insert(0, str(MIMICKIT_ROOT))
sys.path.insert(0, str(MIMICKIT_ROOT / "mimickit"))

from mimickit.anim.motion import LoopMode, Motion  # noqa: E402
from mimickit.anim.mjcf_char_model import MJCFCharModel  # noqa: E402
from mimickit.util.torch_util import quat_to_exp_map  # noqa: E402

EXPECTED_DOFS = 29


def sanitize_asset(source: Path, destination: Path) -> None:
    tree = ET.parse(source)
    root = tree.getroot()

    motor_efforts = {}
    for joint in root.findall("./worldbody//joint"):
        force_range = joint.get("actuatorfrcrange")
        if force_range is not None:
            limits = np.fromstring(force_range, sep=" ")
            if limits.shape != (2,) or not np.isfinite(limits).all() or limits[0] >= limits[1]:
                raise ValueError(f"Invalid actuator force range for {joint.get('name')}: {force_range}")
            motor_efforts[joint.get("name")] = float(max(abs(limits[0]), abs(limits[1])))

    motors = root.findall("./actuator/motor")
    if len(motor_efforts) != EXPECTED_DOFS or len(motors) != EXPECTED_DOFS:
        raise ValueError(
            f"Expected {EXPECTED_DOFS} joint effort ranges and motors, got "
            f"{len(motor_efforts)} and {len(motors)}"
        )
    for motor in motors:
        joint_name = motor.get("joint")
        if joint_name not in motor_efforts:
            raise ValueError(f"Missing actuator force range for motor joint {joint_name}")
        motor.set("gear", f"{motor_efforts[joint_name]:g}")

    for node in root.findall("asset")[1:]:
        root.remove(node)
    for node in root.findall("worldbody")[1:]:
        root.remove(node)
    for tag in ("sensor", "visual", "statistic"):
        for node in list(root.findall(tag)):
            root.remove(node)
    for node in root.iter():
        node.attrib.pop("actuatorfrclimited", None)
        node.attrib.pop("actuatorfrcrange", None)

    destination.parent.mkdir(parents=True, exist_ok=True)
    tree.write(destination, encoding="unicode")


def validate_source(data: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    required = {"fps", "root_pos", "root_rot", "dof_pos"}
    missing = required - data.keys()
    if missing:
        raise ValueError(f"Missing GMR keys: {sorted(missing)}")

    fps = int(data["fps"])
    root_pos = np.asarray(data["root_pos"], dtype=np.float32)
    root_rot_wxyz = np.asarray(data["root_rot"], dtype=np.float32)
    dof_pos = np.asarray(data["dof_pos"], dtype=np.float32)

    if fps <= 0:
        raise ValueError(f"FPS must be positive, got {fps}")
    if root_pos.ndim != 2 or root_pos.shape[1] != 3:
        raise ValueError(f"Expected root_pos (N, 3), got {root_pos.shape}")
    if root_rot_wxyz.shape != (root_pos.shape[0], 4):
        raise ValueError(f"Expected root_rot (N, 4), got {root_rot_wxyz.shape}")
    if dof_pos.shape != (root_pos.shape[0], EXPECTED_DOFS):
        raise ValueError(f"Expected dof_pos (N, {EXPECTED_DOFS}), got {dof_pos.shape}")
    if root_pos.shape[0] < 2:
        raise ValueError("Motion must contain at least two frames")
    for name, value in (("root_pos", root_pos), ("root_rot", root_rot_wxyz), ("dof_pos", dof_pos)):
        if not np.isfinite(value).all():
            raise ValueError(f"{name} contains NaN or Inf")

    quat_norm = np.linalg.norm(root_rot_wxyz, axis=1)
    if not np.allclose(quat_norm, 1.0, atol=1e-4):
        raise ValueError(
            f"Root quaternions are not normalized: [{quat_norm.min()}, {quat_norm.max()}]"
        )
    return root_pos, root_rot_wxyz, dof_pos, fps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--motion",
        type=Path,
        default=WORKSPACE / "GMR/output/acenando/unitree_g1_acenando.pkl",
    )
    parser.add_argument(
        "--asset",
        type=Path,
        default=MIMICKIT_ROOT / "data/assets/g1/g1.xml",
        help="Installed MimicKit G1 asset used as the kinematic contract.",
    )
    parser.add_argument(
        "--install-asset-from",
        type=Path,
        default=None,
        help="Explicitly sanitize and install a replacement G1 asset before conversion.",
    )
    parser.add_argument(
        "--root-height-offset",
        type=float,
        default=0.02,
        help="Clearance added after aligning the lowest kinematic body to the ground.",
    )
    parser.add_argument(
        "--ground-motion",
        action="store_true",
        help="Align kinematic body origins to the ground (not suitable for every collision model).",
    )
    parser.add_argument(
        "--output-fps",
        type=int,
        default=30,
        help="Actual output sampling rate after low-pass filtering and resampling.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=240,
        help="Override legacy GMR FPS metadata (the old exporter truncated 240 to 239).",
    )
    args = parser.parse_args()

    asset_dir = MIMICKIT_ROOT / "data/assets/g1"
    motion_path = MIMICKIT_ROOT / "data/motions/g1/g1_wave_clamp.pkl"
    raw_motion_path = MIMICKIT_ROOT / "data/motions/g1/g1_wave_raw.pkl"

    with args.motion.open("rb") as stream:
        data = pickle.load(stream)
    root_pos, root_rot_wxyz, dof_pos, fps = validate_source(data)
    if args.fps is not None:
        if args.fps <= 0:
            raise ValueError(f"FPS override must be positive, got {args.fps}")
        fps = args.fps
    if args.output_fps <= 0 or args.output_fps > fps:
        raise ValueError(f"Output FPS must be in (0, {fps}], got {args.output_fps}")

    if args.install_asset_from is not None:
        source_meshes = args.install_asset_from.parent / "meshes"
        if not source_meshes.is_dir():
            raise FileNotFoundError(f"Missing G1 meshes: {source_meshes}")
        shutil.copytree(source_meshes, asset_dir / "meshes", dirs_exist_ok=True)
        sanitize_asset(args.install_asset_from, args.asset)
    elif not args.asset.is_file():
        raise FileNotFoundError(
            f"Missing installed G1 asset: {args.asset}. Install the official MimicKit data first."
        )

    # MuJoCo qpos and the Xsens GMR exporter use WXYZ; MimicKit expects XYZW.
    root_rot_xyzw = root_rot_wxyz[:, [1, 2, 3, 0]]
    root_exp_map = quat_to_exp_map(torch.from_numpy(root_rot_xyzw)).numpy()

    character = MJCFCharModel(device="cpu")
    character.load(args.asset)
    root_pos_tensor = torch.from_numpy(root_pos)
    root_rot_tensor = torch.from_numpy(root_rot_xyzw)
    dof_pos_tensor = torch.from_numpy(dof_pos)
    joint_rot = character.dof_to_rot(dof_pos_tensor)
    body_pos, _ = character.forward_kinematics(root_pos_tensor, root_rot_tensor, joint_rot)
    min_body_height = float(torch.min(body_pos[..., 2]).item())
    ground_shift = -min_body_height + args.root_height_offset if args.ground_motion else 0.0

    raw_frames = np.concatenate((root_pos, root_exp_map, dof_pos), axis=1).astype(np.float32)
    grounded_root_pos = root_pos.copy()
    grounded_root_pos[:, 2] += ground_shift
    source_frames = np.concatenate((grounded_root_pos, root_exp_map, dof_pos), axis=1).astype(np.float64)

    source_duration = (source_frames.shape[0] - 1) / fps
    if args.output_fps < fps:
        cutoff_hz = 0.4 * args.output_fps
        sos = signal.butter(4, cutoff_hz, btype="lowpass", fs=fps, output="sos")
        filtered_frames = signal.sosfiltfilt(sos, source_frames, axis=0)
        output_times = np.arange(int(np.floor(source_duration * args.output_fps)) + 1) / args.output_fps
        source_times = np.arange(source_frames.shape[0]) / fps
        frames = np.stack(
            [np.interp(output_times, source_times, filtered_frames[:, i])
             for i in range(filtered_frames.shape[1])],
            axis=1,
        ).astype(np.float32)
    else:
        frames = source_frames.astype(np.float32)

    asset_root = ET.parse(args.asset).getroot()
    asset_joints = asset_root.findall("./worldbody//joint")
    if len(asset_joints) != EXPECTED_DOFS:
        raise ValueError(f"Expected {EXPECTED_DOFS} asset joints, got {len(asset_joints)}")
    joint_limits = np.stack([
        np.fromstring(joint.get("range", ""), sep=" ") for joint in asset_joints
    ])
    if joint_limits.shape != (EXPECTED_DOFS, 2) or not np.isfinite(joint_limits).all():
        raise ValueError("Asset contains missing or invalid joint limits")
    unclipped_dof_pos = frames[:, 6:].copy()
    frames[:, 6:] = np.clip(frames[:, 6:], joint_limits[:, 0], joint_limits[:, 1])
    clip_delta = np.abs(frames[:, 6:] - unclipped_dof_pos)

    motion_path.parent.mkdir(parents=True, exist_ok=True)
    Motion(loop_mode=LoopMode.CLAMP, fps=fps, frames=raw_frames).save(raw_motion_path)
    Motion(loop_mode=LoopMode.CLAMP, fps=args.output_fps, frames=frames).save(motion_path)

    print(f"asset={asset_dir / 'g1.xml'}")
    print(f"raw_motion={raw_motion_path}")
    print(f"motion={motion_path}")
    print(f"min_body_height={min_body_height:.6f}m ground_shift={ground_shift:.6f}m")
    print(f"frames={frames.shape} dtype={frames.dtype} fps={args.output_fps}")
    print(f"joint_limit_clips={np.count_nonzero(clip_delta)} max_clip={clip_delta.max():.9f}rad")
    print(f"duration={(frames.shape[0] - 1) / args.output_fps:.6f}s loop_mode=CLAMP")


if __name__ == "__main__":
    main()
