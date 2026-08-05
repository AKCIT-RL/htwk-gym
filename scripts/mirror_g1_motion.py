#!/usr/bin/env python3
"""Mirror a G1 MimicKit motion across the XZ plane (left <-> right)."""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch

WORKSPACE = Path(__file__).resolve().parents[1]
MIMICKIT_ROOT = WORKSPACE / "MimicKit"
sys.path.insert(0, str(MIMICKIT_ROOT))
sys.path.insert(0, str(MIMICKIT_ROOT / "mimickit"))

from mimickit.anim.motion import Motion, load_motion  # noqa: E402
from mimickit.anim.mjcf_char_model import MJCFCharModel  # noqa: E402
from mimickit.util.torch_util import exp_map_to_quat  # noqa: E402


def build_mirror_map(asset: Path) -> tuple[list[int], np.ndarray, list[str]]:
    root = ET.parse(asset).getroot()
    joints = root.findall("./worldbody//joint")
    names = [j.get("name") for j in joints]
    axes = []
    for j in joints:
        axis = np.fromstring(j.get("axis", "0 0 1"), sep=" ")
        axes.append(axis / np.linalg.norm(axis))

    perm, signs = [], []
    for name, axis in zip(names, axes):
        if name.startswith("left_"):
            partner = "right_" + name[len("left_"):]
        elif name.startswith("right_"):
            partner = "left_" + name[len("right_"):]
        else:
            partner = name
        j = names.index(partner)
        if not np.allclose(axes[j], axis, atol=1e-6):
            raise ValueError(f"Axis mismatch {name} vs {partner}: {axis} vs {axes[j]}")
        # reflection across XZ: rotations about y keep sign, about x/z negate
        sign = 1.0 if abs(axis[1]) > 0.99 else -1.0
        perm.append(j)
        signs.append(sign)
    return perm, np.asarray(signs, dtype=np.float32), names


def mirror_frames(frames: np.ndarray, perm: list[int], signs: np.ndarray) -> np.ndarray:
    out = frames.copy()
    out[:, 1] = -frames[:, 1]                      # root pos y
    out[:, 3] = -frames[:, 3]                      # root exp-map x
    out[:, 5] = -frames[:, 5]                      # root exp-map z
    out[:, 6:] = frames[:, 6:][:, perm] * signs    # dof swap + per-axis sign
    return out


def verify(asset: Path, original: np.ndarray, mirrored: np.ndarray) -> float:
    ch = MJCFCharModel(device="cpu")
    ch.load(asset)

    def fk(fr):
        t = torch.from_numpy(fr)
        pos, _ = ch.forward_kinematics(t[:, :3], exp_map_to_quat(t[:, 3:6]), ch.dof_to_rot(t[:, 6:]))
        return pos.numpy()

    body_names = [ch.get_body_name(i) for i in range(fk(original[:1]).shape[1])]
    body_perm = []
    for n in body_names:
        if n.startswith("left_"):
            body_perm.append(body_names.index("right_" + n[len("left_"):]))
        elif n.startswith("right_"):
            body_perm.append(body_names.index("left_" + n[len("right_"):]))
        else:
            body_perm.append(body_names.index(n))

    ref = fk(original)[:, body_perm]
    ref[..., 1] = -ref[..., 1]
    err = np.abs(fk(mirrored) - ref).max()
    return float(err)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--asset", type=Path, default=MIMICKIT_ROOT / "data/assets/g1/g1.xml")
    args = parser.parse_args()

    m = load_motion(str(args.motion))
    frames = np.asarray(m.frames, dtype=np.float32)
    perm, signs, names = build_mirror_map(args.asset)
    mirrored = mirror_frames(frames, perm, signs)

    joints = ET.parse(args.asset).getroot().findall("./worldbody//joint")
    limits = np.stack([np.fromstring(j.get("range", ""), sep=" ") for j in joints])
    violations = np.maximum(limits[:, 0] - mirrored[:, 6:], mirrored[:, 6:] - limits[:, 1])
    max_violation = float(violations.max())
    mirrored[:, 6:] = np.clip(mirrored[:, 6:], limits[:, 0], limits[:, 1])

    err = verify(args.asset, frames, mirrored)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    Motion(loop_mode=m.loop_mode, fps=m.fps, frames=mirrored).save(str(args.output))
    print(f"output={args.output}")
    print(f"frames={mirrored.shape} fps={m.fps}")
    print(f"fk_mirror_error={err:.9f}m")
    print(f"joint_limit_max_violation={max_violation:.9f}rad")
    if err > 1e-4:
        raise SystemExit("FK mirror verification failed")


if __name__ == "__main__":
    main()
