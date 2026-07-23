#!/usr/bin/env python3
"""Run deterministic reference-pose tracking for the G1 wave motion."""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

WORKSPACE = Path(__file__).resolve().parents[1]
MIMICKIT = WORKSPACE / "MimicKit"
os.chdir(MIMICKIT)
sys.path.insert(0, str(MIMICKIT / "mimickit"))

from envs import env_builder  # noqa: E402

import torch  # noqa: E402

from envs import base_env  # noqa: E402
from envs.deepmimic_env import DeepMimicEnv  # noqa: E402
from util import torch_util, util  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-config", default="data/envs/amp_g1_wave_overfit_env.yaml")
    parser.add_argument("--engine-config", default="data/engines/isaac_gym_engine.yaml")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--horizon", type=float, default=10.0)
    parser.add_argument("--output-dir", default="output/g1_wave_pd_tracking")
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--zero-reset-root-velocity", action="store_true")
    parser.add_argument("--root-height-offset", type=float, default=0.0)
    return parser.parse_args()


def scalar(value):
    return float(value.detach().cpu().item())


def main():
    args = parse_args()
    util.set_rand_seed(args.seed)

    env_config, engine_config = env_builder.load_configs(args.env_config, args.engine_config)
    env_config = env_config.copy()
    env_config["env_name"] = "deepmimic"
    env_config["rand_reset"] = False
    env_config["truncate_reset_time"] = False
    env_config["visualize_ref_char"] = False
    env_config["episode_length"] = args.horizon

    env = DeepMimicEnv(
        env_config=env_config, engine_config=engine_config, num_envs=1,
        device=args.device, visualize=False, record_video=args.video,
    )
    env.set_mode(base_env.EnvMode.TEST)
    env.reset()

    char_id = env._get_char_id()
    env_ids = torch.zeros(1, dtype=torch.long, device=args.device)
    if args.zero_reset_root_velocity:
        env._engine.set_root_vel(env_ids, char_id, 0.0)
        env._engine.set_root_ang_vel(env_ids, char_id, 0.0)
    if args.root_height_offset != 0.0:
        root_pos = env._ref_root_pos.clone()
        root_pos[:, 2] += args.root_height_offset
        env._engine.set_root_pos(env_ids, char_id, root_pos)

    dt = env._engine.get_timestep()
    rows = []
    done_reason = "horizon"
    termination_contacts = []
    max_steps = int(np.ceil(args.horizon / dt))

    for step in range(max_steps):
        target_time = env._get_motion_times() + dt
        motion_len = env._motion_lib.get_motion_length(env._motion_ids)
        target_time = torch.minimum(target_time, motion_len)
        ref = env._motion_lib.calc_motion_frame(env._motion_ids, target_time)
        ref_root_pos, ref_root_rot, _, _, ref_joint_rot, ref_dof_vel = ref
        ref_dof_pos = env._motion_lib.joint_rot_to_dof(ref_joint_rot)
        ref_body_pos, _ = env._kin_char_model.forward_kinematics(
            ref_root_pos, ref_root_rot, ref_joint_rot
        )

        action = torch.minimum(
            torch.maximum(ref_dof_pos, env._action_bound_low),
            env._action_bound_high,
        )
        clip_fraction = torch.mean((action != ref_dof_pos).float())
        _, _, done, _ = env.step(action)

        dof_pos = env._engine.get_dof_pos(char_id)
        dof_vel = env._engine.get_dof_vel(char_id)
        root_pos = env._engine.get_root_pos(char_id)
        root_rot = env._engine.get_root_rot(char_id)
        body_pos = env._engine.get_body_pos(char_id)

        values = [dof_pos, dof_vel, root_pos, root_rot, body_pos]
        finite = all(torch.isfinite(value).all().item() for value in values)
        row = {
            "step": step + 1,
            "time": scalar(env.get_env_time()[0]),
            "dof_pos_rmse": scalar(torch.sqrt(torch.mean((dof_pos - ref_dof_pos) ** 2))),
            "dof_vel_rmse": scalar(torch.sqrt(torch.mean((dof_vel - ref_dof_vel) ** 2))),
            "root_pos_error": scalar(torch.linalg.vector_norm(root_pos - ref_root_pos, dim=-1)[0]),
            "root_rot_error": scalar(torch.abs(torch_util.quat_diff_angle(root_rot, ref_root_rot))[0]),
            "body_pos_rmse": scalar(torch.sqrt(torch.mean((body_pos - ref_body_pos) ** 2))),
            "clip_fraction": scalar(clip_fraction),
            "finite": bool(finite),
            "done": int(done[0].item()),
        }
        rows.append(row)

        if not finite:
            done_reason = "non_finite"
            break
        if row["done"] != base_env.DoneFlags.NULL.value:
            done_reason = base_env.DoneFlags(row["done"]).name.lower()
            contact_forces = env._engine.get_ground_contact_forces(char_id)[0]
            body_names = env._engine.get_obj_body_names(char_id)
            force_norm = torch.linalg.vector_norm(contact_forces, dim=-1)
            termination_contacts = [
                {"body": body_names[i], "force": float(force_norm[i].cpu().item())}
                for i in torch.nonzero(force_norm > 1.0, as_tuple=False).flatten().tolist()
            ]
            break

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_file = output_dir / "steps.csv"
    with csv_file.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "seed": args.seed,
        "zero_reset_root_velocity": args.zero_reset_root_velocity,
        "root_height_offset": args.root_height_offset,
        "steps": len(rows),
        "duration": rows[-1]["time"],
        "horizon": args.horizon,
        "survival_fraction": min(rows[-1]["time"] / args.horizon, 1.0),
        "done_reason": done_reason,
        "termination_contacts": termination_contacts,
        "all_finite": all(row["finite"] for row in rows),
        "mean_dof_pos_rmse": float(np.mean([row["dof_pos_rmse"] for row in rows])),
        "max_dof_pos_rmse": float(np.max([row["dof_pos_rmse"] for row in rows])),
        "mean_root_pos_error": float(np.mean([row["root_pos_error"] for row in rows])),
        "mean_body_pos_rmse": float(np.mean([row["body_pos_rmse"] for row in rows])),
        "max_clip_fraction": float(np.max([row["clip_fraction"] for row in rows])),
    }

    if args.video:
        video_file = output_dir / "tracking.mp4"
        env.record_diagnostics()["sim_recording"].save(str(video_file))
        summary["video"] = str(video_file)

    with (output_dir / "summary.json").open("w") as stream:
        json.dump(summary, stream, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
