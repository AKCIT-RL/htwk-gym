"""Figure 3A-style spatial evaluation for the frozen soccer policy."""

import argparse
import csv
import json
import os
import sys

MIMICKIT_ROOT = "/workspace/MimicKit"
HTWK_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HTWK_ROOT)
sys.path.insert(0, os.path.join(MIMICKIT_ROOT, "mimickit"))
os.chdir(MIMICKIT_ROOT)

import isaacgym  # noqa: F401
import numpy as np
import torch

import envs.base_env as base_env
import envs.env_builder as env_builder
import learning.agent_builder as agent_builder
import learning.base_agent as base_agent_module
import util.mp_util as mp_util
import util.util as util
from utils.soccer_behavior_metrics import approach_angle_improvement
from utils.soccer_spatial_benchmark import (
    OUTCOMES,
    allocate_trials,
    make_field_grid,
    summarize_trials,
)

KICK_SPEED_GAIN = 0.5


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_file", required=True)
    parser.add_argument("--env_config", default="data/envs/mcwamp_g1_soccer_env_s8.yaml")
    parser.add_argument("--agent_config", default="data/agents/mcwamp_g1_soccer_agent.yaml")
    parser.add_argument("--engine_config", default="data/engines/isaac_gym_engine.yaml")
    parser.add_argument("--num_envs", type=int, default=32)
    parser.add_argument("--trials", type=int, default=8192)
    parser.add_argument("--rand_seed", type=int, default=1)
    parser.add_argument("--cell_size", type=float, default=1.0)
    parser.add_argument("--zero_steering", action="store_true")
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def angle_deg(first, second):
    first = first / torch.clamp_min(torch.linalg.norm(first, dim=-1, keepdim=True), 1.0e-6)
    second = second / torch.clamp_min(torch.linalg.norm(second, dim=-1, keepdim=True), 1.0e-6)
    cosine = torch.clamp(torch.sum(first * second, dim=-1), -1.0, 1.0)
    return torch.rad2deg(torch.acos(cosine))


def build(args, device):
    overrides = {
        "rand_reset": False,
        "rand_ball_props": False,
        "rand_char_props": False,
        "char_push_enable": False,
        "ball_perturb_time_min": 1.0e9,
        "ball_perturb_time_max": 1.0e9,
        "visualize_field": False,
        "visualize_debug_arrows": False,
    }
    if args.zero_steering:
        overrides.update({
            "steer_anneal_start_samples": 0.0,
            "steer_anneal_end_samples": 0.0,
        })
    env = env_builder.build_env(
        env_file=args.env_config,
        engine_file=args.engine_config,
        num_envs=args.num_envs,
        device=device,
        visualize=False,
        env_overrides=overrides,
    )
    agent = agent_builder.build_agent(args.agent_config, env, device)
    agent.load(args.model_file)
    agent.eval()
    agent.set_mode(base_agent_module.AgentMode.TEST)
    env.set_mode(base_env.EnvMode.TEST)
    return env, agent


def write_artifacts(args, grid, trials, summary):
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    metadata = {
        "protocol": "Figure 3A-style local adaptation",
        "model_file": args.model_file,
        "env_config": args.env_config,
        "agent_config": args.agent_config,
        "engine_config": args.engine_config,
        "rand_seed": args.rand_seed,
        "num_envs": args.num_envs,
        "requested_trials": args.trials,
        "cell_size_m": args.cell_size,
        "zero_steering": args.zero_steering,
        "robot_spawn": "field center, fixed default pose and heading",
        "terrain": "flat",
        "domain_randomization": False,
        "ball_perturbations": False,
        "robot_pushes": False,
        "virtual_perception": True,
    }
    payload = {"metadata": metadata, "summary": summary, "trials": trials}
    with open(os.path.join(output_dir, "results.json"), "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, allow_nan=False)

    trial_fields = list(trials[0].keys())
    with open(os.path.join(output_dir, "trials.csv"), "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=trial_fields)
        writer.writeheader()
        writer.writerows(trials)

    cell_fields = [
        "cell_id", "length_index", "width_index", "ball_x_m", "ball_y_m", "trials",
        "success", "oob", "fall", "timeout", "success_rate", "success_ci95_low",
        "success_ci95_high",
    ]
    with open(os.path.join(output_dir, "cells.csv"), "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=cell_fields)
        writer.writeheader()
        for cell in summary["cells"]:
            writer.writerow({
                "cell_id": cell["cell_id"],
                "length_index": cell["length_index"],
                "width_index": cell["width_index"],
                "ball_x_m": cell["ball_x_m"],
                "ball_y_m": cell["ball_y_m"],
                "trials": cell["trials"],
                "success": cell["counts"]["success"],
                "oob": cell["counts"]["oob"],
                "fall": cell["counts"]["fall"],
                "timeout": cell["counts"]["timeout"],
                "success_rate": cell["rates"]["success"],
                "success_ci95_low": cell["success_ci95"][0],
                "success_ci95_high": cell["success_ci95"][1],
            })

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib unavailable; JSON and CSV artifacts were written")
        return

    num_x = int(grid[:, 0].max()) + 1
    num_y = int(grid[:, 1].max()) + 1
    for outcome in OUTCOMES:
        values = np.full((num_y, num_x), np.nan)
        for cell in summary["cells"]:
            values[cell["width_index"], cell["length_index"]] = cell["rates"][outcome]
        figure, axis = plt.subplots(figsize=(12, 6))
        image = axis.imshow(values, origin="lower", extent=(-7, 7, -4.5, 4.5),
                            vmin=0.0, vmax=1.0, cmap="viridis", aspect="equal")
        axis.set_title("{} rate ({} steering)".format(
            outcome.capitalize(), "zero" if args.zero_steering else "nominal"))
        axis.set_xlabel("Field x (m), goal at +7 m")
        axis.set_ylabel("Field y (m)")
        figure.colorbar(image, ax=axis, label="Rate")
        figure.tight_layout()
        figure.savefig(os.path.join(output_dir, "{}_heatmap.png".format(outcome)), dpi=160)
        plt.close(figure)


def main():
    args = parse_args()
    if args.num_envs <= 0 or args.trials <= 0:
        raise ValueError("num_envs and trials must be positive")

    device = "cuda:0"
    mp_util.init(0, 1, device, int(np.random.randint(6000, 7000)))
    util.set_rand_seed(args.rand_seed)
    env, agent = build(args, device)

    if not env._virtual_perception:
        raise RuntimeError("the benchmark requires virtual perception")
    if env._engine._ground_config is not None:
        raise RuntimeError("the benchmark requires the flat engine")
    if env._rand_ball_props or env._rand_char_props or env._char_push_enable:
        raise RuntimeError("benchmark domain randomization overrides were not applied")

    grid = make_field_grid(env._field_length, env._field_width, args.cell_size)
    schedule = allocate_trials(len(grid), args.trials)
    num_envs = args.num_envs
    env_ids_all = torch.arange(num_envs, device=device, dtype=torch.long)
    ball_id = env._get_ball_id()
    char_id = env._get_char_id()

    active = torch.zeros(num_envs, device=device, dtype=torch.bool)
    trial_index = torch.full((num_envs,), -1, device=device, dtype=torch.long)
    cell_id = torch.full((num_envs,), -1, device=device, dtype=torch.long)
    initial_approach = torch.full((num_envs,), float("nan"), device=device)
    first_touch_s = torch.full((num_envs,), float("nan"), device=device)
    first_kick_s = torch.full((num_envs,), float("nan"), device=device)
    first_kick_error = torch.full((num_envs,), float("nan"), device=device)
    first_kick_foot = torch.full((num_envs,), -1, device=device, dtype=torch.long)
    approach_improvement = torch.full((num_envs,), float("nan"), device=device)
    kick_count = torch.zeros(num_envs, device=device, dtype=torch.long)
    last_kick_distance = torch.full((num_envs,), float("nan"), device=device)
    prev_contact = torch.zeros(num_envs, device=device, dtype=torch.bool)
    prev_ball_speed = torch.zeros(num_envs, device=device)
    completed = []
    next_schedule = 0
    next_progress = 256

    def assign(ids):
        nonlocal next_schedule
        count = min(len(ids), args.trials - next_schedule)
        if count <= 0:
            active[ids] = False
            return
        ids = ids[:count]
        indices = torch.arange(next_schedule, next_schedule + count, device=device)
        cells = torch.as_tensor(schedule[next_schedule:next_schedule + count],
                                device=device, dtype=torch.long)
        positions = torch.as_tensor(grid[cells.cpu().numpy(), 2:4], device=device,
                                    dtype=torch.float)
        env.reset_to_spatial_benchmark(ids, positions)
        actual_root = env._engine.get_root_pos(char_id)[ids, 0:2] - env._field_offset[ids]
        actual_ball = env._get_ball_pos()[ids, 0:2] - env._field_offset[ids]
        if not torch.allclose(actual_root, torch.zeros_like(actual_root), atol=1.0e-5):
            raise RuntimeError("controlled reset did not place every robot at field center")
        if not torch.allclose(actual_ball, positions, atol=1.0e-5):
            raise RuntimeError("controlled reset mixed or misplaced benchmark ball cells")
        active[ids] = True
        trial_index[ids] = indices
        cell_id[ids] = cells
        first_touch_s[ids] = float("nan")
        first_kick_s[ids] = float("nan")
        first_kick_error[ids] = float("nan")
        first_kick_foot[ids] = -1
        approach_improvement[ids] = float("nan")
        kick_count[ids] = 0
        last_kick_distance[ids] = float("nan")
        prev_contact[ids] = False
        prev_ball_speed[ids] = 0.0
        root = env._engine.get_root_pos(char_id)[ids, 0:2]
        ball = env._get_ball_pos()[ids, 0:2]
        to_ball = ball - root
        to_goal = env._goal_pos[ids] - ball
        initial_approach[ids] = angle_deg(to_ball, to_goal)
        next_schedule += count

    assign(env_ids_all)
    obs, info = env._obs_buf, env._info

    with torch.no_grad():
        while len(completed) < args.trials:
            actions, _ = agent._decide_action(obs, info)
            actions[~active] = 0.0
            obs, _, done, info = env.step(actions)

            ball_pos = env._get_ball_pos()
            ball_vel = env._engine.get_root_vel(ball_id)[:, 0:2]
            ball_speed = torch.linalg.norm(ball_vel, dim=-1)
            root_pos = env._engine.get_root_pos(char_id)
            body_pos = env._engine.get_body_pos(char_id)
            foot_distances = torch.stack([
                torch.linalg.norm(body_pos[:, foot_id, :] - ball_pos, dim=-1)
                for foot_id in env._foot_body_ids
            ], dim=-1)
            contact = torch.any(foot_distances < env._ball_contact_dist, dim=-1) & active
            new_touch = contact & ~prev_contact
            touch_ids = new_touch & torch.isnan(first_touch_s)
            first_touch_s[touch_ids] = env._time_buf[touch_ids]

            kick = contact & ((ball_speed - prev_ball_speed) > KICK_SPEED_GAIN)
            kick_ids = kick.nonzero(as_tuple=False).flatten()
            kick_count[kick_ids] += 1
            if len(kick_ids) > 0:
                to_goal = env._goal_pos[kick_ids] - ball_pos[kick_ids, 0:2]
                error = angle_deg(ball_vel[kick_ids], to_goal)
                first = torch.isnan(first_kick_s[kick_ids])
                first_ids = kick_ids[first]
                first_kick_s[first_ids] = env._time_buf[first_ids]
                first_kick_error[first_ids] = error[first]
                first_kick_foot[first_ids] = foot_distances[first_ids].argmin(dim=-1)
                current_approach = angle_deg(
                    ball_pos[first_ids, 0:2] - root_pos[first_ids, 0:2],
                    env._goal_pos[first_ids] - ball_pos[first_ids, 0:2],
                )
                for env_id, current in zip(first_ids.tolist(), current_approach.tolist()):
                    approach_improvement[env_id] = approach_angle_improvement(
                        initial_approach[env_id].item(), current)
                last_kick_distance[kick_ids] = torch.linalg.norm(to_goal, dim=-1)

            done_ids = ((done != base_env.DoneFlags.NULL.value) & active).nonzero(
                as_tuple=False).flatten()
            if len(done_ids) > 0:
                for env_id in done_ids.tolist():
                    if env._goal_scored_buf[env_id]:
                        outcome = "success"
                    elif env._ball_oob_buf[env_id]:
                        outcome = "oob"
                    elif done[env_id].item() == base_env.DoneFlags.TIME.value:
                        outcome = "timeout"
                    else:
                        outcome = "fall"
                    grid_row = grid[cell_id[env_id].item()]
                    completed.append({
                        "trial_index": int(trial_index[env_id].item()),
                        "cell_id": int(cell_id[env_id].item()),
                        "ball_x_m": float(grid_row[2]),
                        "ball_y_m": float(grid_row[3]),
                        "outcome": outcome,
                        "episode_length_s": float(env._time_buf[env_id].item()),
                        "first_touch_s": float(first_touch_s[env_id].item()),
                        "first_kick_s": float(first_kick_s[env_id].item()),
                        "kick_count": int(kick_count[env_id].item()),
                        "first_kick_error_deg": float(first_kick_error[env_id].item()),
                        "first_kick_foot": int(first_kick_foot[env_id].item()),
                        "approach_improvement_deg": float(approach_improvement[env_id].item()),
                        "scoring_kick_distance_m": float(last_kick_distance[env_id].item())
                        if outcome == "success" else float("nan"),
                    })
                active[done_ids] = False
                assign(done_ids)
                obs, info = env._obs_buf, env._info

            prev_contact = contact
            prev_ball_speed = ball_speed
            if len(completed) >= next_progress:
                print("completed {}/{} trials".format(len(completed), args.trials))
                next_progress += 256

    completed.sort(key=lambda trial: trial["trial_index"])
    for trial in completed:
        for key, value in list(trial.items()):
            if isinstance(value, float) and not np.isfinite(value):
                trial[key] = None
    summary = summarize_trials(grid, completed)
    write_artifacts(args, grid, completed, summary)

    print("\n================ SPATIAL SOCCER BENCHMARK ================")
    print("trials: {}  cells: {}  zero steering: {}".format(
        summary["total_trials"], len(grid), args.zero_steering))
    for outcome in OUTCOMES:
        print("{:8s}: {:5d} ({:.1%})".format(
            outcome, summary["counts"][outcome], summary["rates"][outcome]))
    print("artifacts: {}".format(os.path.abspath(args.output_dir)))
    print("==========================================================")


if __name__ == "__main__":
    main()
