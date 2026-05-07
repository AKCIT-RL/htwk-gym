#!/usr/bin/env python3
"""DribbleMaster Environment Diagnostic
Runs the full training pipeline for ~50 iterations with 64 envs,
printing detailed metrics every iteration to verify the environment
is healthy: rewards, episode lengths, NaN checks, observation stats, etc.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Isaac Gym MUST be imported before torch
try:
    import isaacgym
except ImportError:
    print("ERROR: isaacgym not available. Run inside Docker.")
    sys.exit(1)

import torch
import torch.nn.functional as F
import yaml
import numpy as np
import math
import time


def load_cfg(stage=1):
    """Load the YAML config for the given stage."""
    name = f"Dribble_Master_Stage{stage}"
    cfg_path = os.path.join(os.path.dirname(__file__), "..", "envs", "T1", f"{name}.yaml")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    # Override for diagnostic: small env count, headless, GPU
    cfg["env"]["num_envs"] = 64
    cfg["basic"]["headless"] = True
    cfg["basic"]["sim_device"] = "cuda:0"
    cfg["basic"]["rl_device"] = "cuda:0"
    cfg["basic"]["seed"] = 42
    cfg["basic"]["task"] = f"T1/{name}"
    cfg["runner"]["use_wandb"] = False
    cfg["viewer"]["record_video"] = False  # no rendering in Docker
    return cfg


def check_tensor(name, t, step=None):
    """Check a tensor for NaN, Inf, and extreme values. Returns issues list."""
    issues = []
    prefix = f"[step {step}] " if step is not None else ""
    if torch.isnan(t).any():
        count = torch.isnan(t).sum().item()
        issues.append(f"{prefix}{name}: {count} NaN values!")
    if torch.isinf(t).any():
        count = torch.isinf(t).sum().item()
        issues.append(f"{prefix}{name}: {count} Inf values!")
    if t.numel() > 0:
        amax = t.abs().max().item()
        if amax > 1e6:
            issues.append(f"{prefix}{name}: extreme abs max = {amax:.2e}")
    return issues


def print_section(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def run_diagnostic(stage=1, num_iters=50, horizon=24, mini_epochs=5):
    from envs.T1.dribble_master import DribbleMaster
    from utils.model import ActorCritic
    from utils.buffer import ExperienceBuffer
    from utils.utils import discount_values, surrogate_loss

    print_section(f"DribbleMaster Stage {stage} Diagnostic")
    print(f"  Envs: 64 | Iters: {num_iters} | Horizon: {horizon} | Mini-epochs: {mini_epochs}")

    cfg = load_cfg(stage)
    device = cfg["basic"]["rl_device"]

    # Set seed
    torch.manual_seed(cfg["basic"]["seed"])
    np.random.seed(cfg["basic"]["seed"])

    # --- 1. Environment creation ---
    print_section("1. Environment Creation")
    t0 = time.time()
    env = DribbleMaster(cfg)
    t_create = time.time() - t0
    print(f"  Created in {t_create:.2f}s")
    print(f"  num_obs: {env.num_obs}, num_privileged_obs: {env.num_privileged_obs}, num_actions: {env.num_actions}")
    print(f"  DOF names ({len(env.dof_names)}): {env.dof_names}")
    print(f"  dt: {env.dt:.4f}s, decimation: {cfg['control']['decimation']}")
    print(f"  Ball init distance range: {cfg['ball']['init_distance_range']}")
    print(f"  Virtual camera FOV: {cfg['virtual_camera']['hfov_deg']}° H × {cfg['virtual_camera']['vfov_deg']}° V")

    # --- 2. Reset ---
    print_section("2. Initial Reset")
    obs, extras = env.reset()
    priv_obs = extras["privileged_obs"]
    print(f"  obs shape: {obs.shape}, priv_obs shape: {priv_obs.shape}")
    issues = check_tensor("obs", obs) + check_tensor("priv_obs", priv_obs)
    if issues:
        for iss in issues:
            print(f"  ⚠ {iss}")
    else:
        print("  ✓ No NaN/Inf in initial observations")

    # Obs stats
    print(f"  obs — min: {obs.min():.4f}, max: {obs.max():.4f}, mean: {obs.mean():.4f}, std: {obs.std():.4f}")
    print(f"  priv — min: {priv_obs.min():.4f}, max: {priv_obs.max():.4f}, mean: {priv_obs.mean():.4f}")

    # Ball distance check
    ball_dist = torch.norm(env.ball_pos[:, :2] - env.base_pos[:, :2], dim=-1)
    print(f"  Ball distance — min: {ball_dist.min():.2f}, max: {ball_dist.max():.2f}, mean: {ball_dist.mean():.2f}")
    print(f"  Ball in FOV: {env.ball_in_fov.mean():.2%}")
    print(f"  Robot height: {env.base_pos[:, 2].mean():.3f}m")

    # --- 3. Model creation ---
    print_section("3. Model")
    actor_hidden = cfg.get("model", {}).get("actor_hidden_dims", None)
    critic_hidden = cfg.get("model", {}).get("critic_hidden_dims", None)
    model = ActorCritic(
        env.num_actions, env.num_obs, env.num_privileged_obs,
        actor_hidden_dims=actor_hidden, critic_hidden_dims=critic_hidden,
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Actor: {actor_hidden}, Critic: {critic_hidden}")
    print(f"  Total params: {total_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["algorithm"]["learning_rate"])

    buffer = ExperienceBuffer(horizon, env.num_envs, device)
    buffer.add_buffer("actions", (env.num_actions,))
    buffer.add_buffer("obses", (env.num_obs,))
    buffer.add_buffer("privileged_obses", (env.num_privileged_obs,))
    buffer.add_buffer("rewards", ())
    buffer.add_buffer("dones", (), dtype=bool)
    buffer.add_buffer("time_outs", (), dtype=bool)

    # --- 4. Training loop ---
    print_section("4. Training Loop")
    header = (
        f"{'Iter':>4} | {'Reward':>9} | {'EpLen':>6} | {'ValLoss':>9} | "
        f"{'ActLoss':>9} | {'KL':>8} | {'LR':>9} | {'FOV%':>5} | "
        f"{'BallDist':>8} | {'Height':>6} | {'Resets':>6} | {'Issues':>6}"
    )
    print(header)
    print("-" * len(header))

    learning_rate = cfg["algorithm"]["learning_rate"]
    all_issues = []
    reward_history = []
    ep_len_history = []

    for it in range(num_iters):
        # -- Collect rollouts --
        step_rewards = []
        step_dones = []
        iter_issues = []

        for n in range(horizon):
            buffer.update_data("obses", n, obs)
            buffer.update_data("privileged_obses", n, priv_obs)
            with torch.no_grad():
                dist = model.act(obs)
                act = dist.sample()
            obs, rew, done, infos = env.step(act)
            priv_obs = infos["privileged_obs"]
            buffer.update_data("actions", n, act)
            buffer.update_data("rewards", n, rew)
            buffer.update_data("dones", n, done)
            buffer.update_data("time_outs", n, infos["time_outs"])

            step_rewards.append(rew.mean().item())
            step_dones.append(done.float().mean().item())

            # Check for issues
            iter_issues += check_tensor("obs", obs, n)
            iter_issues += check_tensor("rew", rew, n)
            if torch.isnan(rew).any() or torch.isinf(rew).any():
                # Detailed reward breakdown
                print(f"\n  ⚠ BAD REWARD at iter {it}, step {n}!")
                for rname, rval in infos.get("rew_terms", {}).items():
                    if torch.isnan(rval).any() or torch.isinf(rval).any():
                        print(f"    → {rname}: has NaN/Inf!")

        # -- PPO update --
        with torch.no_grad():
            old_dist = model.act(buffer["obses"])
            old_log_prob = old_dist.log_prob(buffer["actions"]).sum(dim=-1)

        mean_vloss = 0
        mean_aloss = 0
        kl_mean = torch.tensor(0.0)

        for _ in range(mini_epochs):
            values = model.est_value(buffer["obses"], buffer["privileged_obses"])
            last_values = model.est_value(obs, priv_obs)
            with torch.no_grad():
                buffer["rewards"][buffer["time_outs"]] = values[buffer["time_outs"]]
                advantages = discount_values(
                    buffer["rewards"], buffer["dones"] | buffer["time_outs"],
                    values, last_values,
                    cfg["algorithm"]["gamma"], cfg["algorithm"]["lam"],
                )
                returns = values + advantages
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            value_loss = F.mse_loss(values, returns)
            dist = model.act(buffer["obses"])
            log_prob = dist.log_prob(buffer["actions"]).sum(dim=-1)
            actor_loss = surrogate_loss(old_log_prob, log_prob, advantages)
            bound_loss = (
                torch.clip(dist.loc - 1.0, min=0.0).square().mean()
                + torch.clip(dist.loc + 1.0, max=0.0).square().mean()
            )
            entropy = dist.entropy().sum(dim=-1)
            loss = (
                value_loss + actor_loss
                + cfg["algorithm"]["bound_coef"] * bound_loss
                + cfg["algorithm"]["entropy_coef"] * entropy.mean()
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            with torch.no_grad():
                kl = torch.sum(
                    torch.log(dist.scale / old_dist.scale)
                    + 0.5 * (old_dist.scale.square() + (dist.loc - old_dist.loc).square()) / dist.scale.square()
                    - 0.5,
                    axis=-1,
                )
                kl_mean = kl.mean()
                if kl_mean > cfg["algorithm"]["desired_kl"] * 2:
                    learning_rate = max(1e-5, learning_rate / 1.5)
                elif kl_mean < cfg["algorithm"]["desired_kl"] / 2:
                    learning_rate = min(1e-2, learning_rate * 1.5)
                for pg in optimizer.param_groups:
                    pg["lr"] = learning_rate

            mean_vloss += value_loss.item()
            mean_aloss += actor_loss.item()

        mean_vloss /= mini_epochs
        mean_aloss /= mini_epochs

        # -- Metrics --
        mean_rew = np.mean(step_rewards)
        mean_reset_rate = np.mean(step_dones)
        ball_dist_now = torch.norm(env.ball_pos[:, :2] - env.base_pos[:, :2], dim=-1).mean().item()
        fov_pct = env.ball_in_fov.mean().item() * 100
        height = env.base_pos[:, 2].mean().item()
        ep_len_approx = 1.0 / max(mean_reset_rate, 1e-6)

        reward_history.append(mean_rew)
        ep_len_history.append(ep_len_approx)

        n_issues = len(iter_issues)
        all_issues += iter_issues

        print(
            f"{it+1:4d} | {mean_rew:9.4f} | {ep_len_approx:6.1f} | {mean_vloss:9.4f} | "
            f"{mean_aloss:9.4f} | {kl_mean.item():8.5f} | {learning_rate:9.2e} | {fov_pct:5.1f} | "
            f"{ball_dist_now:8.3f} | {height:6.3f} | {mean_reset_rate:6.3f} | {n_issues:6d}"
        )

        # Print first few issues if any
        if iter_issues:
            for iss in iter_issues[:3]:
                print(f"    ⚠ {iss}")
            if len(iter_issues) > 3:
                print(f"    ... and {len(iter_issues)-3} more issues")

    # --- 5. Final summary ---
    print_section("5. Summary")

    # Check for reward trends
    first_5 = np.mean(reward_history[:5])
    last_5 = np.mean(reward_history[-5:])
    print(f"  Mean reward (first 5 iters):  {first_5:.4f}")
    print(f"  Mean reward (last 5 iters):   {last_5:.4f}")
    print(f"  Reward trend:                 {'IMPROVING ✓' if last_5 > first_5 else 'DECLINING ✗' if last_5 < first_5 - 0.1 else 'STABLE ~'}")

    first_ep = np.mean(ep_len_history[:5])
    last_ep = np.mean(ep_len_history[-5:])
    print(f"  Mean episode length (first):  {first_ep:.1f} steps")
    print(f"  Mean episode length (last):   {last_ep:.1f} steps")

    print(f"  Total issues found:           {len(all_issues)}")
    if all_issues:
        unique_issues = list(set(all_issues))
        print(f"  Unique issue types:           {len(unique_issues)}")
        for iss in unique_issues[:10]:
            print(f"    - {iss}")

    # Final state check
    print(f"\n  Final robot height:           {env.base_pos[:, 2].mean():.3f}m")
    print(f"  Final ball distance:          {torch.norm(env.ball_pos[:, :2] - env.base_pos[:, :2], dim=-1).mean():.2f}m")
    print(f"  Final ball in FOV:            {env.ball_in_fov.mean():.2%}")

    # Reward breakdown for last iteration
    print(f"\n  Reward breakdown (last iter):")
    for rname, rval in sorted(infos.get("rew_terms", {}).items()):
        val = rval.mean().item()
        print(f"    {rname:30s}: {val:10.6f}")

    # Health check
    print_section("6. Health Verdict")
    healthy = True
    verdicts = []

    if len(all_issues) > 0:
        verdicts.append(("WARN", f"{len(all_issues)} numerical issues (NaN/Inf/extreme)"))
        if any("NaN" in i for i in all_issues):
            healthy = False
            verdicts.append(("FAIL", "NaN values detected — environment has a bug"))

    if last_ep < 3:
        healthy = False
        verdicts.append(("FAIL", f"Episodes too short ({last_ep:.1f} steps) — robot falls immediately"))
    elif last_ep < 10:
        verdicts.append(("WARN", f"Episodes quite short ({last_ep:.1f} steps) — robot unstable"))
    else:
        verdicts.append(("OK", f"Episode length {last_ep:.1f} steps — robot surviving"))

    if env.base_pos[:, 2].mean() < 0.3:
        healthy = False
        verdicts.append(("FAIL", "Robot height < 0.3m — likely collapsed"))
    else:
        verdicts.append(("OK", f"Robot height {env.base_pos[:, 2].mean():.3f}m — standing"))

    if last_5 > first_5:
        verdicts.append(("OK", "Reward improving over training"))
    elif last_5 < first_5 - 0.5:
        verdicts.append(("WARN", "Reward declining significantly"))
    else:
        verdicts.append(("OK", "Reward stable (normal for few iterations)"))

    for status, msg in verdicts:
        icon = {"OK": "✓", "WARN": "⚠", "FAIL": "✗"}[status]
        print(f"  [{icon}] {msg}")

    print(f"\n  Overall: {'HEALTHY ✓' if healthy else 'ISSUES DETECTED ✗'}")
    return healthy


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, default=1, choices=[1, 2])
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--horizon", type=int, default=24)
    args = parser.parse_args()

    healthy = run_diagnostic(stage=args.stage, num_iters=args.iters, horizon=args.horizon)
    sys.exit(0 if healthy else 1)
