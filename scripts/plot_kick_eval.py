"""Pretty plots for an evaluate_kick.py all_scenarios_summary.json.

Usage:
    python3 scripts/plot_kick_eval.py <eval_dir>

`<eval_dir>` must contain `all_scenarios_summary.json`. Output PNGs are written
to `<eval_dir>/plots/`.
"""
import json
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _label_value(label, key):
    """Extract first float matching key=NUMBER from a label like 'dx=+0.20,dy=-0.15'."""
    m = re.search(rf"{key}\s*=\s*([+-]?\d*\.?\d+)", label)
    return float(m.group(1)) if m else None


def _per_cond_field(conds, path):
    """conds: list[dict]; path: tuple of keys to dive into."""
    out = []
    for c in conds:
        v = c
        ok = True
        for k in path:
            if isinstance(v, dict) and k in v:
                v = v[k]
            else:
                ok = False
                break
        out.append(v if ok else np.nan)
    return np.array(out, dtype=float)


def plot_scenario(scenario_name, scen, out_dir):
    """Generic bar chart: hit_rate + angular error per condition."""
    conds = scen["per_condition"]
    labels = [c["condition_label"] for c in conds]
    hit = np.array([c["hit_rate"] for c in conds]) * 100
    kick = np.array([c["n_kicks_detected"] / max(c["n_attempts"], 1) for c in conds]) * 100
    fall = np.array([(c["n_fell_before_kick"] + c["n_fell_after_kick"]) / max(c["n_attempts"], 1) for c in conds]) * 100
    ang = _per_cond_field(conds, ("angular_error_deg", "mean"))
    ang_std = _per_cond_field(conds, ("angular_error_deg", "std"))
    speed = _per_cond_field(conds, ("ball_speed_at_kick_mps", "mean"))

    fig, axs = plt.subplots(1, 3, figsize=(15, 4.2))
    x = np.arange(len(labels))

    # Panel 1: success / kick / fall percentages
    w = 0.28
    axs[0].bar(x - w, hit, width=w, label="hit %", color="#2a9d8f")
    axs[0].bar(x, kick, width=w, label="kick %", color="#577590")
    axs[0].bar(x + w, fall, width=w, label="fall %", color="#e76f51")
    axs[0].set_title(f"{scenario_name}: outcome rates")
    axs[0].set_ylabel("percent")
    axs[0].set_ylim(0, 105)
    axs[0].set_xticks(x)
    axs[0].set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    axs[0].axhline(50, color="gray", linewidth=0.5, linestyle=":")
    axs[0].legend(fontsize=8)
    for xi, hi in zip(x, hit):
        axs[0].text(xi - w, hi + 2, f"{hi:.0f}", ha="center", fontsize=7)

    # Panel 2: angular error
    axs[1].bar(x, ang, yerr=ang_std, color="#f4a261", capsize=3)
    axs[1].axhline(20, color="red", linewidth=0.8, linestyle="--",
                   label="hit threshold (20°)")
    axs[1].set_title(f"{scenario_name}: angular error at kick")
    axs[1].set_ylabel("deg")
    axs[1].set_xticks(x)
    axs[1].set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    axs[1].legend(fontsize=8)

    # Panel 3: ball speed at kick
    axs[2].bar(x, speed, color="#264653")
    axs[2].set_title(f"{scenario_name}: ball speed at kick")
    axs[2].set_ylabel("m/s")
    axs[2].set_xticks(x)
    axs[2].set_xticklabels(labels, rotation=35, ha="right", fontsize=8)

    fig.suptitle(f"Scenario: {scenario_name}  (n_attempts={scen['overall']['n_attempts']})",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path = os.path.join(out_dir, f"{scenario_name}.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_ball_pos_heatmap(scen, out_dir):
    """Heatmap of hit_rate over (dx, dy) grid for the `ball_pos` scenario."""
    conds = scen["per_condition"]
    dx = [_label_value(c["condition_label"], "dx") for c in conds]
    dy = [_label_value(c["condition_label"], "dy") for c in conds]
    if any(v is None for v in dx + dy):
        return None
    dx_u = sorted(set(dx))
    dy_u = sorted(set(dy))
    hit = np.full((len(dy_u), len(dx_u)), np.nan)
    ang = np.full_like(hit, np.nan)
    for c, x, y in zip(conds, dx, dy):
        ix = dx_u.index(x)
        iy = dy_u.index(y)
        hit[iy, ix] = c["hit_rate"] * 100
        ang[iy, ix] = c["angular_error_deg"]["mean"]

    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    im0 = axs[0].imshow(hit, origin="lower", cmap="RdYlGn", vmin=0, vmax=100,
                        aspect="auto")
    axs[0].set_xticks(range(len(dx_u))); axs[0].set_xticklabels([f"{v:+.2f}" for v in dx_u])
    axs[0].set_yticks(range(len(dy_u))); axs[0].set_yticklabels([f"{v:+.2f}" for v in dy_u])
    axs[0].set_xlabel("dx (m, robot frame)")
    axs[0].set_ylabel("dy (m, robot frame)")
    axs[0].set_title("Hit rate (%)")
    for iy in range(hit.shape[0]):
        for ix in range(hit.shape[1]):
            axs[0].text(ix, iy, f"{hit[iy, ix]:.0f}", ha="center", va="center",
                        fontsize=8, color="black")
    fig.colorbar(im0, ax=axs[0], shrink=0.85)

    im1 = axs[1].imshow(ang, origin="lower", cmap="RdYlGn_r", vmin=0, vmax=40,
                        aspect="auto")
    axs[1].set_xticks(range(len(dx_u))); axs[1].set_xticklabels([f"{v:+.2f}" for v in dx_u])
    axs[1].set_yticks(range(len(dy_u))); axs[1].set_yticklabels([f"{v:+.2f}" for v in dy_u])
    axs[1].set_xlabel("dx (m, robot frame)")
    axs[1].set_ylabel("dy (m, robot frame)")
    axs[1].set_title("Angular error (deg)")
    for iy in range(ang.shape[0]):
        for ix in range(ang.shape[1]):
            axs[1].text(ix, iy, f"{ang[iy, ix]:.0f}", ha="center", va="center",
                        fontsize=8, color="black")
    fig.colorbar(im1, ax=axs[1], shrink=0.85)

    fig.suptitle("ball_pos: initial ball offset (robot frame)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path = os.path.join(out_dir, "ball_pos_heatmap.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_overview(summary, out_dir):
    """One-page overview: hit% per scenario + total ang err per scenario."""
    scen = summary["scenarios"]
    names = list(scen.keys())
    hit = []
    ang = []
    fall = []
    for n in names:
        o = scen[n]["overall"]
        hit.append(100 * o["n_hits"] / max(o["n_attempts"], 1))
        fall.append(100 * (o["n_fell_before_kick"] + o["n_fell_after_kick"]) / max(o["n_attempts"], 1))
        ang.append(o["angular_error_deg"]["mean"])

    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    x = np.arange(len(names))

    bars = axs[0].bar(x, hit, color="#2a9d8f")
    axs[0].bar(x, fall, color="#e76f51", alpha=0.55, label="fall %")
    axs[0].set_xticks(x); axs[0].set_xticklabels(names, rotation=20, ha="right")
    axs[0].set_ylabel("%"); axs[0].set_ylim(0, 105)
    axs[0].set_title("Hit rate (green) and fall rate (red) per scenario")
    for xi, hi in zip(x, hit):
        axs[0].text(xi, hi + 2, f"{hi:.0f}%", ha="center", fontsize=9, fontweight="bold")

    axs[1].bar(x, ang, color="#f4a261")
    axs[1].axhline(20, color="red", linewidth=0.8, linestyle="--", label="hit threshold (20°)")
    axs[1].set_xticks(x); axs[1].set_xticklabels(names, rotation=20, ha="right")
    axs[1].set_ylabel("deg")
    axs[1].set_title("Mean angular error per scenario")
    axs[1].legend(fontsize=8)

    ckpt = summary.get("checkpoint", "?")
    fig.suptitle(f"Kicking Bikinha eval — overview\n{ckpt}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    path = os.path.join(out_dir, "0_overview.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def main():
    if len(sys.argv) < 2:
        print("usage: plot_kick_eval.py <eval_dir>")
        sys.exit(1)
    eval_dir = sys.argv[1]
    summary_path = os.path.join(eval_dir, "all_scenarios_summary.json")
    with open(summary_path) as f:
        summary = json.load(f)
    out_dir = os.path.join(eval_dir, "plots")
    os.makedirs(out_dir, exist_ok=True)

    written = [plot_overview(summary, out_dir)]
    for name, scen in summary["scenarios"].items():
        written.append(plot_scenario(name, scen, out_dir))
        if name == "ball_pos":
            p = plot_ball_pos_heatmap(scen, out_dir)
            if p:
                written.append(p)
    for p in written:
        print("wrote", p)


if __name__ == "__main__":
    main()
