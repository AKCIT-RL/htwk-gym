"""Render a paper-style soccer field figure from a benchmark result JSON."""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np


FIELD_LENGTH = 14.0
FIELD_WIDTH = 9.0
GOAL_WIDTH = 2.6


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--title", default="G1 spatial scoring benchmark")
    return parser.parse_args()


def finite_values(trials, key):
    return [trial[key] for trial in trials if trial.get(key) is not None]


def draw_field(axis):
    line = "#f7faf6"
    axis.add_patch(patches.Rectangle(
        (-7.0, -4.5), FIELD_LENGTH, FIELD_WIDTH,
        fill=False, edgecolor=line, linewidth=2.2, zorder=5,
    ))
    axis.plot([0.0, 0.0], [-4.5, 4.5], color=line, linewidth=1.5, zorder=5)
    axis.add_patch(patches.Circle(
        (0.0, 0.0), 0.75, fill=False, edgecolor=line, linewidth=1.5, zorder=5,
    ))
    axis.scatter([0.0], [0.0], s=18, color=line, zorder=6)
    axis.add_patch(patches.Rectangle(
        (7.0, -GOAL_WIDTH / 2.0), 0.35, GOAL_WIDTH,
        facecolor="none", edgecolor="#dce8e1", linewidth=2.2, zorder=6,
    ))
    axis.text(7.48, 0.0, "GOAL", rotation=90, ha="center", va="center",
              color="#263c34", fontsize=9, fontweight="bold")

    axis.scatter([0.0], [0.0], marker="o", s=125, facecolor="#f7faf6",
                 edgecolor="#16251f", linewidth=1.5, zorder=8)
    axis.annotate("", xy=(0.65, 0.0), xytext=(0.15, 0.0),
                  arrowprops={"arrowstyle": "-|>", "color": "#16251f", "lw": 2},
                  zorder=9)
    axis.text(0.0, -0.42, "fixed robot start", ha="center", va="top",
              color="#f7faf6", fontsize=8, fontweight="bold", zorder=9)


def render(payload, output, title):
    summary = payload["summary"]
    trials = payload["trials"]
    metadata = payload["metadata"]
    cells = summary["cells"]

    rates = np.full((9, 14), np.nan)
    for cell in cells:
        rates[cell["width_index"], cell["length_index"]] = cell["rates"]["success"]

    figure = plt.figure(figsize=(16, 8.8), facecolor="#eef2ed")
    grid = figure.add_gridspec(1, 2, width_ratios=[3.25, 1.0], wspace=0.08)
    field = figure.add_subplot(grid[0, 0])
    info = figure.add_subplot(grid[0, 1])
    field.set_facecolor("#244f3c")

    image = field.imshow(
        rates, origin="lower", extent=(-7.0, 7.0, -4.5, 4.5),
        vmin=0.0, vmax=1.0, cmap="RdYlGn", alpha=0.88,
        interpolation="nearest", aspect="equal", zorder=1,
    )
    for x in np.arange(-7.0, 7.01, 1.0):
        field.plot([x, x], [-4.5, 4.5], color="white", alpha=0.16,
                   linewidth=0.55, zorder=3)
    for y in np.arange(-4.5, 4.51, 1.0):
        field.plot([-7.0, 7.0], [y, y], color="white", alpha=0.16,
                   linewidth=0.55, zorder=3)

    for cell in cells:
        rate = cell["rates"]["success"]
        text_color = "#102018" if 0.28 <= rate <= 0.82 else "white"
        field.text(
            cell["ball_x_m"], cell["ball_y_m"], "{:.0f}".format(100.0 * rate),
            ha="center", va="center", color=text_color, fontsize=7.0,
            fontweight="bold", zorder=4,
        )

    draw_field(field)
    field.set_xlim(-7.25, 7.75)
    field.set_ylim(-4.75, 4.75)
    field.set_xlabel("Field x (m)  |  attacking direction ->", fontsize=11)
    field.set_ylabel("Field y (m)", fontsize=11)
    field.set_xticks(np.arange(-6, 7, 2))
    field.set_yticks(np.arange(-4, 5, 1))
    field.set_title(title, loc="left", fontsize=18, fontweight="bold", pad=18)
    field.text(-7.0, 4.82, "Cell labels: goal rate (%)  |  65-66 trials per cell",
               ha="left", va="bottom", color="#334b40", fontsize=10)

    colorbar = figure.colorbar(image, ax=field, orientation="horizontal",
                              fraction=0.055, pad=0.09, aspect=42)
    colorbar.set_label("Goal rate", fontsize=10)
    colorbar.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
    colorbar.set_ticklabels(["0%", "25%", "50%", "75%", "100%"])

    info.set_facecolor("#f8faf7")
    for spine in info.spines.values():
        spine.set_visible(False)
    info.set_xticks([])
    info.set_yticks([])
    info.set_xlim(0.0, 1.0)
    info.set_ylim(0.0, 1.0)

    touch = finite_values(trials, "first_touch_s")
    kick = finite_values(trials, "first_kick_s")
    angular = finite_values(trials, "first_kick_error_deg")
    improvement = finite_values(trials, "approach_improvement_deg")
    feet = [trial["first_kick_foot"] for trial in trials
            if trial.get("first_kick_foot", -1) >= 0]
    left_rate = feet.count(0) / len(feet) if feet else float("nan")

    info.text(0.07, 0.95, "FINAL G1 EVALUATION", fontsize=10, fontweight="bold",
              color="#5b7468", va="top")
    info.text(0.07, 0.90, "8192 trials", fontsize=25, fontweight="bold",
              color="#172c23", va="top")
    info.text(0.07, 0.845, "126 fixed ball positions", fontsize=11,
              color="#405b4f", va="top")

    outcomes = [
        ("GOAL", summary["rates"]["success"], "#26734d"),
        ("OOB", summary["rates"]["oob"], "#b2533c"),
        ("FALL", summary["rates"]["fall"], "#8b6c35"),
        ("TIMEOUT", summary["rates"]["timeout"], "#69766f"),
    ]
    y = 0.76
    for label, value, color in outcomes:
        info.text(0.07, y, label, fontsize=9, fontweight="bold", color="#52675e")
        info.text(0.93, y, "{:.2%}".format(value), fontsize=15, fontweight="bold",
                  color=color, ha="right")
        y -= 0.065

    info.plot([0.07, 0.93], [0.49, 0.49], color="#d4ddd7", linewidth=1)
    metrics = [
        ("Touch coverage", "{:.1%}".format(len(touch) / len(trials))),
        ("Kick coverage", "{:.1%}".format(len(kick) / len(trials))),
        ("First touch", "{:.2f} s".format(np.mean(touch))),
        ("First kick", "{:.2f} s".format(np.mean(kick))),
        ("Kick angular error", "{:.1f} deg".format(np.mean(angular))),
        ("Approach improvement", "+{:.1f} deg".format(np.mean(improvement))),
        ("First-kick foot", "L {:.0%} / R {:.0%}".format(left_rate, 1.0 - left_rate)),
    ]
    y = 0.445
    for label, value in metrics:
        info.text(0.07, y, label, fontsize=9.5, color="#52675e")
        info.text(0.93, y, value, fontsize=10.5, fontweight="bold",
                  color="#1b3127", ha="right")
        y -= 0.052

    info.plot([0.07, 0.93], [0.075, 0.075], color="#d4ddd7", linewidth=1)
    protocol = "Flat terrain | nominal physics\nvirtual perception | no pushes\nno ball perturbations | seed {}".format(
        metadata["rand_seed"])
    info.text(0.07, 0.055, protocol, fontsize=8.5, color="#60766b",
              va="top", linespacing=1.45)

    figure.text(0.055, 0.015,
                "Figure 3A-style local adaptation. Field markings are schematic; "
                "cell values and metrics are measured from the frozen s10 checkpoint.",
                fontsize=8.5, color="#5a6d64")
    figure.savefig(output, dpi=180, bbox_inches="tight", facecolor=figure.get_facecolor())
    plt.close(figure)


def main():
    args = parse_args()
    with open(args.results, "r", encoding="utf-8") as file:
        payload = json.load(file)
    output_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(output_dir, exist_ok=True)
    render(payload, args.output, args.title)
    print("wrote {}".format(os.path.abspath(args.output)))


if __name__ == "__main__":
    main()