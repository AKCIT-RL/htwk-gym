"""Pure contracts for the Figure 3A-style soccer benchmark."""

import math

import numpy as np


OUTCOMES = ("success", "oob", "fall", "timeout")


def make_field_grid(field_length=14.0, field_width=9.0, cell_size=1.0):
    """Return cell centers as ``[length_index, width_index, x, y]`` rows."""
    if field_length <= 0.0 or field_width <= 0.0 or cell_size <= 0.0:
        raise ValueError("field dimensions and cell_size must be positive")

    num_x = int(round(field_length / cell_size))
    num_y = int(round(field_width / cell_size))
    if not math.isclose(num_x * cell_size, field_length, abs_tol=1.0e-9) \
            or not math.isclose(num_y * cell_size, field_width, abs_tol=1.0e-9):
        raise ValueError("field dimensions must be exact multiples of cell_size")

    x = -0.5 * field_length + (np.arange(num_x) + 0.5) * cell_size
    y = -0.5 * field_width + (np.arange(num_y) + 0.5) * cell_size
    rows = []
    for length_index, x_pos in enumerate(x):
        for width_index, y_pos in enumerate(y):
            rows.append((length_index, width_index, x_pos, y_pos))
    return np.asarray(rows, dtype=np.float64)


def allocate_trials(num_cells, total_trials):
    """Allocate trials in rounds so every cell differs by at most one trial."""
    if num_cells <= 0:
        raise ValueError("num_cells must be positive")
    if total_trials <= 0:
        raise ValueError("total_trials must be positive")

    full_rounds, remainder = divmod(total_trials, num_cells)
    cell_ids = np.tile(np.arange(num_cells, dtype=np.int64), full_rounds)
    if remainder > 0:
        cell_ids = np.concatenate((cell_ids, np.arange(remainder, dtype=np.int64)))
    return cell_ids


def wilson_interval(successes, trials, z=1.959963984540054):
    """Return the Wilson score interval for a binomial proportion."""
    if trials < 0 or successes < 0 or successes > trials:
        raise ValueError("require 0 <= successes <= trials")
    if trials == 0:
        return float("nan"), float("nan")

    proportion = successes / trials
    denominator = 1.0 + z * z / trials
    center = (proportion + z * z / (2.0 * trials)) / denominator
    margin = z * math.sqrt(
        proportion * (1.0 - proportion) / trials + z * z / (4.0 * trials * trials)
    ) / denominator
    return center - margin, center + margin


def aggregate_outcomes(cell_ids, outcomes, num_cells):
    """Aggregate one exclusive terminal outcome for every assigned trial."""
    cell_ids = np.asarray(cell_ids, dtype=np.int64)
    outcomes = np.asarray(outcomes)
    if cell_ids.ndim != 1 or outcomes.ndim != 1 or len(cell_ids) != len(outcomes):
        raise ValueError("cell_ids and outcomes must be same-length vectors")
    if num_cells <= 0 or np.any(cell_ids < 0) or np.any(cell_ids >= num_cells):
        raise ValueError("cell_ids must refer to existing cells")
    unknown = sorted(set(outcomes.tolist()) - set(OUTCOMES))
    if unknown:
        raise ValueError("unknown outcomes: {}".format(unknown))

    counts = np.zeros((num_cells, len(OUTCOMES)), dtype=np.int64)
    for outcome_index, outcome in enumerate(OUTCOMES):
        np.add.at(counts[:, outcome_index], cell_ids[outcomes == outcome], 1)
    trials = counts.sum(axis=1)
    rates = np.divide(
        counts,
        trials[:, None],
        out=np.full(counts.shape, np.nan, dtype=np.float64),
        where=trials[:, None] > 0,
    )
    success_ci = np.asarray(
        [wilson_interval(int(row[0]), int(total)) for row, total in zip(counts, trials)],
        dtype=np.float64,
    )
    return {
        "outcomes": OUTCOMES,
        "counts": counts,
        "trials": trials,
        "rates": rates,
        "success_ci": success_ci,
    }


def summarize_trials(grid, trials):
    """Build a JSON-safe aggregate from per-trial benchmark records."""
    grid = np.asarray(grid, dtype=np.float64)
    if grid.ndim != 2 or grid.shape[1] != 4:
        raise ValueError("grid must contain [length_index, width_index, x, y] rows")
    if len(trials) == 0:
        raise ValueError("at least one completed trial is required")

    cell_ids = np.asarray([trial["cell_id"] for trial in trials], dtype=np.int64)
    outcomes = np.asarray([trial["outcome"] for trial in trials])
    aggregate = aggregate_outcomes(cell_ids, outcomes, len(grid))

    cells = []
    for cell_id, row in enumerate(grid):
        cells.append({
            "cell_id": cell_id,
            "length_index": int(row[0]),
            "width_index": int(row[1]),
            "ball_x_m": float(row[2]),
            "ball_y_m": float(row[3]),
            "trials": int(aggregate["trials"][cell_id]),
            "counts": {
                outcome: int(aggregate["counts"][cell_id, outcome_index])
                for outcome_index, outcome in enumerate(OUTCOMES)
            },
            "rates": {
                outcome: float(aggregate["rates"][cell_id, outcome_index])
                for outcome_index, outcome in enumerate(OUTCOMES)
            },
            "success_ci95": [float(value) for value in aggregate["success_ci"][cell_id]],
        })

    total_counts = {
        outcome: int(aggregate["counts"][:, outcome_index].sum())
        for outcome_index, outcome in enumerate(OUTCOMES)
    }
    total = len(trials)
    return {
        "total_trials": total,
        "outcomes": list(OUTCOMES),
        "counts": total_counts,
        "rates": {outcome: count / total for outcome, count in total_counts.items()},
        "cells": cells,
    }