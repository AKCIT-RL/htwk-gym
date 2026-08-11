import os
import sys

import numpy as np
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

from utils.soccer_spatial_benchmark import (  # noqa: E402
    OUTCOMES,
    aggregate_outcomes,
    allocate_trials,
    make_field_grid,
    summarize_trials,
    wilson_interval,
)


def test_paper_field_uses_126_one_meter_cells():
    grid = make_field_grid()

    assert grid.shape == (126, 4)
    np.testing.assert_array_equal(np.unique(grid[:, 0]), np.arange(14))
    np.testing.assert_array_equal(np.unique(grid[:, 1]), np.arange(9))
    assert (grid[0, 2], grid[0, 3]) == (-6.5, -4.0)
    assert (grid[-1, 2], grid[-1, 3]) == (6.5, 4.0)


def test_8192_trials_are_balanced_and_stable():
    cell_ids = allocate_trials(126, 8192)
    counts = np.bincount(cell_ids, minlength=126)

    assert len(cell_ids) == 8192
    assert counts.min() == 65
    assert counts.max() == 66
    np.testing.assert_array_equal(cell_ids[:126], np.arange(126))
    np.testing.assert_array_equal(np.flatnonzero(counts == 66), [0, 1])


def test_outcomes_are_exclusive_and_sum_to_denominator():
    cell_ids = np.asarray([0, 0, 0, 0, 1, 1])
    outcomes = np.asarray(["success", "oob", "fall", "timeout", "success", "success"])

    result = aggregate_outcomes(cell_ids, outcomes, num_cells=2)

    assert result["outcomes"] == OUTCOMES
    np.testing.assert_array_equal(result["counts"][0], [1, 1, 1, 1])
    np.testing.assert_array_equal(result["counts"][1], [2, 0, 0, 0])
    np.testing.assert_array_equal(result["trials"], [4, 2])
    np.testing.assert_allclose(result["rates"].sum(axis=1), 1.0)


def test_empty_cells_are_nan_not_zero_rate():
    result = aggregate_outcomes([0], ["timeout"], num_cells=2)

    assert np.isnan(result["rates"][1]).all()
    assert np.isnan(result["success_ci"][1]).all()


def test_invalid_outcome_fails_early():
    with pytest.raises(ValueError, match="unknown outcomes"):
        aggregate_outcomes([0], ["unknown"], num_cells=1)


def test_wilson_interval_contains_observed_rate():
    low, high = wilson_interval(7, 10)

    assert low < 0.7
    assert high > 0.7


def test_summary_is_json_safe_and_preserves_denominators():
    grid = make_field_grid(field_length=2.0, field_width=1.0)
    trials = [
        {"cell_id": 0, "outcome": "success"},
        {"cell_id": 0, "outcome": "oob"},
        {"cell_id": 1, "outcome": "timeout"},
    ]

    summary = summarize_trials(grid, trials)

    assert summary["total_trials"] == 3
    assert summary["counts"] == {
        "success": 1, "oob": 1, "fall": 0, "timeout": 1,
    }
    assert summary["cells"][0]["trials"] == 2
    assert summary["cells"][1]["trials"] == 1
    assert summary["cells"][0]["rates"]["success"] == pytest.approx(0.5)