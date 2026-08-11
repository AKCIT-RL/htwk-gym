import math
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from utils.soccer_behavior_metrics import (
    approach_angle_improvement,
    classify_fall_phase,
    distance_bin,
)


def test_distance_bin_boundaries():
    assert distance_bin(0.0) == 0
    assert distance_bin(0.999) == 0
    assert distance_bin(1.0) == 1
    assert distance_bin(4.0) == 3
    assert distance_bin(8.0) == 4


def test_distance_bin_rejects_invalid_values():
    with pytest.raises(ValueError, match="finite and non-negative"):
        distance_bin(-0.1)
    with pytest.raises(ValueError, match="finite and non-negative"):
        distance_bin(float("nan"))


def test_approach_angle_improvement_sign():
    assert approach_angle_improvement(140.0, 35.0) == pytest.approx(105.0)
    assert approach_angle_improvement(20.0, 60.0) == pytest.approx(-40.0)


def test_approach_angle_rejects_non_finite_or_out_of_range():
    with pytest.raises(ValueError, match="finite"):
        approach_angle_improvement(math.inf, 10.0)
    with pytest.raises(ValueError, match="\[0, 180\]"):
        approach_angle_improvement(181.0, 10.0)


def test_fall_phase_precedence_and_windows():
    assert classify_fall_phase(None, None, None, 3.0) == "approach"
    assert classify_fall_phase(1.0, 2.8, None, 3.0) == "contact"
    assert classify_fall_phase(1.0, 2.9, 2.5, 3.0) == "post_kick"
    assert classify_fall_phase(1.0, 1.5, 1.8, 3.0) == "other"


def test_fall_phase_rejects_negative_time():
    with pytest.raises(ValueError, match="non-negative"):
        classify_fall_phase(None, None, None, -0.1)