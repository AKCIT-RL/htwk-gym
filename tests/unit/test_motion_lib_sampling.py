import sys
from pathlib import Path

import pytest
import torch

MIMICKIT_ROOT = Path(__file__).resolve().parents[2] / "MimicKit"
sys.path.insert(0, str(MIMICKIT_ROOT / "mimickit"))

from anim.motion_lib import MotionLib


def build_motion_lib(lengths):
    library = MotionLib.__new__(MotionLib)
    library._device = "cpu"
    library._motion_lengths = torch.tensor(lengths, dtype=torch.float32)
    return library


def test_sample_time_respects_truncation():
    library = build_motion_lib([15.0, 20.0])
    motion_ids = torch.tensor([0, 0, 1, 1], dtype=torch.long)

    torch.manual_seed(7)
    times = library.sample_time(motion_ids, truncate_time=10.0)

    assert torch.all(times >= 0.0)
    assert torch.all(times[:2] <= 5.0)
    assert torch.all(times[2:] <= 10.0)


def test_sample_time_rejects_short_motion():
    library = build_motion_lib([8.0])

    with pytest.raises(ValueError, match="shortest sampled motion"):
        library.sample_time(torch.zeros(2, dtype=torch.long), truncate_time=10.0)
