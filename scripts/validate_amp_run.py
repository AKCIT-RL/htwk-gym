#!/usr/bin/env python3
"""Validate a completed MimicKit AMP training output against finite smoke gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REQUIRED_COLUMNS = {
    "Samples",
    "Test_Return",
    "Test_Episode_Length",
    "Disc_Reward_Mean",
    "Disc_Reward_Std",
    "Disc_Loss",
    "Disc_Grad_Penalty",
    "Disc_Agent_Acc",
    "Disc_Demo_Acc",
    "Actor_Loss",
    "Critic_Loss",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--min-samples", type=int, required=True)
    parser.add_argument("--min-test-episode-length", type=float, default=1.0)
    args = parser.parse_args()

    log_path = args.run_dir / "log.txt"
    model_path = args.run_dir / "model.pt"
    assert log_path.is_file(), f"Missing log: {log_path}"
    assert model_path.is_file() and model_path.stat().st_size > 0, f"Missing model: {model_path}"

    with log_path.open() as stream:
        header = stream.readline().split()
    missing = REQUIRED_COLUMNS - set(header)
    assert not missing, f"Missing log columns: {sorted(missing)}"

    values = np.loadtxt(log_path, skiprows=1, ndmin=2)
    assert values.shape[0] >= 1
    assert values.shape[1] == len(header)
    assert np.isfinite(values).all(), "AMP log contains NaN or Inf"

    last = dict(zip(header, values[-1]))
    assert last["Samples"] >= args.min_samples
    assert last["Test_Episode_Length"] >= args.min_test_episode_length
    assert 0.0 <= last["Disc_Agent_Acc"] <= 1.0
    assert 0.0 <= last["Disc_Demo_Acc"] <= 1.0
    assert last["Disc_Reward_Mean"] >= 0.0

    summary = {key: float(last[key]) for key in sorted(REQUIRED_COLUMNS)}
    summary["rows"] = int(values.shape[0])
    summary["model_bytes"] = model_path.stat().st_size
    print("AMP_RUN_OK")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
