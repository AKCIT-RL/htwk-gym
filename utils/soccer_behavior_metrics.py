"""Pure helpers for soccer behavioral evaluation metrics."""

import numpy as np


DISTANCE_BINS_M = (0.0, 1.0, 2.0, 4.0, 8.0, float("inf"))
FALL_PHASES = ("approach", "contact", "post_kick", "other")


def distance_bin(distance_m, edges=DISTANCE_BINS_M):
    if not np.isfinite(distance_m) or distance_m < 0.0:
        raise ValueError("distance_m must be finite and non-negative")
    for index in range(len(edges) - 1):
        if edges[index] <= distance_m < edges[index + 1]:
            return index
    raise ValueError("distance_m is outside the configured bins")


def approach_angle_improvement(initial_deg, impact_deg):
    if not np.isfinite(initial_deg) or not np.isfinite(impact_deg):
        raise ValueError("approach angles must be finite")
    if not 0.0 <= initial_deg <= 180.0 or not 0.0 <= impact_deg <= 180.0:
        raise ValueError("approach angles must be in [0, 180]")
    return initial_deg - impact_deg


def classify_fall_phase(first_touch_s, last_touch_s, last_kick_s, fall_s,
                        contact_window_s=0.5, post_kick_window_s=1.0):
    if fall_s < 0.0:
        raise ValueError("fall_s must be non-negative")
    if first_touch_s is None or first_touch_s > fall_s:
        return "approach"
    if last_kick_s is not None and 0.0 <= fall_s - last_kick_s <= post_kick_window_s:
        return "post_kick"
    if last_touch_s is not None and 0.0 <= fall_s - last_touch_s <= contact_window_s:
        return "contact"
    return "other"