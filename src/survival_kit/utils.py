"""Shared validation and step-function helpers."""

from __future__ import annotations

import numpy as np


def validate_durations_events(durations, events):
    """Coerce and validate paired duration/event observations.

    Returns float durations and a boolean event mask.
    """
    durations = np.asarray(durations, dtype=float)
    events = np.asarray(events).astype(bool)
    if durations.ndim != 1 or events.ndim != 1:
        raise ValueError("durations and events must be one-dimensional")
    if durations.shape[0] != events.shape[0]:
        raise ValueError("durations and events must have the same length")
    if durations.size == 0:
        raise ValueError("at least one observation is required")
    if not np.all(np.isfinite(durations)):
        raise ValueError("durations must contain only finite values")
    if np.any(durations < 0):
        raise ValueError("durations must be non-negative")
    return durations, events


def step_value(times, values, query, outside=0.0):
    """Evaluate a right-continuous step function at ``query``.

    The function holds ``values[i]`` on ``[times[i], times[i + 1])``; before
    the first knot it returns ``outside``.
    """
    times = np.asarray(times, dtype=float)
    values = np.asarray(values, dtype=float)
    query = np.asarray(query, dtype=float)
    scalar = query.ndim == 0
    flat = np.atleast_1d(query)
    idx = np.searchsorted(times, flat, side="right") - 1
    result = np.full(flat.shape, float(outside), dtype=float)
    valid = idx >= 0
    result[valid] = values[idx[valid]]
    return float(result[0]) if scalar else result