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


def validate_event_types(durations, event_types):
    """Coerce durations with integer cause codes (0 = right-censored)."""
    event_types = np.asarray(event_types)
    if event_types.ndim == 0:
        event_types = event_types.reshape(1)
    if event_types.ndim != 1:
        raise ValueError("event_types must be one-dimensional")
    dummy = np.zeros(event_types.shape[0], dtype=bool)
    durations, _ = validate_durations_events(durations, dummy)
    if not np.all(np.isfinite(np.asarray(event_types, dtype=float))):
        raise ValueError("event_types must contain only finite values")
    if np.any(np.asarray(event_types, dtype=float) < 0):
        raise ValueError("event_types must be non-negative (0 = censored)")
    as_float = np.asarray(event_types, dtype=float)
    rounded = np.rint(as_float)
    if not np.allclose(as_float, rounded):
        raise ValueError("event_types must be integer cause codes")
    return durations, rounded.astype(int)


def observed_causes(event_types):
    """Sorted tuple of positive cause codes present in ``event_types``."""
    event_types = np.asarray(event_types)
    positive = event_types[event_types > 0]
    if positive.size == 0:
        return ()
    return tuple(int(code) for code in np.unique(positive))


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