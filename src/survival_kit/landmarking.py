"""Landmarking for dynamic survival prediction.

At a landmark time ``L``, only subjects still at risk (``duration > L``)
enter the analysis. Their remaining lifetime is ``duration - L``, and
censoring indicators are unchanged. Fitting a Kaplan–Meier (or Cox) model
on this truncated sample yields the conditional survival
``P(T > L + h | T > L)``.

Repeating over a grid of landmarks produces a family of dynamic
prediction curves without fitting a full joint longitudinal model
(van Houwelingen & Putter).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .kaplan_meier import SurvivalCurve, fit_kaplan_meier

__all__ = [
    "LandmarkResult",
    "landmark_dataset",
    "landmark_kaplan_meier",
    "landmark_survival",
]


@dataclass
class LandmarkResult:
    """Landmarked Kaplan–Meier fit at one landmark time."""

    landmark: float
    n_at_risk: int
    curve: SurvivalCurve

    def survival_at(self, horizon: float) -> float:
        """Estimate ``P(T > landmark + horizon | T > landmark)``."""
        return float(self.curve.at(np.asarray([float(horizon)]))[0])


def landmark_dataset(
    durations,
    events,
    landmark: float,
    covariates=None,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Truncate a right-censored sample at ``landmark``.

    Returns ``(remaining_durations, events, covariates_subset)`` for
    subjects with ``duration > landmark``. Remaining duration is
    ``duration - landmark``.
    """
    durations = np.asarray(durations, dtype=float).ravel()
    events = np.asarray(events).ravel().astype(bool)
    if durations.shape != events.shape:
        raise ValueError("durations and events must have the same shape")
    landmark = float(landmark)
    if not np.isfinite(landmark) or landmark < 0.0:
        raise ValueError("landmark must be a finite non-negative number")
    mask = durations > landmark
    if not np.any(mask):
        raise ValueError(f"no subjects remain at risk after landmark={landmark}")
    rem = durations[mask] - landmark
    ev = events[mask]
    if covariates is None:
        return rem, ev, None
    X = np.asarray(covariates)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.shape[0] != durations.shape[0]:
        raise ValueError("covariates rows must match durations length")
    return rem, ev, X[mask]


def landmark_kaplan_meier(
    durations,
    events,
    landmarks: Sequence[float],
    *,
    confidence_level: float = 0.95,
) -> List[LandmarkResult]:
    """Fit a Kaplan–Meier curve at each landmark time."""
    results: List[LandmarkResult] = []
    for L in landmarks:
        rem, ev, _ = landmark_dataset(durations, events, L)
        curve = fit_kaplan_meier(rem, ev, confidence_level=confidence_level)
        results.append(
            LandmarkResult(landmark=float(L), n_at_risk=int(rem.size), curve=curve)
        )
    return results


def landmark_survival(
    durations,
    events,
    landmark: float,
    horizons: Sequence[float],
) -> Dict[str, np.ndarray]:
    """Conditional survival at several horizons after one landmark.

    Returns a dict with keys ``horizons`` and ``survival`` (arrays).
    """
    rem, ev, _ = landmark_dataset(durations, events, landmark)
    curve = fit_kaplan_meier(rem, ev)
    h = np.asarray(list(horizons), dtype=float)
    if np.any(h < 0):
        raise ValueError("horizons must be non-negative")
    return {"horizons": h, "survival": curve.at(h)}
