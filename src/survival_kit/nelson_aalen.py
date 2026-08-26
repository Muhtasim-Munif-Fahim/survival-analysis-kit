"""Nelson-Aalen estimator of the cumulative hazard function."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from .utils import step_value, validate_durations_events


@dataclass
class CumulativeHazard:
    """Nelson-Aalen cumulative hazard curve on distinct event times."""

    time: np.ndarray
    cumhazard: np.ndarray
    std_err: np.ndarray
    ci_lower: np.ndarray
    ci_upper: np.ndarray

    def at(self, times):
        """Cumulative hazard just after each time in ``times``."""
        return step_value(self.time, self.cumhazard, times, outside=0.0)


def fit_nelson_aalen(durations, events, confidence_level=0.95, variance_method="aalen"):
    """Estimate the cumulative hazard ``H(t) = sum d_j / n_j`` over event times.

    ``variance_method`` selects the pointwise variance estimate: ``"aalen"``
    accumulates ``d_j / n_j**2`` while ``"greenwood"`` accumulates the larger
    ``d_j / (n_j * (n_j - d_j))``.
    """
    durations, events = validate_durations_events(durations, events)
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie strictly between 0 and 1")
    if variance_method not in ("aalen", "greenwood"):
        raise ValueError("variance_method must be 'aalen' or 'greenwood'")

    z = float(norm.ppf(0.5 + confidence_level / 2.0))

    event_times = np.unique(durations[events])
    at_risk = np.array([np.count_nonzero(durations >= t) for t in event_times])
    deaths = np.array(
        [np.count_nonzero((durations == t) & events) for t in event_times]
    )

    cumhazard = np.cumsum(deaths / at_risk)
    if variance_method == "aalen":
        variance = np.cumsum(deaths / at_risk**2)
    else:
        denom = at_risk * (at_risk - deaths)
        raw = deaths / np.where(denom > 0, denom, 1.0)
        variance = np.cumsum(np.where(denom > 0, raw, np.inf))
    finite = np.isfinite(variance)
    std_err = np.where(finite, np.sqrt(np.where(finite, variance, 0.0)), np.inf)

    return CumulativeHazard(
        time=event_times,
        cumhazard=cumhazard,
        std_err=std_err,
        ci_lower=np.maximum(cumhazard - z * std_err, 0.0),
        ci_upper=cumhazard + z * std_err,
    )