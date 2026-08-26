"""Kaplan-Meier estimator of the survival function."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from .utils import step_value, validate_durations_events


@dataclass
class SurvivalCurve:
    """Kaplan-Meier survival curve evaluated on distinct event times."""

    time: np.ndarray
    survival: np.ndarray
    std_err: np.ndarray
    ci_lower: np.ndarray
    ci_upper: np.ndarray

    def at(self, times):
        """Survival probability just after each time in ``times``."""
        return step_value(self.time, self.survival, times, outside=1.0)

    def median(self):
        """Smallest time whose survival estimate drops to 0.5 or below."""
        below = np.nonzero(self.survival <= 0.5)[0]
        return float(self.time[below[0]]) if below.size else None


def fit_kaplan_meier(durations, events, confidence_level=0.95, ci_method="log-log"):
    """Estimate the survival function with Greenwood-based confidence bands.

    ``ci_method`` selects between ``"log-log"`` (complementary log-log
    transform, default) and ``"linear"`` confidence limits. Points where the
    log-log transform is undefined fall back to linear limits.
    """
    durations, events = validate_durations_events(durations, events)
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie strictly between 0 and 1")
    if ci_method not in ("linear", "log-log"):
        raise ValueError("ci_method must be 'linear' or 'log-log'")

    z = float(norm.ppf(0.5 + confidence_level / 2.0))

    event_times = np.unique(durations[events])
    at_risk = np.array([np.count_nonzero(durations >= t) for t in event_times])
    deaths = np.array(
        [np.count_nonzero((durations == t) & events) for t in event_times]
    )

    steps = 1.0 - deaths / at_risk
    survival = np.cumprod(steps)

    denom = at_risk * (at_risk - deaths)
    raw = deaths / np.where(denom > 0, denom, 1.0)
    increments = np.where(denom > 0, raw, np.inf)
    greenwood = np.cumsum(increments)
    std_err = np.where(
        np.isfinite(greenwood), survival * np.sqrt(np.where(np.isfinite(greenwood), greenwood, 0.0)), np.inf
    )

    lower = np.clip(survival - z * std_err, 0.0, 1.0)
    upper = np.clip(survival + z * std_err, 0.0, 1.0)
    if ci_method == "log-log":
        usable = (
            (survival > 0.0)
            & (survival < 1.0)
            & np.isfinite(greenwood)
            & (greenwood > 0.0)
        )
        s = survival[usable]
        theta = np.log(-np.log(s))
        spread = z * np.sqrt(greenwood[usable]) / np.abs(np.log(s))
        lower = lower.copy()
        upper = upper.copy()
        lower[usable] = np.exp(-np.exp(theta + spread))
        upper[usable] = np.exp(-np.exp(theta - spread))

    return SurvivalCurve(event_times, survival, std_err, lower, upper)