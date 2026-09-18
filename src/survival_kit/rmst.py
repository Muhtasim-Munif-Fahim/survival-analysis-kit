"""Restricted mean survival time from a Kaplan-Meier curve."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from .utils import validate_durations_events


@dataclass
class RMSTResult:
    """Area under a Kaplan-Meier curve up to a truncation time ``tau``."""

    tau: float
    rmst: float
    rmtl: float
    std_err: float
    ci_lower: float
    ci_upper: float
    n_observations: int
    n_events: int


@dataclass
class RMSTDifferenceResult:
    """Two-sample comparison of restricted mean survival times."""

    tau: float
    rmst_a: float
    rmst_b: float
    difference: float
    std_err: float
    ci_lower: float
    ci_upper: float
    z_score: float
    p_value: float
    std_err_a: float
    std_err_b: float


def default_truncation_time(*duration_groups):
    """Largest identifiable ``tau``: the minimum of each group's last follow-up."""
    if not duration_groups:
        raise ValueError("at least one duration vector is required")
    maxima = [float(np.max(np.asarray(durations, dtype=float))) for durations in duration_groups]
    return min(maxima)


def _resolve_tau(tau, *duration_groups):
    identifiable = default_truncation_time(*duration_groups)
    if tau is None:
        return identifiable
    tau = float(tau)
    if not np.isfinite(tau) or tau <= 0.0:
        raise ValueError("tau must be a finite positive number")
    if tau > identifiable:
        raise ValueError(
            "tau must not exceed the last follow-up time "
            f"({identifiable})"
        )
    return tau


def area_under_survival(times, survival, tau):
    """Integrate a right-continuous KM step function from 0 to ``tau``."""
    times = np.asarray(times, dtype=float)
    survival = np.asarray(survival, dtype=float)
    tau = float(tau)
    if times.size == 0:
        return tau
    knots = times[times <= tau]
    levels = survival[times <= tau]
    breaks = np.concatenate(([0.0], knots, [tau]))
    heights = np.concatenate(([1.0], levels))
    return float(np.dot(np.diff(breaks), heights))


def _greenwood_increments(at_risk, deaths):
    denom = at_risk * (at_risk - deaths)
    raw = deaths / np.where(denom > 0, denom, 1.0)
    return np.where(denom > 0, raw, 0.0)


def restricted_mean_survival_time(durations, events, tau=None, confidence_level=0.95):
    """Estimate RMST(``tau``) = ``integral_0^tau S(t) dt`` from the KM curve.

    The Greenwood plug-in variance is
    ``sum_i [integral_{t_i}^tau S(u) du]^2 * d_i / (n_i (n_i - d_i))``
    over event times ``t_i <= tau``. ``tau`` defaults to the last observed
    follow-up time.
    """
    durations, events = validate_durations_events(durations, events)
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie strictly between 0 and 1")
    tau = _resolve_tau(tau, durations)

    in_window = events & (durations <= tau)
    event_times = np.unique(durations[in_window])
    n_events = int(np.count_nonzero(in_window))

    if event_times.size == 0:
        rmst = tau
        std_err = 0.0
    else:
        at_risk = np.array([np.count_nonzero(durations >= t) for t in event_times], dtype=float)
        deaths = np.array(
            [np.count_nonzero((durations == t) & events) for t in event_times],
            dtype=float,
        )
        survival = np.cumprod(1.0 - deaths / at_risk)
        rmst = area_under_survival(event_times, survival, tau)

        wk_time = np.concatenate([event_times, [tau]])
        time_diff = np.diff(np.concatenate([[0.0], wk_time]))
        areas = time_diff * np.concatenate([[1.0], survival])

        wk_var = np.concatenate([_greenwood_increments(at_risk, deaths), [0.0]])
        remaining_area = np.cumsum(areas[1:][::-1])[::-1]
        variance = float(np.sum(remaining_area**2 * wk_var[:-1]))
        std_err = float(np.sqrt(max(variance, 0.0)))

    z = float(norm.ppf(0.5 + confidence_level / 2.0))
    half = z * std_err
    return RMSTResult(
        tau=tau,
        rmst=rmst,
        rmtl=tau - rmst,
        std_err=std_err,
        ci_lower=max(rmst - half, 0.0),
        ci_upper=min(rmst + half, tau),
        n_observations=int(durations.size),
        n_events=n_events,
    )


def rmst_difference_test(
    durations_a,
    events_a,
    durations_b,
    events_b,
    tau=None,
    confidence_level=0.95,
):
    """Test ``RMST_a(tau) - RMST_b(tau)`` for two independent samples.

    Groups are independent, so the difference variance is the sum of the
    Greenwood plug-in variances. ``tau`` defaults to the earlier of the two
    last follow-up times.
    """
    durations_a, events_a = validate_durations_events(durations_a, events_a)
    durations_b, events_b = validate_durations_events(durations_b, events_b)
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie strictly between 0 and 1")
    tau = _resolve_tau(tau, durations_a, durations_b)

    arm_a = restricted_mean_survival_time(
        durations_a, events_a, tau=tau, confidence_level=confidence_level
    )
    arm_b = restricted_mean_survival_time(
        durations_b, events_b, tau=tau, confidence_level=confidence_level
    )
    difference = arm_a.rmst - arm_b.rmst
    std_err = float(np.sqrt(arm_a.std_err**2 + arm_b.std_err**2))
    if std_err == 0.0:
        z_score = 0.0 if difference == 0.0 else float(np.sign(difference) * np.inf)
        p_value = 1.0 if difference == 0.0 else 0.0
    else:
        z_score = float(difference / std_err)
        p_value = float(2.0 * norm.sf(abs(z_score)))

    z = float(norm.ppf(0.5 + confidence_level / 2.0))
    half = z * std_err
    return RMSTDifferenceResult(
        tau=tau,
        rmst_a=arm_a.rmst,
        rmst_b=arm_b.rmst,
        difference=difference,
        std_err=std_err,
        ci_lower=difference - half,
        ci_upper=difference + half,
        z_score=z_score,
        p_value=p_value,
        std_err_a=arm_a.std_err,
        std_err_b=arm_b.std_err,
    )
