"""Aalen-Johansen cumulative incidence for competing events."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import chi2, norm

from .kaplan_meier import fit_kaplan_meier
from .utils import observed_causes, step_value, validate_event_types


@dataclass
class CumulativeIncidence:
    """Aalen-Johansen cumulative incidence curve for one cause."""

    time: np.ndarray
    incidence: np.ndarray
    std_err: np.ndarray
    ci_lower: np.ndarray
    ci_upper: np.ndarray
    cause: int
    overall_survival: np.ndarray
    n_observations: int
    n_events: int

    def at(self, times):
        """Cause-specific cumulative incidence just after each time in ``times``."""
        return step_value(self.time, self.incidence, times, outside=0.0)


@dataclass
class GrayTestResult:
    """k-sample Gray test comparing cumulative incidence for one cause."""

    statistic: float
    p_value: float
    degrees_of_freedom: int
    cause: int
    observed: np.ndarray
    expected: np.ndarray


def _empty_incidence(cause, n_observations, confidence_level):
    z = float(norm.ppf(0.5 + confidence_level / 2.0))
    empty = np.array([], dtype=float)
    return CumulativeIncidence(
        time=empty,
        incidence=empty,
        std_err=empty,
        ci_lower=empty,
        ci_upper=empty,
        cause=int(cause),
        overall_survival=empty,
        n_observations=int(n_observations),
        n_events=0,
    )


def _aalen_johansen_tables(durations, event_types, cause):
    """CIF, overall survival, and Aalen variance pieces on any-event times."""
    any_event = event_types > 0
    event_times = np.unique(durations[any_event])
    if event_times.size == 0:
        return event_times, None, None, None, None

    at_risk = np.array(
        [np.count_nonzero(durations >= t) for t in event_times], dtype=float
    )
    deaths = np.array(
        [np.count_nonzero((durations == t) & any_event) for t in event_times],
        dtype=float,
    )
    cause_deaths = np.array(
        [np.count_nonzero((durations == t) & (event_types == cause)) for t in event_times],
        dtype=float,
    )

    steps = 1.0 - deaths / at_risk
    survival = np.cumprod(steps)
    survival_before = np.ones(event_times.size, dtype=float)
    survival_before[1:] = survival[:-1]

    incidence = np.cumsum(survival_before * (cause_deaths / at_risk))

    denom = at_risk * (at_risk - deaths)
    raw = deaths / np.where(denom > 0, denom, 1.0)
    term1 = np.where(denom > 0, raw, 0.0)
    term2 = survival_before**2 * (at_risk - cause_deaths) * cause_deaths / at_risk**3
    term3 = survival_before * cause_deaths / at_risk**2

    delta = incidence.reshape(-1, 1) - incidence.reshape(1, -1)
    lower = np.tril(np.ones((event_times.size, event_times.size), dtype=bool))
    contrib = (
        delta**2 * term1.reshape(1, -1)
        + term2.reshape(1, -1)
        - 2.0 * delta * term3.reshape(1, -1)
    )
    variance = np.where(lower, contrib, 0.0).sum(axis=1)
    variance = np.maximum(variance, 0.0)
    return event_times, incidence, survival, cause_deaths, variance


def fit_cumulative_incidence(durations, event_types, cause=1, confidence_level=0.95):
    """Estimate the cause-specific cumulative incidence function.

    ``event_types`` are integer codes: ``0`` is right-censoring and each
    positive integer is a competing event. The estimator is Aalen-Johansen,
    ``F_k(t) = sum_{t_j <= t} S(t_j-) d_{jk} / n_j``, where ``S`` is the
    Kaplan-Meier curve that treats any event as a failure. Pointwise
    variances follow the Aalen / Coviello-Boggess formula, which reduces to
    Greenwood's variance of ``1 - S(t)`` when only one cause is present.
    """
    durations, event_types = validate_event_types(durations, event_types)
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie strictly between 0 and 1")
    cause = int(cause)
    if cause < 1:
        raise ValueError("cause must be a positive integer")

    event_times, incidence, survival, cause_deaths, variance = _aalen_johansen_tables(
        durations, event_types, cause
    )
    n_events = int(np.count_nonzero(event_types == cause))
    if event_times is None or event_times.size == 0 or n_events == 0:
        return _empty_incidence(cause, durations.size, confidence_level)

    keep = cause_deaths > 0
    finite = np.isfinite(variance)
    std_err = np.where(finite, np.sqrt(np.where(finite, variance, 0.0)), np.inf)
    z = float(norm.ppf(0.5 + confidence_level / 2.0))
    lower = np.clip(incidence - z * std_err, 0.0, 1.0)
    upper = np.clip(incidence + z * std_err, 0.0, 1.0)
    return CumulativeIncidence(
        time=event_times[keep],
        incidence=incidence[keep],
        std_err=std_err[keep],
        ci_lower=lower[keep],
        ci_upper=upper[keep],
        cause=cause,
        overall_survival=survival[keep],
        n_observations=int(durations.size),
        n_events=n_events,
    )


def fit_all_cumulative_incidence(durations, event_types, confidence_level=0.95):
    """Fit an Aalen-Johansen curve for every observed competing cause."""
    durations, event_types = validate_event_types(durations, event_types)
    causes = observed_causes(event_types)
    if not causes:
        return ()
    return tuple(
        fit_cumulative_incidence(
            durations, event_types, cause=cause, confidence_level=confidence_level
        )
        for cause in causes
    )


def _censoring_left_limit(curve, times):
    """Left-continuous Kaplan-Meier of the censoring distribution."""
    times = np.asarray(times, dtype=float)
    scalar = times.ndim == 0
    flat = np.atleast_1d(times)
    idx = np.searchsorted(curve.time, flat, side="left") - 1
    result = np.ones(flat.shape, dtype=float)
    valid = idx >= 0
    result[valid] = curve.survival[idx[valid]]
    return float(result[0]) if scalar else result


def _fine_gray_weights(durations, event_types, t, cause, censor_curve):
    """IPCW Fine-Gray risk-set weights at time ``t`` for cause ``cause``."""
    weights = np.zeros(durations.shape[0], dtype=float)
    still_at_risk = durations >= t
    weights[still_at_risk] = 1.0
    competing = (~still_at_risk) & (event_types != 0) & (event_types != cause)
    if not np.any(competing):
        return weights
    g_t = _censoring_left_limit(censor_curve, t)
    g_ti = _censoring_left_limit(censor_curve, durations[competing])
    ratio = np.zeros(g_ti.shape, dtype=float)
    positive = g_ti > 0.0
    ratio[positive] = g_t / g_ti[positive]
    weights[competing] = ratio
    return weights


def gray_test_groups(durations, event_types, groups, cause=1):
    """Compare cause-specific CIFs across groups with Gray's k-sample test.

    The statistic is the Fine-Gray score test of group indicators at the
    null, using IPCW weights so subjects who fail of a competing cause
    remain in the subdistribution risk set. With a single cause the test
    reduces to the log-rank test.
    """
    durations, event_types = validate_event_types(durations, event_types)
    groups = np.asarray(groups)
    if groups.ndim != 1 or groups.shape[0] != durations.shape[0]:
        raise ValueError("groups must be one-dimensional and match the durations")
    cause = int(cause)
    if cause < 1:
        raise ValueError("cause must be a positive integer")

    labels, index = np.unique(groups, return_inverse=True)
    if labels.size < 2:
        raise ValueError("at least two distinct groups are required")
    k = labels.size
    p = k - 1
    index = index.astype(int)
    design = np.eye(k, dtype=float)[index][:, :p]

    observed = np.zeros(k, dtype=float)
    expected = np.zeros(k, dtype=float)
    score = np.zeros(p, dtype=float)
    information = np.zeros((p, p), dtype=float)

    censor_curve = fit_kaplan_meier(durations, event_types == 0)
    cause_times = np.unique(durations[event_types == cause])
    for t in cause_times:
        at_event = (durations == t) & (event_types == cause)
        deaths = int(np.count_nonzero(at_event))
        if deaths == 0:
            continue
        weights = _fine_gray_weights(durations, event_types, t, cause, censor_curve)
        total_weight = float(weights.sum())
        if total_weight <= 0.0:
            continue
        group_weight = np.bincount(index, weights=weights, minlength=k).astype(float)
        group_deaths = np.bincount(index[at_event], minlength=k).astype(float)
        observed += group_deaths
        expected += deaths * group_weight / total_weight

        xbar = (weights.reshape(-1, 1) * design).sum(axis=0) / total_weight
        score += design[at_event].sum(axis=0) - deaths * xbar
        centered = design - xbar
        covariance = (weights.reshape(-1, 1) * centered).T @ centered / total_weight
        if total_weight > 1.0:
            covariance = covariance * (total_weight - deaths) / (total_weight - 1.0)
        information += deaths * covariance

    try:
        statistic = float(score @ np.linalg.solve(information, score))
    except np.linalg.LinAlgError:
        statistic = float(score @ np.linalg.pinv(information) @ score)
    statistic = max(statistic, 0.0)
    return GrayTestResult(
        statistic=statistic,
        p_value=float(chi2.sf(statistic, p)),
        degrees_of_freedom=p,
        cause=cause,
        observed=observed,
        expected=expected,
    )


def gray_test(durations_a, event_types_a, durations_b, event_types_b, cause=1):
    """Two-sample Gray test of cause-specific cumulative incidence."""
    a_durations, a_types = validate_event_types(durations_a, event_types_a)
    b_durations, b_types = validate_event_types(durations_b, event_types_b)
    return gray_test_groups(
        np.concatenate([a_durations, b_durations]),
        np.concatenate([a_types, b_types]),
        np.concatenate(
            [
                np.zeros(a_durations.size, dtype=int),
                np.ones(b_durations.size, dtype=int),
            ]
        ),
        cause=cause,
    )
