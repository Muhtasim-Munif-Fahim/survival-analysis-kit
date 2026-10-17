"""Log-rank tests for comparing survival curves across groups.

Besides the standard (unweighted) log-rank test this module provides the
weighted family of k-sample tests,

    U_g = sum_t w(t) * (d_g(t) - d(t) * n_g(t) / n(t)),

with the hypergeometric covariance scaled by ``w(t)^2``. The weight decides
which part of follow-up drives the test: Gehan-Breslow (``n(t)``) and
Tarone-Ware (``sqrt(n(t))``) emphasise early differences, Peto-Peto uses a
modified survival estimate, and Fleming-Harrington ``G(rho, gamma)`` uses
``S(t-)^rho * (1 - S(t-))^gamma`` from the pooled Kaplan-Meier curve
(``rho > 0`` early, ``gamma > 0`` late / delayed effects, ``(0, 0)`` the
log-rank test). Every test can be stratified: scores and covariances are
computed within strata and summed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import chi2

from .utils import validate_durations_events


@dataclass
class LogRankResult:
    """Outcome of a k-group log-rank comparison."""

    statistic: float
    p_value: float
    degrees_of_freedom: int
    observed_deaths: np.ndarray
    expected_deaths: np.ndarray


def log_rank_test_groups(durations, events, groups):
    """Compare survival across all groups with the k-sample log-rank test.

    Observed minus expected deaths and their hypergeometric covariance are
    accumulated over the pooled event times; the statistic is the quadratic
    form of the first ``k - 1`` contrasts.
    """
    durations, events = validate_durations_events(durations, events)
    groups = np.asarray(groups)
    if groups.ndim != 1 or groups.shape[0] != durations.shape[0]:
        raise ValueError("groups must be one-dimensional and match the durations")
    labels, index = np.unique(groups, return_inverse=True)
    if labels.size < 2:
        raise ValueError("at least two distinct groups are required")
    k = labels.size
    index = index.astype(int)

    observed = np.zeros(k)
    expected = np.zeros(k)
    covariance = np.zeros((k, k))
    for t in np.unique(durations[events]):
        at_risk_mask = durations >= t
        at_event_mask = at_risk_mask & (durations == t) & events
        n_total = int(np.count_nonzero(at_risk_mask))
        d_total = int(np.count_nonzero(at_event_mask))
        n_g = np.bincount(index[at_risk_mask], minlength=k).astype(float)
        d_g = np.bincount(index[at_event_mask], minlength=k).astype(float)
        observed += d_g
        expected += d_total * n_g / n_total
        if n_total < 2:
            continue
        factor = d_total * (n_total - d_total) / ((n_total - 1) * n_total)
        covariance += factor * (np.diag(n_g) - np.outer(n_g, n_g) / n_total)

    m = k - 1
    contrast = (observed - expected)[:m]
    sub_covariance = covariance[:m, :m]
    try:
        statistic = float(contrast @ np.linalg.solve(sub_covariance, contrast))
    except np.linalg.LinAlgError:
        statistic = float(contrast @ np.linalg.pinv(sub_covariance) @ contrast)
    statistic = max(statistic, 0.0)

    return LogRankResult(
        statistic=statistic,
        p_value=float(chi2.sf(statistic, m)),
        degrees_of_freedom=m,
        observed_deaths=observed,
        expected_deaths=expected,
    )


def log_rank_test(durations_a, events_a, durations_b, events_b):
    """Two-sample log-rank test between two independent cohorts."""
    a_durations, a_events = validate_durations_events(durations_a, events_a)
    b_durations, b_events = validate_durations_events(durations_b, events_b)
    return log_rank_test_groups(
        np.concatenate([a_durations, b_durations]),
        np.concatenate([a_events, b_events]),
        np.concatenate(
            [
                np.zeros(a_durations.size, dtype=int),
                np.ones(b_durations.size, dtype=int),
            ]
        ),
    )


WEIGHTINGS = ("log-rank", "wilcoxon", "tarone-ware", "peto-peto", "fleming-harrington")


@dataclass
class WeightedLogRankResult:
    """Outcome of a (possibly stratified) weighted k-group log-rank test."""

    statistic: float
    p_value: float
    degrees_of_freedom: int
    weighting: str
    rho: float
    gamma: float
    groups: np.ndarray
    observed_deaths: np.ndarray
    expected_deaths: np.ndarray
    weighted_score: np.ndarray
    n_strata: int


def _normalize_weighting(weighting):
    key = str(weighting).lower().replace("_", "-").replace(" ", "-")
    aliases = {
        "logrank": "log-rank",
        "gehan": "wilcoxon",
        "gehan-breslow": "wilcoxon",
        "breslow": "wilcoxon",
        "fh": "fleming-harrington",
    }
    key = aliases.get(key, key)
    if key not in WEIGHTINGS:
        raise ValueError(f"weighting must be one of {WEIGHTINGS}")
    return key


def _stratum_scores(durations, events, index, k, weighting, rho, gamma):
    observed = np.zeros(k)
    expected = np.zeros(k)
    score = np.zeros(k)
    covariance = np.zeros((k, k))
    km_left = 1.0  # pooled KM S(t-) for Fleming-Harrington
    peto = 1.0  # Peto-Peto modified survival
    for t in np.unique(durations[events]):
        at_risk_mask = durations >= t
        at_event_mask = at_risk_mask & (durations == t) & events
        n_total = int(np.count_nonzero(at_risk_mask))
        d_total = int(np.count_nonzero(at_event_mask))
        n_g = np.bincount(index[at_risk_mask], minlength=k).astype(float)
        d_g = np.bincount(index[at_event_mask], minlength=k).astype(float)
        peto *= 1.0 - d_total / (n_total + 1.0)
        if weighting == "log-rank":
            w = 1.0
        elif weighting == "wilcoxon":
            w = float(n_total)
        elif weighting == "tarone-ware":
            w = float(np.sqrt(n_total))
        elif weighting == "peto-peto":
            w = peto
        else:
            w = km_left**rho * (1.0 - km_left) ** gamma
        exp_g = d_total * n_g / n_total
        observed += d_g
        expected += exp_g
        score += w * (d_g - exp_g)
        if n_total >= 2:
            factor = d_total * (n_total - d_total) / ((n_total - 1) * n_total)
            covariance += w * w * factor * (np.diag(n_g) - np.outer(n_g, n_g) / n_total)
        km_left *= 1.0 - d_total / n_total
    return observed, expected, score, covariance


def weighted_log_rank_test_groups(
    durations,
    events,
    groups,
    weighting="fleming-harrington",
    rho=0.0,
    gamma=0.0,
    strata=None,
):
    """Weighted (and optionally stratified) k-sample log-rank test.

    Parameters
    ----------
    durations, events, groups : array-like
        Follow-up times, event indicators and group labels.
    weighting : str
        ``"log-rank"`` (w = 1), ``"wilcoxon"`` (Gehan-Breslow, w = n(t)),
        ``"tarone-ware"`` (w = sqrt(n(t))), ``"peto-peto"``
        (w = prod_{s <= t} (1 - d(s) / (n(s) + 1))) or
        ``"fleming-harrington"`` (w = S(t-)^rho (1 - S(t-))^gamma, pooled
        Kaplan-Meier left limit). Weights are computed within each stratum.
    rho, gamma : float
        Fleming-Harrington exponents (ignored for other weightings).
    strata : array-like, optional
        Stratum labels; groups are compared within strata and the scores and
        covariances are summed.

    Returns
    -------
    WeightedLogRankResult
        Chi-square statistic on ``k - 1`` degrees of freedom, the summed
        weighted scores, and unweighted observed / expected deaths.
    """
    durations, events = validate_durations_events(durations, events)
    groups = np.asarray(groups)
    if groups.ndim != 1 or groups.shape[0] != durations.shape[0]:
        raise ValueError("groups must be one-dimensional and match the durations")
    weighting = _normalize_weighting(weighting)
    rho, gamma = float(rho), float(gamma)
    if weighting == "fleming-harrington" and (rho < 0 or gamma < 0):
        raise ValueError("rho and gamma must be non-negative")
    labels, index = np.unique(groups, return_inverse=True)
    if labels.size < 2:
        raise ValueError("at least two distinct groups are required")
    k = labels.size
    index = index.astype(int)
    if strata is None:
        strata_index = np.zeros(durations.shape[0], dtype=int)
    else:
        strata = np.asarray(strata)
        if strata.ndim != 1 or strata.shape[0] != durations.shape[0]:
            raise ValueError("strata must be one-dimensional and match the durations")
        strata_index = np.unique(strata, return_inverse=True)[1].astype(int)
    n_strata = int(strata_index.max()) + 1

    observed = np.zeros(k)
    expected = np.zeros(k)
    score = np.zeros(k)
    covariance = np.zeros((k, k))
    for s in range(n_strata):
        mask = strata_index == s
        o, e, u, v = _stratum_scores(
            durations[mask], events[mask], index[mask], k, weighting, rho, gamma
        )
        observed += o
        expected += e
        score += u
        covariance += v

    m = k - 1
    contrast = score[:m]
    sub_covariance = covariance[:m, :m]
    if np.allclose(sub_covariance, 0.0):
        statistic = 0.0
    else:
        try:
            statistic = float(contrast @ np.linalg.solve(sub_covariance, contrast))
        except np.linalg.LinAlgError:
            statistic = float(contrast @ np.linalg.pinv(sub_covariance) @ contrast)
    statistic = max(statistic, 0.0)
    return WeightedLogRankResult(
        statistic=statistic,
        p_value=float(chi2.sf(statistic, m)),
        degrees_of_freedom=m,
        weighting=weighting,
        rho=rho,
        gamma=gamma,
        groups=labels,
        observed_deaths=observed,
        expected_deaths=expected,
        weighted_score=score,
        n_strata=n_strata,
    )
