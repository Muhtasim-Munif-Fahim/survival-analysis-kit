"""Log-rank tests for comparing survival curves across groups."""

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