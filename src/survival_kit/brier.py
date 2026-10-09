"""Brier score and Integrated Brier Score for right-censored survival.

Implements the IPCW Brier score of Graf et al. (1999) / Gerds & Schumacher
(2006). At a horizon ``t``, subjects who fail by ``t`` contribute
``(0 - S(t))**2 / G(T)``, subjects still at risk after ``t`` contribute
``(1 - S(t))**2 / G(t)``, and subjects censored before ``t`` contribute 0.
``G`` is the Kaplan–Meier estimator of the censoring distribution
(treating censoring as the event).

The Integrated Brier Score (IBS) is the time-average of that curve over
``[0, tau]`` (or over a provided grid).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .kaplan_meier import fit_kaplan_meier
from .utils import step_value, validate_durations_events


@dataclass
class BrierScoreResult:
    """Pointwise IPCW Brier scores on a time grid."""

    times: np.ndarray
    brier: np.ndarray
    n_observations: int
    n_events: int


@dataclass
class IntegratedBrierResult:
    """Integrated Brier Score plus the underlying pointwise curve."""

    tau: float
    ibs: float
    times: np.ndarray
    brier: np.ndarray
    n_observations: int
    n_events: int


def _censoring_survival(durations, events):
    """KM of the censoring distribution: event = not observed failure."""
    censored_as_event = ~events
    # If nobody is censored, G≡1.
    if not np.any(censored_as_event):
        return None
    return fit_kaplan_meier(durations, censored_as_event)


def _g_at(censor_curve, times):
    if censor_curve is None:
        return np.ones_like(np.asarray(times, dtype=float), dtype=float)
    return censor_curve.at(times)


def brier_score(
    durations,
    events,
    surv_probs,
    times,
) -> BrierScoreResult:
    """IPCW Brier score at each time in ``times``.

    Parameters
    ----------
    durations, events:
        Right-censored observations.
    surv_probs:
        Array of shape ``(n_samples, n_times)`` with predicted ``S(t_j | i)``.
        Values should lie in ``[0, 1]``.
    times:
        Horizons ``t_j`` aligned with the columns of ``surv_probs``.
    """
    durations, events = validate_durations_events(durations, events)
    times = np.asarray(times, dtype=float).ravel()
    if times.size == 0:
        raise ValueError("times must be non-empty")
    if np.any(~np.isfinite(times)) or np.any(times < 0.0):
        raise ValueError("times must be finite and non-negative")
    if np.any(np.diff(times) < 0.0):
        raise ValueError("times must be non-decreasing")

    surv = np.asarray(surv_probs, dtype=float)
    if surv.ndim != 2:
        raise ValueError("surv_probs must be 2-D (n_samples, n_times)")
    if surv.shape[0] != durations.shape[0]:
        raise ValueError("surv_probs rows must match the number of observations")
    if surv.shape[1] != times.shape[0]:
        raise ValueError("surv_probs columns must match times")
    if not np.all(np.isfinite(surv)):
        raise ValueError("surv_probs must contain only finite values")

    censor_curve = _censoring_survival(durations, events)
    g_at_times = np.maximum(_g_at(censor_curve, times), 1e-12)
    g_at_duration = np.maximum(_g_at(censor_curve, durations), 1e-12)

    n = durations.shape[0]
    scores = np.zeros(times.shape[0], dtype=float)
    for j, t in enumerate(times):
        s_j = surv[:, j]
        # Event by t
        event_by_t = (durations <= t) & events
        # Survived past t
        survive_past_t = durations > t
        contrib = np.zeros(n, dtype=float)
        contrib[event_by_t] = ((0.0 - s_j[event_by_t]) ** 2) / g_at_duration[event_by_t]
        contrib[survive_past_t] = ((1.0 - s_j[survive_past_t]) ** 2) / g_at_times[j]
        scores[j] = float(np.mean(contrib))

    return BrierScoreResult(
        times=times,
        brier=scores,
        n_observations=int(n),
        n_events=int(np.count_nonzero(events)),
    )


def integrated_brier_score(
    durations,
    events,
    surv_probs,
    times,
    *,
    tau: float | None = None,
) -> IntegratedBrierResult:
    """Integrate the IPCW Brier score over time up to ``tau``.

    Uses the trapezoidal rule on the provided ``times`` grid, restricted to
    ``[0, tau]``. When ``tau`` is ``None``, uses ``max(times)``.
    """
    pointwise = brier_score(durations, events, surv_probs, times)
    grid = pointwise.times
    values = pointwise.brier
    if tau is None:
        tau_val = float(grid[-1])
    else:
        tau_val = float(tau)
        if not np.isfinite(tau_val) or tau_val <= 0.0:
            raise ValueError("tau must be a finite positive number")
        if tau_val > float(grid[-1]) + 1e-12:
            raise ValueError("tau must not exceed the largest evaluation time")

    mask = grid <= tau_val + 1e-12
    g = grid[mask]
    v = values[mask]
    if g.size == 0:
        raise ValueError("no evaluation times at or below tau")
    # Ensure the integral starts at 0 with Brier(0)=0 when grid[0] > 0.
    if g[0] > 0.0:
        g = np.concatenate(([0.0], g))
        v = np.concatenate(([0.0], v))
    if g[-1] < tau_val:
        # Hold last value out to tau.
        g = np.concatenate((g, [tau_val]))
        v = np.concatenate((v, [v[-1]]))
    else:
        # Clip last knot exactly to tau if needed.
        g = g.copy()
        g[-1] = tau_val

    area = float(np.trapezoid(v, g))
    ibs = area / tau_val if tau_val > 0.0 else 0.0
    return IntegratedBrierResult(
        tau=tau_val,
        ibs=ibs,
        times=pointwise.times,
        brier=pointwise.brier,
        n_observations=pointwise.n_observations,
        n_events=pointwise.n_events,
    )


def km_survival_matrix(durations, events, times):
    """Build an ``(n, n_times)`` matrix from the marginal Kaplan–Meier."""
    durations, events = validate_durations_events(durations, events)
    times = np.asarray(times, dtype=float).ravel()
    curve = fit_kaplan_meier(durations, events)
    row = curve.at(times)
    return np.repeat(row.reshape(1, -1), durations.shape[0], axis=0)
