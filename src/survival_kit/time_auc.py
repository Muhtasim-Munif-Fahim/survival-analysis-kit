"""Time-dependent ROC AUC for right-censored survival predictions.

Cumulative/dynamic AUC (Heagerty, Lumley & Pepe, 2000) asks, at a horizon
``t``, how well a risk score separates subjects who failed by ``t``
(*cumulative cases*, ``T_i <= t`` with an observed event) from those still
event-free after ``t`` (*dynamic controls*, ``T_j > t``)::

    AUC(t) = P(eta_i > eta_j | T_i <= t, T_j > t)

Subjects censored before ``t`` have unknown status. Uno et al. (2007) and
Hung & Chiang (2010) correct for them with inverse probability of censoring
weights: each case gets ``w_i = 1 / G(T_i-)``, where ``G`` is the
Kaplan-Meier estimate of the censoring survival function. Controls all
share the weight ``1 / G(t)``, which cancels. The estimator is::

    AUC(t) = sum_i sum_j w_i 1(T_j > t) [1(eta_i > eta_j) + 0.5 1(eta_i = eta_j)]
             -------------------------------------------------------------------
                         sum_i w_i  *  #{j : T_j > t}

``G`` can be estimated on a separate training sample (``train_durations``
/ ``train_events``), as Uno recommends. The summary ``mean_auc`` weights
``AUC(t_k)`` by the drop in the Kaplan-Meier survival over each horizon
(Lambert & Chevret, 2016), the same convention as scikit-survival.

Higher ``estimate`` means higher risk. Pass a 2-D ``(n, n_times)`` array
for time-dependent risks (for example ``1 - S(t | x)``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .kaplan_meier import fit_kaplan_meier
from .utils import validate_durations_events

__all__ = ["TimeDependentAUCResult", "cumulative_dynamic_auc"]


@dataclass
class TimeDependentAUCResult:
    """Pointwise cumulative/dynamic AUC and its survival-weighted mean."""

    times: np.ndarray
    auc: np.ndarray
    mean_auc: float
    n_cases: np.ndarray
    n_controls: np.ndarray
    ipcw: bool


def _censoring_curve(durations, events):
    if np.all(events):
        return None
    return fit_kaplan_meier(durations, ~events)


def _left_limit(curve, t):
    """``G(t-)``: censoring survival just before ``t``."""
    if curve is None:
        return np.ones_like(np.asarray(t, dtype=float))
    t = np.asarray(t, dtype=float)
    return curve.at(np.nextafter(t, -np.inf))


def cumulative_dynamic_auc(
    durations,
    events,
    estimate,
    times,
    *,
    train_durations=None,
    train_events=None,
    ipcw: bool = True,
    tied_tol: float = 1e-8,
) -> TimeDependentAUCResult:
    """Cumulative/dynamic time-dependent AUC at each horizon in ``times``.

    Parameters
    ----------
    durations, events:
        Right-censored test observations.
    estimate:
        Risk scores, shape ``(n,)`` (static) or ``(n, len(times))``.
        Higher = earlier expected failure.
    times:
        Strictly increasing horizons. Every horizon must have at least one
        case and one control.
    train_durations, train_events:
        Optional sample for the censoring distribution ``G``. Defaults to
        the test data. Test times beyond the training follow-up have ``G``
        evaluated at its last value.
    ipcw:
        ``False`` gives the unweighted (naive) estimator that simply drops
        subjects censored before ``t``.
    tied_tol:
        Scores closer than this count as ties (half credit).
    """
    durations, events = validate_durations_events(durations, events)
    n = durations.shape[0]
    times = np.asarray(times, dtype=float).ravel()
    if times.size == 0:
        raise ValueError("times must be non-empty")
    if not np.all(np.isfinite(times)):
        raise ValueError("times must be finite")
    if np.any(np.diff(times) <= 0.0):
        raise ValueError("times must be strictly increasing")

    est = np.asarray(estimate, dtype=float)
    if est.ndim == 1:
        if est.shape[0] != n:
            raise ValueError("estimate must have one entry per observation")
        est = np.repeat(est[:, None], times.size, axis=1)
    elif est.ndim == 2:
        if est.shape != (n, times.size):
            raise ValueError("2-D estimate must have shape (n_samples, n_times)")
    else:
        raise ValueError("estimate must be 1-D or 2-D")
    if not np.all(np.isfinite(est)):
        raise ValueError("estimate must contain only finite values")

    if (train_durations is None) != (train_events is None):
        raise ValueError("pass both train_durations and train_events, or neither")
    if train_durations is not None:
        tr_d, tr_e = validate_durations_events(train_durations, train_events)
        g_curve = _censoring_curve(tr_d, tr_e)
    else:
        g_curve = _censoring_curve(durations, events)

    if ipcw:
        g_left = _left_limit(g_curve, durations)
        if np.any((g_left <= 0.0) & events):
            raise ValueError(
                "censoring survival G(T-) is 0 for some events; restrict times to the "
                "censoring follow-up"
            )
        weights = np.where(events, 1.0 / np.maximum(g_left, 1e-300), 0.0)
    else:
        weights = events.astype(float)

    auc = np.empty(times.size)
    n_cases = np.empty(times.size, dtype=int)
    n_controls = np.empty(times.size, dtype=int)
    for k, t in enumerate(times):
        cases = events & (durations <= t)
        controls = durations > t
        n_cases[k] = int(cases.sum())
        n_controls[k] = int(controls.sum())
        if n_cases[k] == 0 or n_controls[k] == 0:
            raise ValueError(
                f"time {t:g} has {n_cases[k]} cases and {n_controls[k]} controls; "
                "every horizon needs at least one of each"
            )
        s_case = est[cases, k]
        s_ctrl = np.sort(est[controls, k])
        # count controls strictly below / tied with each case via sorted search
        lo = np.searchsorted(s_ctrl, s_case - tied_tol, side="left")
        hi = np.searchsorted(s_ctrl, s_case + tied_tol, side="right")
        credit = lo + 0.5 * (hi - lo)
        w = weights[cases]
        auc[k] = float(np.sum(w * credit) / (np.sum(w) * n_controls[k]))

    km = fit_kaplan_meier(durations, events).at(times)
    drop = -np.diff(np.concatenate(([1.0], km)))
    total = 1.0 - km[-1]
    if times.size == 1 or total <= 0.0:
        mean_auc = float(auc[0]) if times.size == 1 else float(np.mean(auc))
    else:
        mean_auc = float(np.dot(auc, drop) / total)

    return TimeDependentAUCResult(
        times=times,
        auc=auc,
        mean_auc=mean_auc,
        n_cases=n_cases,
        n_controls=n_controls,
        ipcw=bool(ipcw),
    )
