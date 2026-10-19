"""Turnbull nonparametric MLE of the survival function for interval-censored data.

Each observation is only known to fail inside ``(left, right]``:

* ``left == right`` is an exactly observed failure at that time,
* ``right == inf`` is right-censored at ``left``,
* ``left == 0`` is left-censored (failure at some time up to ``right``),
* otherwise the failure is interval-censored between two inspections.

The NPMLE (Turnbull 1976) places all probability mass on the *innermost*
intervals: the regions where a left endpoint is immediately followed by a
right endpoint once every endpoint is sorted.  Masses are found with the
self-consistency (EM) iteration

    p_j <- (1 / n) * sum_i alpha_ij p_j / sum_k alpha_ik p_k,

where ``alpha_ij = 1`` when innermost interval ``j`` lies inside observation
``i``.  The likelihood does not say where the mass sits *within* an innermost
interval, so the survival curve is only identified between them; this module
reports both the upper envelope (mass at the right end) and the lower
envelope (mass at the left end).  For exact and right-censored data the
innermost intervals collapse to the event times and the NPMLE equals the
Kaplan-Meier estimator.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Ordering of endpoints that share a value.  An exact failure at ``v`` is the
# closed interval ``[v, v]`` so its left end must precede right ends at ``v``;
# an ordinary left end ``(v, ...`` excludes ``v`` so it must follow them.
_CLOSED_LEFT = 0
_RIGHT = 1
_OPEN_LEFT = 2


@dataclass
class TurnbullResult:
    """Turnbull NPMLE for interval-censored failure times."""

    left: np.ndarray
    right: np.ndarray
    left_closed: np.ndarray
    probabilities: np.ndarray
    log_likelihood: float
    n_iterations: int
    converged: bool
    max_gradient: float
    n_observations: int
    n_exact: int
    n_right_censored: int
    n_left_censored: int
    n_interval_censored: int

    @property
    def n_intervals(self):
        return int(self.probabilities.shape[0])

    def survival_at(self, times, envelope="upper"):
        """Evaluate ``S(t) = P(T > t)`` of the NPMLE.

        ``envelope="upper"`` places each innermost interval's mass at its
        right end (the largest survival curve consistent with the NPMLE);
        ``"lower"`` places it at the left end.  Outside the innermost
        intervals both envelopes coincide.
        """
        if envelope not in ("upper", "lower"):
            raise ValueError("envelope must be 'upper' or 'lower'")
        times = np.asarray(times, dtype=float)
        scalar = times.ndim == 0
        flat = np.atleast_1d(times)
        if envelope == "upper":
            # Mass of interval j has failed by t once t >= right_j.
            failed = flat[:, None] >= self.right[None, :]
        else:
            # Lower envelope: mass sits just inside the left end, so it has
            # failed once t > left_j (or t >= left_j for a closed left end).
            failed = np.where(
                self.left_closed[None, :],
                flat[:, None] >= self.left[None, :],
                flat[:, None] > self.left[None, :],
            )
        survival = 1.0 - failed.astype(float) @ self.probabilities
        survival = np.clip(survival, 0.0, 1.0)
        return float(survival[0]) if scalar else survival

    def median(self, envelope="upper"):
        """Smallest ``t`` with ``S(t) <= 0.5`` on the chosen envelope (``inf`` if none)."""
        knots = np.unique(np.concatenate((self.left, self.right)))
        knots = knots[np.isfinite(knots)]
        if knots.size == 0:
            return float("inf")
        values = self.survival_at(knots, envelope=envelope)
        hit = np.nonzero(values <= 0.5 + 1e-12)[0]
        if hit.size == 0:
            return float("inf")
        return float(knots[hit[0]])

    def mass_beyond_last_finite(self):
        """Probability placed on an unbounded interval (failure after all inspections)."""
        return float(self.probabilities[~np.isfinite(self.right)].sum())


def validate_intervals(left, right):
    """Coerce and validate ``(left, right]`` censoring intervals."""
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if left.ndim != 1 or right.ndim != 1:
        raise ValueError("left and right must be one-dimensional")
    if left.shape[0] != right.shape[0]:
        raise ValueError("left and right must have the same length")
    if left.size == 0:
        raise ValueError("at least one observation is required")
    if not np.all(np.isfinite(left)):
        raise ValueError("left endpoints must be finite")
    if np.any(np.isnan(right)):
        raise ValueError("right endpoints must not be NaN (use inf for right-censoring)")
    if np.any(left < 0):
        raise ValueError("left endpoints must be non-negative")
    if np.any(right < left):
        raise ValueError("each right endpoint must be >= its left endpoint")
    if np.any((right == left) & (left == 0)):
        raise ValueError("an exact failure at time 0 is not supported")
    return left, right


def turnbull_intervals(left, right):
    """Innermost (Turnbull) intervals and the observation-by-interval incidence matrix.

    Returns ``(int_left, int_right, int_left_closed, alpha)`` where
    ``alpha[i, j]`` is ``True`` when innermost interval ``j`` lies inside
    observation ``i``.
    """
    left, right = validate_intervals(left, right)
    exact = left == right
    left_kind = np.where(exact, _CLOSED_LEFT, _OPEN_LEFT)

    values = np.concatenate((left, right))
    kinds = np.concatenate((left_kind, np.full(right.shape, _RIGHT)))
    order = np.lexsort((kinds, values))
    values_sorted = values[order]
    kinds_sorted = kinds[order]

    # Dense rank of (value, kind) so that identical endpoints compare equal.
    keys = list(zip(values_sorted.tolist(), kinds_sorted.tolist()))
    rank_of = {}
    for key in keys:
        if key not in rank_of:
            rank_of[key] = len(rank_of)

    int_left, int_right, int_closed = [], [], []
    rank_left, rank_right = [], []
    for pos in range(len(keys) - 1):
        k0, k1 = kinds_sorted[pos], kinds_sorted[pos + 1]
        if k0 != _RIGHT and k1 == _RIGHT:
            int_left.append(values_sorted[pos])
            int_right.append(values_sorted[pos + 1])
            int_closed.append(k0 == _CLOSED_LEFT)
            rank_left.append(rank_of[keys[pos]])
            rank_right.append(rank_of[keys[pos + 1]])

    obs_left_rank = np.array(
        [rank_of[(float(v), int(k))] for v, k in zip(left, left_kind)], dtype=int
    )
    obs_right_rank = np.array([rank_of[(float(v), _RIGHT)] for v in right], dtype=int)
    rank_left = np.asarray(rank_left, dtype=int)
    rank_right = np.asarray(rank_right, dtype=int)
    alpha = (obs_left_rank[:, None] <= rank_left[None, :]) & (
        obs_right_rank[:, None] >= rank_right[None, :]
    )
    return (
        np.asarray(int_left, dtype=float),
        np.asarray(int_right, dtype=float),
        np.asarray(int_closed, dtype=bool),
        alpha,
    )


def fit_turnbull(left, right, weights=None, tol=1e-9, max_iter=20000):
    """Fit the Turnbull NPMLE by self-consistency (EM).

    Parameters
    ----------
    left, right:
        Interval endpoints; observation ``i`` fails in ``(left[i], right[i]]``.
        Use ``right = inf`` for right-censoring and ``left == right`` for an
        exact failure.
    weights:
        Optional non-negative case weights (frequencies).
    tol:
        Stop when the largest absolute change in any mass is below ``tol``.
        Small masses whose Kuhn-Tucker gradient is below 1 are then pruned
        and EM is re-run on the reduced support (accepted only if the
        likelihood does not decrease); ``max_gradient`` on the result should
        be ~1 at the NPMLE.
    max_iter:
        Iteration cap; ``converged`` is ``False`` if it is reached.
    """
    left, right = validate_intervals(left, right)
    n = left.shape[0]
    if weights is None:
        weights = np.ones(n, dtype=float)
    else:
        weights = np.asarray(weights, dtype=float)
        if weights.shape != (n,):
            raise ValueError("weights must have one entry per observation")
        if not np.all(np.isfinite(weights)) or np.any(weights < 0):
            raise ValueError("weights must be finite and non-negative")
        if weights.sum() <= 0:
            raise ValueError("weights must not all be zero")
    if tol <= 0:
        raise ValueError("tol must be positive")
    if max_iter < 1:
        raise ValueError("max_iter must be at least 1")

    int_left, int_right, int_closed, alpha = turnbull_intervals(left, right)
    a = alpha.astype(float)
    w = weights / weights.sum()
    m = a.shape[1]
    p = np.full(m, 1.0 / m)
    p, iterations, converged = _self_consistency(a, w, p, tol, int(max_iter))

    # Support reduction.  At the NPMLE every interval with positive mass has
    # gradient d_j = sum_i w_i alpha_ij / (alpha p)_i equal to 1, while EM only
    # shrinks spurious masses geometrically.  Drop small masses whose gradient
    # is clearly below 1 and re-run EM on the reduced support; keep the result
    # only if the likelihood does not get worse.
    for _ in range(min(m, 25)):
        grad = a.T @ _ratio(w, a @ p)
        drop = (p > 0) & (p < 1e-3) & (grad < 1.0 - 1e-6)
        if not np.any(drop):
            break
        trial = p.copy()
        trial[drop] = 0.0
        trial /= trial.sum()
        if np.any((a @ trial <= 0) & (w > 0)):
            break
        trial, extra, trial_converged = _self_consistency(a, w, trial, tol, int(max_iter))
        if _weighted_loglik(a, w, trial) + 1e-10 < _weighted_loglik(a, w, p):
            break
        p, iterations, converged = trial, iterations + extra, trial_converged

    p[p < 1e-14] = 0.0
    p /= p.sum()
    likelihood = a @ p
    positive = weights > 0
    log_likelihood = float(np.sum(weights[positive] * np.log(likelihood[positive])))
    max_gradient = float(np.max(a.T @ _ratio(w, likelihood)))

    exact = left == right
    right_cens = ~np.isfinite(right)
    left_cens = (left == 0) & np.isfinite(right) & ~exact
    interval_cens = ~exact & ~right_cens & ~left_cens
    return TurnbullResult(
        left=int_left,
        right=int_right,
        left_closed=int_closed,
        probabilities=p,
        log_likelihood=log_likelihood,
        n_iterations=iterations,
        converged=converged,
        max_gradient=max_gradient,
        n_observations=int(n),
        n_exact=int(exact.sum()),
        n_right_censored=int(right_cens.sum()),
        n_left_censored=int(left_cens.sum()),
        n_interval_censored=int(interval_cens.sum()),
    )


def _ratio(w, lik):
    return np.divide(w, lik, out=np.zeros_like(w), where=lik > 0)


def _self_consistency(a, w, p, tol, max_iter):
    converged = False
    iterations = 0
    for iterations in range(1, max_iter + 1):
        p_new = p * (a.T @ _ratio(w, a @ p))
        p_new /= p_new.sum()
        delta = float(np.max(np.abs(p_new - p)))
        p = p_new
        if delta < tol:
            converged = True
            break
    return p, iterations, converged


def _weighted_loglik(a, w, p):
    lik = a @ p
    positive = w > 0
    return float(np.sum(w[positive] * np.log(lik[positive])))


def interval_censor(durations, inspection_times, events=None):
    """Turn exact failure times into ``(left, right]`` intervals from an inspection schedule.

    A failure is only noticed at the first inspection at or after it, so it
    becomes ``(previous inspection, that inspection]``; failures after the last
    inspection (or censored subjects, when ``events`` is given) become
    right-censored at the last inspection they were seen event-free.

    ``inspection_times`` may be one shared schedule (1-D) or one schedule per
    subject (a sequence of 1-D arrays).
    """
    durations = np.asarray(durations, dtype=float)
    if durations.ndim != 1 or durations.size == 0:
        raise ValueError("durations must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(durations)) or np.any(durations <= 0):
        raise ValueError("durations must be finite and positive")
    n = durations.shape[0]
    if events is None:
        events = np.ones(n, dtype=bool)
    else:
        events = np.asarray(events).astype(bool)
        if events.shape != (n,):
            raise ValueError("events must have one entry per duration")

    if len(inspection_times) and np.ndim(inspection_times[0]) == 0:
        shared = np.asarray(inspection_times, dtype=float)
        schedules = [shared] * n
    else:
        schedules = list(inspection_times)
    if len(schedules) != n:
        raise ValueError("need one inspection schedule per subject")

    left = np.zeros(n, dtype=float)
    right = np.full(n, np.inf, dtype=float)
    for i, schedule in enumerate(schedules):
        grid = np.sort(np.asarray(schedule, dtype=float))
        if grid.size == 0 or np.any(grid <= 0) or not np.all(np.isfinite(grid)):
            raise ValueError("inspection times must be finite and positive")
        t = durations[i]
        if events[i]:
            after = grid[grid >= t]
            before = grid[grid < t]
            left[i] = float(before[-1]) if before.size else 0.0
            right[i] = float(after[0]) if after.size else np.inf
        else:
            seen = grid[grid <= t]
            left[i] = float(seen[-1]) if seen.size else 0.0
            right[i] = np.inf
    # A subject never inspected before right-censoring carries no information
    # beyond survival past 0; keep it as (0, inf].
    return left, right
