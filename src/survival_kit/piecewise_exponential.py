"""Piecewise exponential proportional-hazards model.

The baseline hazard is constant on a set of time intervals
``[c_0 = 0, c_1), [c_1, c_2), ..., [c_{K-1}, inf)``:

    h(t | x) = lambda_k * exp(x @ beta),   t in [c_{k-1}, c_k).

It sits between the fully parametric exponential model (``K = 1``) and the
semi-parametric Cox model: with many intervals the coefficients approach the
Cox estimates, while the baseline hazard, survival curves and medians stay
available in closed form.

The likelihood equals that of a Poisson GLM on the *person-period* data set
(one row per subject and interval reached, with exposure ``E_ik`` and event
``d_ik``). See :func:`piecewise_exponential_data`. We fit it directly:
for fixed ``beta`` the interval rates have the closed form
``lambda_k = D_k / sum_i E_ik exp(x_i beta)``, and the profile
log-likelihood

    l_p(beta) = sum_i d_i x_i beta - sum_k D_k log S_k(beta) + const,
    S_k(beta) = sum_i E_ik exp(x_i beta),

is concave. It is maximized by Newton-Raphson with step halving, using an
analytic gradient and Hessian. Standard errors come from the full observed
information of ``(log lambda, beta)``. Intervals without events have
``lambda_k = 0`` and a ``nan`` standard error.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import chi2, norm

from .cox import as_covariate_matrix
from .utils import validate_durations_events

__all__ = [
    "PiecewiseExponentialResult",
    "fit_piecewise_exponential",
    "piecewise_exponential_data",
]


def _resolve_breakpoints(breakpoints, t: np.ndarray, e: np.ndarray) -> np.ndarray:
    if breakpoints is None:
        breakpoints = 5
    if isinstance(breakpoints, (int, np.integer)) and not isinstance(breakpoints, bool):
        k = int(breakpoints)
        if k < 1:
            raise ValueError("number of intervals must be >= 1")
        event_times = t[e]
        if k == 1 or event_times.size == 0:
            return np.empty(0)
        cuts = np.quantile(event_times, np.arange(1, k) / k)
        cuts = np.unique(cuts[cuts > 0])
        return cuts
    cuts = np.asarray(breakpoints, dtype=float).ravel()
    if cuts.size and (not np.all(np.isfinite(cuts)) or np.any(cuts <= 0)):
        raise ValueError("breakpoints must be finite and positive")
    if np.any(np.diff(cuts) <= 0):
        raise ValueError("breakpoints must be strictly increasing")
    return cuts


def _exposure(t: np.ndarray, cuts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Exposure matrix ``E[i, k]`` and the interval index of each time."""
    lower = np.concatenate([[0.0], cuts])
    upper = np.concatenate([cuts, [np.inf]])
    E = np.clip(np.minimum(t[:, None], upper[None, :]) - lower[None, :], 0.0, None)
    idx = np.searchsorted(cuts, t, side="right")
    return E, idx


def piecewise_exponential_data(durations, events, covariates=None, breakpoints=5):
    """Expand subjects into person-period (Poisson) rows.

    Returns a dict of equal-length arrays: ``id``, ``interval``, ``start``,
    ``stop``, ``exposure``, ``event`` and, if covariates are given, a
    ``covariates`` matrix with the subject's row repeated. A Poisson GLM of
    ``event`` on interval dummies plus covariates with ``log(exposure)``
    offset reproduces :func:`fit_piecewise_exponential`.
    """
    t, e = validate_durations_events(durations, events)
    cuts = _resolve_breakpoints(breakpoints, t, e)
    E, idx = _exposure(t, cuts)
    lower = np.concatenate([[0.0], cuts])
    rows_i, rows_k = np.nonzero((E > 0) | (np.arange(E.shape[1])[None, :] == idx[:, None]))
    out = {
        "id": rows_i,
        "interval": rows_k,
        "start": lower[rows_k],
        "stop": lower[rows_k] + E[rows_i, rows_k],
        "exposure": E[rows_i, rows_k],
        "event": (e[rows_i] & (idx[rows_i] == rows_k)).astype(int),
        "breakpoints": cuts,
    }
    if covariates is not None:
        raw = np.asarray(covariates, dtype=float)
        p = 1 if raw.ndim == 1 else raw.shape[1]
        out["covariates"] = as_covariate_matrix(raw, p, t.size)[rows_i]
    return out


@dataclass
class PiecewiseExponentialResult:
    """Fitted piecewise exponential PH model."""

    breakpoints: np.ndarray
    baseline_hazards: np.ndarray
    std_err_log_hazards: np.ndarray
    events_per_interval: np.ndarray
    exposure_per_interval: np.ndarray
    coefficients: np.ndarray
    std_err: np.ndarray
    hazard_ratios: np.ndarray
    hazard_ratio_ci_lower: np.ndarray
    hazard_ratio_ci_upper: np.ndarray
    covariance: np.ndarray
    log_likelihood: float
    log_likelihood_null: float
    lr_statistic: float
    lr_p_value: float
    n_observations: int
    n_events: int
    n_iterations: int
    converged: bool
    feature_names: tuple

    @property
    def n_intervals(self) -> int:
        return int(self.baseline_hazards.size)

    @property
    def intervals(self) -> list[tuple[float, float]]:
        lower = np.concatenate([[0.0], self.breakpoints])
        upper = np.concatenate([self.breakpoints, [np.inf]])
        return [(float(a), float(b)) for a, b in zip(lower, upper)]

    @property
    def aic(self) -> float:
        n_params = int(np.sum(self.events_per_interval > 0)) + self.coefficients.size
        return 2.0 * n_params - 2.0 * self.log_likelihood

    @property
    def z_scores(self) -> np.ndarray:
        return self.coefficients / self.std_err

    @property
    def p_values(self) -> np.ndarray:
        return 2.0 * norm.sf(np.abs(self.z_scores))

    def _risk(self, covariates) -> np.ndarray:
        if self.coefficients.size == 0:
            return np.ones(1)
        if covariates is None:
            raise ValueError("covariates are required for a model fitted with covariates")
        X = as_covariate_matrix(covariates, self.coefficients.size)
        return np.exp(X @ self.coefficients)

    def baseline_cumulative_hazard(self, times) -> np.ndarray:
        t = np.atleast_1d(np.asarray(times, dtype=float))
        E, _ = _exposure(t, self.breakpoints)
        return E @ self.baseline_hazards

    def cumulative_hazard(self, times, covariates=None) -> np.ndarray:
        """``H(t | x)``, shape ``(n_rows, n_times)``. There is one row if the model has no covariates."""
        return self._risk(covariates)[:, None] * self.baseline_cumulative_hazard(times)[None, :]

    def survival_function(self, times, covariates=None) -> np.ndarray:
        return np.exp(-self.cumulative_hazard(times, covariates))

    def hazard(self, times, covariates=None) -> np.ndarray:
        t = np.atleast_1d(np.asarray(times, dtype=float))
        base = self.baseline_hazards[np.searchsorted(self.breakpoints, t, side="right")]
        return self._risk(covariates)[:, None] * base[None, :]

    def quantile_survival_time(self, p: float = 0.5, covariates=None) -> np.ndarray:
        """Time at which survival falls to ``p``, found by inverting the piecewise-linear ``H``."""
        if not 0.0 < p < 1.0:
            raise ValueError("p must lie in (0, 1)")
        target = -np.log(p) / self._risk(covariates)
        lower = np.concatenate([[0.0], self.breakpoints])
        widths = np.diff(self.breakpoints, prepend=0.0)
        H_upper = np.cumsum(self.baseline_hazards[:-1] * widths)  # H at each finite cut
        H_lower = np.concatenate([[0.0], H_upper])
        out = np.full(target.shape, np.inf)
        for r, h in enumerate(target):
            # first interval whose end reaches h (last interval extends to inf)
            k = int(np.searchsorted(H_upper, h, side="left"))
            rate = self.baseline_hazards[k]
            if rate > 0:
                out[r] = lower[k] + (h - H_lower[k]) / rate
        return out

    def median_survival_time(self, covariates=None) -> np.ndarray:
        return self.quantile_survival_time(0.5, covariates)

    def summary(self) -> str:
        lines = [f"Piecewise exponential PH: n={self.n_observations}, events={self.n_events}, "
                 f"intervals={self.n_intervals}, logLik={self.log_likelihood:.3f}"]
        for (a, b), lam, d in zip(self.intervals, self.baseline_hazards, self.events_per_interval):
            lines.append(f"  [{a:g}, {b:g}): hazard={lam:.5g} events={int(d)}")
        for name, coef, hr, lo, hi, pv in zip(self.feature_names, self.coefficients, self.hazard_ratios,
                                              self.hazard_ratio_ci_lower, self.hazard_ratio_ci_upper, self.p_values):
            lines.append(f"  {name}: coef={coef:.4f} HR={hr:.4f} [{lo:.4f}, {hi:.4f}] p={pv:.3g}")
        return "\n".join(lines)


def _profile(beta, X, d, E, D):
    eta = X @ beta
    shift = eta.max() if eta.size else 0.0
    w = np.exp(eta - shift)
    S = E.T @ w  # (K,)
    keep = D > 0
    ll = float(d @ eta - np.sum(D[keep] * (np.log(S[keep]) + shift)))
    WX = (E[:, keep] * w[:, None]).T @ X  # (K', p)
    m = WX / S[keep, None]
    grad = X.T @ d - D[keep] @ m
    hess = np.zeros((X.shape[1], X.shape[1]))
    for j, k in enumerate(np.flatnonzero(keep)):
        wk = E[:, k] * w
        second = (X * wk[:, None]).T @ X / S[k]
        hess -= D[k] * (second - np.outer(m[j], m[j]))
    return ll, grad, hess


def fit_piecewise_exponential(
    durations,
    events,
    covariates=None,
    *,
    breakpoints=5,
    feature_names=None,
    alpha: float = 0.05,
    max_iter: int = 100,
    tol: float = 1e-9,
) -> PiecewiseExponentialResult:
    """Fit a piecewise exponential PH model by maximum likelihood.

    ``breakpoints`` is either an int ``K`` (the number of intervals, cut at
    the event-time quantiles) or an increasing sequence of positive interior
    cut points. ``covariates`` may be ``None`` for the baseline-only model.
    In that case ``lambda_k = D_k / E_k`` is the occurrence/exposure rate.
    ``lr_statistic`` is the likelihood-ratio test of ``beta = 0``.
    """
    t, e = validate_durations_events(durations, events)
    if not e.any():
        raise ValueError("at least one event is required")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    cuts = _resolve_breakpoints(breakpoints, t, e)
    E, idx = _exposure(t, cuts)
    K = E.shape[1]
    dmat = np.zeros_like(E)
    dmat[np.arange(t.size)[e], idx[e]] = 1.0
    D = dmat.sum(axis=0)
    d = e.astype(float)

    if covariates is None:
        X = np.zeros((t.size, 0))
    else:
        raw = np.asarray(covariates, dtype=float)
        p = 1 if raw.ndim == 1 else raw.shape[1]
        X = as_covariate_matrix(raw, p, t.size)
    p = X.shape[1]
    if feature_names is None:
        feature_names = tuple(f"x{j}" for j in range(p))
    else:
        feature_names = tuple(feature_names)
        if len(feature_names) != p:
            raise ValueError("feature_names must have one entry per covariate")

    keep = D > 0
    const = float(np.sum(D[keep] * (np.log(D[keep]) - 1.0)))
    beta = np.zeros(p)
    ll, grad, hess = _profile(beta, X, d, E, D)
    ll_null = ll + const
    converged = p == 0
    it = 0
    for it in range(1, max_iter + 1):
        if p == 0:
            break
        try:
            step = np.linalg.solve(-hess, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(-hess, grad, rcond=None)[0]
        scale = 1.0
        while True:
            cand = beta + scale * step
            ll_c, g_c, h_c = _profile(cand, X, d, E, D)
            if ll_c >= ll - 1e-12 or scale < 1e-10:
                break
            scale *= 0.5
        delta = ll_c - ll
        beta, ll, grad, hess = cand, ll_c, g_c, h_c
        if abs(delta) < tol * (1.0 + abs(ll)) and np.max(np.abs(scale * step)) < 1e-7:
            converged = True
            break
    if p == 0:
        it = 0

    w = np.exp(X @ beta)
    S = E.T @ w
    lam = np.where(keep, D / np.where(S > 0, S, 1.0), 0.0)
    # Full observed information for (log lambda_k [k with events], beta).
    kk = np.flatnonzero(keep)
    q = kk.size
    info = np.zeros((q + p, q + p))
    info[np.arange(q), np.arange(q)] = D[kk]
    if p:
        mu = E[:, kk] * w[:, None] * lam[kk][None, :]  # expected counts (n, q)
        info[:q, q:] = mu.T @ X
        info[q:, :q] = info[:q, q:].T
        info[q:, q:] = (X * mu.sum(axis=1)[:, None]).T @ X
    try:
        cov_full = np.linalg.inv(info)
    except np.linalg.LinAlgError:
        cov_full = np.linalg.pinv(info)
    se_log_lam = np.full(K, np.nan)
    se_log_lam[kk] = np.sqrt(np.clip(np.diag(cov_full)[:q], 0.0, None))
    cov_beta = cov_full[q:, q:]
    se = np.sqrt(np.clip(np.diag(cov_beta), 0.0, None))
    z = norm.ppf(1.0 - alpha / 2.0)
    ll_full = ll + const
    lr = max(0.0, 2.0 * (ll_full - ll_null))
    return PiecewiseExponentialResult(
        breakpoints=cuts,
        baseline_hazards=lam,
        std_err_log_hazards=se_log_lam,
        events_per_interval=D.astype(int),
        exposure_per_interval=E.sum(axis=0),
        coefficients=beta,
        std_err=se,
        hazard_ratios=np.exp(beta),
        hazard_ratio_ci_lower=np.exp(beta - z * se),
        hazard_ratio_ci_upper=np.exp(beta + z * se),
        covariance=cov_beta,
        log_likelihood=ll_full,
        log_likelihood_null=ll_null,
        lr_statistic=float(lr),
        lr_p_value=float(chi2.sf(lr, p)) if p else 1.0,
        n_observations=int(t.size),
        n_events=int(e.sum()),
        n_iterations=int(it),
        converged=bool(converged),
        feature_names=feature_names,
    )
