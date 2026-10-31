"""Gompertz proportional-hazards regression for right-censored data.

The Gompertz hazard grows (or decays) exponentially with time,

    h(t | x) = b * exp(g * t) * exp(x @ beta),     b > 0,

which is the classical model for human/animal mortality and for failure
processes that age. ``g > 0`` is an increasing hazard, ``g = 0`` reduces to
the exponential model and ``g < 0`` gives a decreasing hazard whose
survival function plateaus at ``exp(b * exp(x @ beta) / g)`` (a *defective*
distribution, i.e. a cured fraction).

The cumulative hazard is ``H(t | x) = b * exp(x @ beta) * expm1(g t) / g``
and the log-likelihood with event indicators ``d_i`` is

    sum_i d_i (log b + g t_i + x_i beta) - sum_i H(t_i | x_i).

Parameters are estimated by maximum likelihood (L-BFGS-B with an analytic
gradient); standard errors come from the observed information matrix,
obtained by central differences of the analytic gradient.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2, norm

from .cox import as_covariate_matrix
from .utils import validate_durations_events

__all__ = [
    "GompertzResult",
    "fit_gompertz",
    "generate_gompertz_data",
    "gompertz_log_likelihood",
]

_SERIES_CUTOFF = 1e-6


def _a_and_da(g: float, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``A(t) = expm1(g t) / g`` and ``dA/dg``, stable near ``g = 0``."""
    gt = g * t
    small = np.abs(gt) < _SERIES_CUTOFF
    A = np.empty_like(t)
    dA = np.empty_like(t)
    if np.any(small):
        ts = t[small]
        A[small] = ts + 0.5 * g * ts**2
        dA[small] = 0.5 * ts**2 + g * ts**3 / 3.0
    big = ~small
    if np.any(big):
        tb = t[big]
        e = np.exp(g * tb)
        A[big] = np.expm1(g * tb) / g
        dA[big] = (tb * e - A[big]) / g
    return A, dA


def gompertz_log_likelihood(theta, durations, events, covariates=None):
    """Log-likelihood and gradient at ``theta = [log b, g, beta...]``."""
    theta = np.asarray(theta, dtype=float)
    t = np.asarray(durations, dtype=float)
    d = np.asarray(events, dtype=float)
    log_b, g = theta[0], theta[1]
    beta = theta[2:]
    if covariates is None or beta.size == 0:
        eta = np.zeros_like(t)
        X = None
    else:
        X = np.asarray(covariates, dtype=float)
        eta = X @ beta
    risk = np.exp(log_b + eta)
    A, dA = _a_and_da(g, t)
    H = risk * A
    ll = float(np.sum(d * (log_b + g * t + eta)) - np.sum(H))
    grad = np.empty_like(theta)
    grad[0] = np.sum(d) - np.sum(H)
    grad[1] = np.sum(d * t) - np.sum(risk * dA)
    if X is not None:
        grad[2:] = X.T @ (d - H)
    return ll, grad


@dataclass
class GompertzResult:
    """Fitted Gompertz proportional-hazards model."""

    log_b: float
    b: float
    g: float
    coefficients: np.ndarray
    std_err_log_b: float
    std_err_g: float
    std_err: np.ndarray
    hazard_ratios: np.ndarray
    hazard_ratio_ci_lower: np.ndarray
    hazard_ratio_ci_upper: np.ndarray
    covariance: np.ndarray
    log_likelihood: float
    log_likelihood_exponential: float
    shape_lr_statistic: float
    shape_lr_p_value: float
    n_observations: int
    n_events: int
    converged: bool
    feature_names: tuple

    @property
    def aic(self) -> float:
        return 2.0 * (2 + self.coefficients.size) - 2.0 * self.log_likelihood

    @property
    def z_scores(self) -> np.ndarray:
        return self.coefficients / self.std_err

    @property
    def p_values(self) -> np.ndarray:
        return 2.0 * norm.sf(np.abs(self.z_scores))

    def _linear(self, covariates, n_rows=None) -> np.ndarray:
        if self.coefficients.size == 0:
            return np.zeros(1 if n_rows is None else n_rows)
        if covariates is None:
            raise ValueError("covariates are required for a model fitted with covariates")
        X = as_covariate_matrix(covariates, self.coefficients.size)
        return X @ self.coefficients

    def cumulative_hazard(self, times, covariates=None) -> np.ndarray:
        """``H(t | x)``; shape ``(n_rows, n_times)`` (one row if no covariates)."""
        t = np.atleast_1d(np.asarray(times, dtype=float))
        eta = self._linear(covariates)
        A, _ = _a_and_da(self.g, t)
        return np.exp(self.log_b + eta)[:, None] * A[None, :]

    def survival_function(self, times, covariates=None) -> np.ndarray:
        return np.exp(-self.cumulative_hazard(times, covariates))

    def hazard(self, times, covariates=None) -> np.ndarray:
        t = np.atleast_1d(np.asarray(times, dtype=float))
        eta = self._linear(covariates)
        return np.exp(self.log_b + eta)[:, None] * np.exp(self.g * t)[None, :]

    def cure_fraction(self, covariates=None) -> np.ndarray:
        """Long-run survival ``S(inf | x)``; zero unless ``g < 0``."""
        eta = self._linear(covariates)
        if self.g >= 0:
            return np.zeros_like(eta)
        return np.exp(np.exp(self.log_b + eta) / self.g)

    def quantile_survival_time(self, p: float = 0.5, covariates=None) -> np.ndarray:
        """Time at which survival falls to ``p`` (``inf`` if it never does)."""
        if not 0.0 < p < 1.0:
            raise ValueError("p must lie in (0, 1)")
        eta = self._linear(covariates)
        target = -np.log(p) / np.exp(self.log_b + eta)
        if abs(self.g) < 1e-12:
            return target
        arg = 1.0 + self.g * target
        out = np.full_like(arg, np.inf)
        ok = arg > 0
        out[ok] = np.log(arg[ok]) / self.g
        return out

    def median_survival_time(self, covariates=None) -> np.ndarray:
        return self.quantile_survival_time(0.5, covariates)


def _numerical_hessian(fun_grad, theta: np.ndarray) -> np.ndarray:
    k = theta.size
    H = np.empty((k, k))
    for j in range(k):
        h = 1e-5 * max(1.0, abs(theta[j]))
        tp = theta.copy()
        tm = theta.copy()
        tp[j] += h
        tm[j] -= h
        H[:, j] = (fun_grad(tp)[1] - fun_grad(tm)[1]) / (2.0 * h)
    return 0.5 * (H + H.T)


def fit_gompertz(
    durations,
    events,
    covariates=None,
    *,
    feature_names=None,
    alpha: float = 0.05,
    max_iter: int = 500,
) -> GompertzResult:
    """Fit a Gompertz PH model by maximum likelihood.

    ``covariates`` may be ``None`` for the one-sample (no-covariate) model.
    ``shape_lr_statistic`` tests ``g = 0`` (exponential) against the Gompertz
    fit with one degree of freedom.
    """
    t, e = validate_durations_events(durations, events)
    if not e.any():
        raise ValueError("at least one event is required to fit a Gompertz model")
    if covariates is None:
        X = None
        p = 0
    else:
        raw = np.asarray(covariates, dtype=float)
        p = 1 if raw.ndim == 1 else raw.shape[1]
        X = as_covariate_matrix(raw, p, t.shape[0])
    if feature_names is None:
        feature_names = tuple(f"x{j}" for j in range(p))
    else:
        feature_names = tuple(feature_names)
        if len(feature_names) != p:
            raise ValueError("feature_names must have one entry per covariate")

    d = e.astype(float)
    scale = float(np.mean(t[t > 0])) if np.any(t > 0) else 1.0

    def neg(theta):
        ll, g = gompertz_log_likelihood(theta, t, d, X)
        return -ll, -g

    # Exponential fit (closed form when there are no covariates) as a start.
    exp_rate = d.sum() / max(t.sum(), 1e-12)
    theta0 = np.zeros(2 + p)
    theta0[0] = np.log(exp_rate)
    if p:
        res_exp = minimize(
            lambda th: neg(np.concatenate([[th[0], 0.0], th[1:]]))[0],
            np.concatenate([[theta0[0]], np.zeros(p)]),
            jac=lambda th: np.delete(neg(np.concatenate([[th[0], 0.0], th[1:]]))[1], 1),
            method="BFGS",
        )
        theta0[0] = res_exp.x[0]
        theta0[2:] = res_exp.x[1:]
        ll_exp = -res_exp.fun
    else:
        ll_exp = gompertz_log_likelihood(np.array([theta0[0], 0.0]), t, d)[0]

    res = minimize(
        lambda th: neg(th)[0],
        theta0,
        jac=lambda th: neg(th)[1],
        method="L-BFGS-B",
        options={"maxiter": max_iter, "gtol": 1e-9 / max(scale, 1.0)},
    )
    theta = res.x
    ll, grad = gompertz_log_likelihood(theta, t, d, X)
    hess = _numerical_hessian(lambda th: gompertz_log_likelihood(th, t, d, X), theta)
    try:
        cov = np.linalg.inv(-hess)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(-hess)
    se = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    beta = theta[2:]
    z = norm.ppf(1.0 - alpha / 2.0)
    lr = max(0.0, 2.0 * (ll - ll_exp))
    converged = bool(res.success) or float(np.max(np.abs(grad))) < 1e-4 * max(1.0, d.sum())
    return GompertzResult(
        log_b=float(theta[0]),
        b=float(np.exp(theta[0])),
        g=float(theta[1]),
        coefficients=beta,
        std_err_log_b=float(se[0]),
        std_err_g=float(se[1]),
        std_err=se[2:],
        hazard_ratios=np.exp(beta),
        hazard_ratio_ci_lower=np.exp(beta - z * se[2:]),
        hazard_ratio_ci_upper=np.exp(beta + z * se[2:]),
        covariance=cov,
        log_likelihood=ll,
        log_likelihood_exponential=float(ll_exp),
        shape_lr_statistic=float(lr),
        shape_lr_p_value=float(chi2.sf(lr, 1)),
        n_observations=int(t.size),
        n_events=int(e.sum()),
        converged=converged,
        feature_names=feature_names,
    )


def generate_gompertz_data(
    n: int,
    b: float = 0.05,
    g: float = 0.3,
    coefficients=None,
    *,
    censor_time: float | None = None,
    random_state=None,
):
    """Simulate ``(durations, events, covariates)`` from a Gompertz PH model.

    Covariates are standard normal (one column per coefficient). Event
    times are drawn by inverting the survival function; units that would
    never fail (possible when ``g < 0``) and those past ``censor_time`` are
    right-censored there.
    """
    if n < 1:
        raise ValueError("n must be positive")
    if b <= 0:
        raise ValueError("b must be positive")
    rng = np.random.default_rng(random_state)
    beta = np.zeros(0) if coefficients is None else np.atleast_1d(np.asarray(coefficients, float))
    X = rng.standard_normal((n, beta.size))
    risk = b * np.exp(X @ beta)
    E = rng.exponential(size=n)
    if abs(g) < 1e-12:
        T = E / risk
    else:
        arg = 1.0 + g * E / risk
        T = np.full(n, np.inf)
        ok = arg > 0
        T[ok] = np.log(arg[ok]) / g
    if censor_time is None:
        finite = T[np.isfinite(T)]
        censor_time = float(np.quantile(finite, 0.9)) if finite.size else 1.0
    events = T <= censor_time
    durations = np.minimum(T, censor_time)
    return durations, events, X
