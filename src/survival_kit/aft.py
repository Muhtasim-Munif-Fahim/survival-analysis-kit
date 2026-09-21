"""Weibull accelerated failure time estimator for right-censored data."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import chi2, norm

from .cox import as_covariate_matrix
from .utils import validate_durations_events

_EULER_GAMMA = 0.5772156649015329
_MAX_Z = 60.0


@dataclass
class WeibullAFTResult:
    """Fitted Weibull accelerated failure time model.

    Times follow ``log T = intercept + X @ beta + sigma * W`` with ``W`` a
    standard Gumbel (minimum extreme value) residual. The Weibull shape is
    ``1 / sigma`` and the scale at a covariate row ``x`` is
    ``exp(intercept + x @ beta)``. Positive ``beta`` lengthens survival
    (time ratio ``exp(beta)``); the proportional-hazards log hazard ratio is
    ``-beta / sigma``.
    """

    intercept: float
    coefficients: np.ndarray
    log_sigma: float
    sigma: float
    shape: float
    std_err_intercept: float
    std_err: np.ndarray
    std_err_log_sigma: float
    std_err_sigma: float
    std_err_shape: float
    ci_lower_intercept: float
    ci_upper_intercept: float
    ci_lower: np.ndarray
    ci_upper: np.ndarray
    ci_lower_log_sigma: float
    ci_upper_log_sigma: float
    acceleration_factors: np.ndarray
    acceleration_factor_ci_lower: np.ndarray
    acceleration_factor_ci_upper: np.ndarray
    log_likelihood: float
    log_likelihood_null: float
    likelihood_ratio_statistic: float
    likelihood_ratio_p_value: float
    n_observations: int
    n_events: int
    n_iterations: int
    converged: bool
    feature_names: tuple

    @property
    def z_scores(self):
        return self.coefficients / self.std_err

    @property
    def p_values(self):
        return 2.0 * norm.sf(np.abs(self.z_scores))

    @property
    def intercept_z(self):
        return self.intercept / self.std_err_intercept

    @property
    def intercept_p_value(self):
        return 2.0 * norm.sf(np.abs(self.intercept_z))

    @property
    def log_sigma_z(self):
        return self.log_sigma / self.std_err_log_sigma

    @property
    def log_sigma_p_value(self):
        return 2.0 * norm.sf(np.abs(self.log_sigma_z))

    @property
    def ph_coefficients(self):
        """Log hazard ratios ``-beta / sigma`` of the equivalent Weibull PH model."""
        return -self.coefficients / self.sigma

    @property
    def scale(self):
        """Weibull scale at a zero covariate vector, ``exp(intercept)``."""
        return float(np.exp(self.intercept))

    def linear_predictor(self, covariates=None):
        """Location ``intercept + X @ beta`` on the log-time scale."""
        if self.coefficients.size == 0:
            if covariates is not None:
                raise ValueError("this fit has no covariates")
            return np.array([self.intercept], dtype=float)
        X = as_covariate_matrix(covariates, n_features=self.coefficients.size)
        return self.intercept + X @ self.coefficients

    def time_ratio(self, covariates):
        """Acceleration factor ``exp(X @ beta)`` relative to a zero covariate vector."""
        if self.coefficients.size == 0:
            raise ValueError("this fit has no covariates")
        X = as_covariate_matrix(covariates, n_features=self.coefficients.size)
        return np.exp(X @ self.coefficients)

    def survival_at(self, times, covariates=None):
        """Weibull survival ``exp(-exp((log t - eta) / sigma))`` at ``times``.

        ``covariates`` is a single observation (length ``p`` or shape ``(1, p)``)
        when the fit includes covariates; omit it for an intercept-only fit.
        """
        eta = self._eta_one(covariates)
        times = np.asarray(times, dtype=float)
        if np.any(times < 0.0):
            raise ValueError("times must be non-negative")
        scalar = times.ndim == 0
        flat = np.atleast_1d(times)
        result = np.ones(flat.shape, dtype=float)
        positive = flat > 0.0
        z = (np.log(flat[positive]) - eta) / self.sigma
        result[positive] = np.exp(-np.exp(z))
        return float(result[0]) if scalar else result

    def cumulative_hazard_at(self, times, covariates=None):
        """Cumulative hazard ``-log S(t | X)`` at ``times`` for a single observation."""
        survival = np.asarray(self.survival_at(times, covariates), dtype=float)
        return -np.log(np.clip(survival, 1e-300, 1.0))

    def quantile(self, probs, covariates=None):
        """Failure-time quantiles ``exp(eta + sigma * log(-log(1 - p)))``."""
        eta = self._eta_one(covariates)
        p = np.asarray(probs, dtype=float)
        if np.any((p <= 0.0) | (p >= 1.0)):
            raise ValueError("probs must lie strictly between 0 and 1")
        values = np.exp(eta + self.sigma * np.log(-np.log(1.0 - p)))
        return float(values) if np.asarray(probs).ndim == 0 else values

    def median(self, covariates=None):
        """Median survival time for a single observation."""
        return self.quantile(0.5, covariates)

    def _eta_one(self, covariates):
        if self.coefficients.size == 0:
            if covariates is not None:
                raise ValueError("this fit has no covariates")
            return float(self.intercept)
        X = as_covariate_matrix(covariates, n_features=self.coefficients.size)
        if X.shape[0] != 1:
            raise ValueError("survival_at expects a single observation")
        return float(self.intercept + X[0] @ self.coefficients)


def weibull_aft_log_likelihood(params, log_times, events, covariates):
    """Right-censored Weibull AFT log-likelihood, score, and observed information.

    ``params`` is ``[intercept, coefficients..., log_sigma]``. Residuals are
    Gumbel (minimum) on the log-time scale. The information matrix is ``-H``.
    """
    params = np.asarray(params, dtype=float).reshape(-1)
    X = np.asarray(covariates, dtype=float)
    log_times = np.asarray(log_times, dtype=float)
    events_f = np.asarray(events, dtype=float)
    n, p = X.shape
    n_params = p + 2
    nan_score = np.full(n_params, np.nan)
    nan_info = np.full((n_params, n_params), np.nan)
    if params.size != n_params:
        raise ValueError("params must be [intercept, coefficients..., log_sigma]")

    mu = float(params[0])
    beta = params[1 : 1 + p]
    nu = float(params[-1])
    sigma = float(np.exp(nu))
    if (
        not np.isfinite(sigma)
        or sigma <= 0.0
        or not np.isfinite(mu)
        or not np.isfinite(nu)
        or not np.all(np.isfinite(beta))
    ):
        return -np.inf, nan_score, nan_info

    eta = mu + (X @ beta if p else 0.0)
    z = (log_times - eta) / sigma
    if not np.all(np.isfinite(z)) or np.max(z) > _MAX_Z:
        return -np.inf, nan_score, nan_info
    u = np.exp(z)
    if not np.all(np.isfinite(u)):
        return -np.inf, nan_score, nan_info

    loglik = float(np.sum(-u + events_f * (-nu - log_times + z)))
    if not np.isfinite(loglik):
        return -np.inf, nan_score, nan_info

    inv_sigma = 1.0 / sigma
    g_mu = np.full(n, -inv_sigma)
    g_nu = -z
    if p:
        G = np.column_stack((g_mu, -X * inv_sigma, g_nu))
    else:
        G = np.column_stack((g_mu, g_nu))

    residual = events_f - u
    score = G.T @ residual
    score[-1] -= float(events_f.sum())

    hessian = -(G * u[:, None]).T @ G
    extra_nu = residual * inv_sigma
    hessian[0, -1] += extra_nu.sum()
    hessian[-1, 0] += extra_nu.sum()
    if p:
        extra_beta_nu = X.T @ extra_nu
        hessian[1 : 1 + p, -1] += extra_beta_nu
        hessian[-1, 1 : 1 + p] += extra_beta_nu
    hessian[-1, -1] += float(np.dot(residual, z))
    information = -hessian
    return loglik, score, information


def _initial_params(log_times, events, covariates):
    """Method-of-moments / OLS start on the log-time scale."""
    n, p = covariates.shape
    enough = np.count_nonzero(events) >= max(p + 2, 2)
    mask = events if enough else np.ones(n, dtype=bool)
    y = log_times[mask]
    if p:
        design = np.column_stack((np.ones(y.size), covariates[mask]))
        coef, *_ = np.linalg.lstsq(design, y, rcond=None)
        resid = y - design @ coef
    else:
        coef = np.array([float(np.mean(y))], dtype=float)
        resid = y - coef[0]
    if resid.size > 1:
        var = float(np.var(resid, ddof=1))
    else:
        var = 1.0
    sigma = float(np.sqrt(max(6.0 * var, 1e-8) / np.pi**2))
    sigma = min(max(sigma, 0.05), 10.0)
    theta = np.zeros(p + 2)
    theta[0] = float(coef[0]) + _EULER_GAMMA * sigma
    if p:
        theta[1 : 1 + p] = coef[1:]
    theta[-1] = np.log(sigma)
    return theta


def _newton_delta(information, score):
    """Ascent-constrained Newton step; ridge until the direction increases ell."""
    n_params = score.size
    eye = np.eye(n_params)
    ridge = 0.0
    for _ in range(24):
        matrix = information if ridge == 0.0 else information + ridge * eye
        try:
            if ridge == 0.0:
                evals = np.linalg.eigvalsh(matrix)
                if not np.all(np.isfinite(evals)) or evals.min() <= 1e-10:
                    ridge = max(1e-3, 1e-3 - float(np.nanmin(evals)))
                    continue
            delta = np.linalg.solve(matrix, score)
        except np.linalg.LinAlgError:
            ridge = 1e-3 if ridge == 0.0 else ridge * 10.0
            continue
        if np.all(np.isfinite(delta)) and float(np.dot(score, delta)) > 0.0:
            max_abs = float(np.max(np.abs(delta)))
            if max_abs > 2.0:
                delta = delta * (2.0 / max_abs)
            return delta
        ridge = 1e-3 if ridge == 0.0 else ridge * 10.0
        if ridge > 1e12:
            break
    if not np.any(np.isfinite(score)) or np.max(np.abs(score)) == 0.0:
        raise ValueError("observed information is singular; check for collinear covariates")
    return score.copy()


def _maximize(log_times, events, covariates, max_iter, tol):
    theta = _initial_params(log_times, events, covariates)
    loglik, score, information = weibull_aft_log_likelihood(
        theta, log_times, events, covariates
    )
    if not np.isfinite(loglik):
        raise ValueError("Weibull AFT log-likelihood is not finite at the start")
    converged = np.all(np.isfinite(score)) and np.max(np.abs(score)) < tol
    iteration = 0

    while not converged and iteration < max_iter:
        if not np.all(np.isfinite(information)):
            raise ValueError("observed information is singular; check for collinear covariates")
        delta = _newton_delta(information, score)
        if not np.all(np.isfinite(delta)):
            raise ValueError("Newton step is not finite")

        step = 1.0
        accepted = False
        trial_state = None
        while step >= 1e-12:
            trial = theta + step * delta
            trial_ll, trial_score, trial_info = weibull_aft_log_likelihood(
                trial, log_times, events, covariates
            )
            if np.isfinite(trial_ll) and trial_ll + 1e-12 >= loglik:
                trial_state = (trial, trial_ll, trial_score, trial_info)
                accepted = True
                break
            step *= 0.5
        if not accepted:
            raise ValueError("line search failed to increase the log-likelihood")

        theta, loglik, score, information = trial_state
        iteration += 1
        if np.max(np.abs(delta)) * step < tol:
            converged = True

    if not converged:
        raise ValueError(f"Newton-Raphson failed to converge in {max_iter} iterations")
    if not np.all(np.isfinite(score)) or np.max(np.abs(score)) > 1e-3:
        raise ValueError(f"Newton-Raphson failed to converge in {max_iter} iterations")

    return theta, float(loglik), score, information, iteration


def _as_design(covariates, n_observations):
    if covariates is None:
        return np.zeros((n_observations, 0), dtype=float), ()
    X_probe = np.asarray(covariates, dtype=float)
    if X_probe.ndim == 2 and X_probe.shape[1] == 0:
        if X_probe.shape[0] not in (0, n_observations):
            raise ValueError("covariates must have one row per observation")
        return np.zeros((n_observations, 0), dtype=float), ()
    n_features = 1 if X_probe.ndim == 1 else X_probe.shape[1]
    X = as_covariate_matrix(
        covariates, n_features=n_features, n_observations=n_observations
    )
    return X, tuple(f"x{i}" for i in range(X.shape[1]))


def fit_weibull_aft(
    durations,
    events,
    covariates=None,
    *,
    feature_names=None,
    confidence_level=0.95,
    max_iter=100,
    tol=1e-8,
):
    """Maximize the right-censored Weibull AFT log-likelihood.

    ``covariates`` is optional. When given it is an ``(n, p)`` matrix, or a
    length-``n`` vector for a single covariate. An intercept is always
    estimated; a constant column is collinear with it and rejected. Time
    ratios are ``exp(beta)`` for a one-unit increase in the corresponding
    column (an acceleration factor greater than one means longer survival).
    """
    durations, events = validate_durations_events(durations, events)
    if np.any(durations <= 0.0):
        raise ValueError("durations must be strictly positive for a Weibull AFT model")
    if not np.any(events):
        raise ValueError("at least one event is required")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie strictly between 0 and 1")
    if max_iter < 1:
        raise ValueError("max_iter must be a positive integer")
    if tol <= 0:
        raise ValueError("tol must be positive")

    X, default_names = _as_design(covariates, durations.size)
    p = X.shape[1]
    if p and np.any(np.ptp(X, axis=0) == 0.0):
        raise ValueError("each covariate must vary across observations")

    if feature_names is None:
        names = default_names
    else:
        names = tuple(feature_names)
        if len(names) != p:
            raise ValueError("feature_names must match the number of covariates")

    log_times = np.log(durations)
    theta, loglik, _score, information, iteration = _maximize(
        log_times, events, X, max_iter, tol
    )
    try:
        covariance = np.linalg.inv(information)
    except np.linalg.LinAlgError as exc:
        raise ValueError("observed information is singular; check for collinear covariates") from exc
    if not np.all(np.isfinite(covariance)) or np.any(np.diag(covariance) <= 0.0):
        raise ValueError("observed information is singular; check for collinear covariates")

    if p:
        null_X = np.zeros((durations.size, 0), dtype=float)
        _, loglik_null, _, _, _ = _maximize(log_times, events, null_X, max_iter, tol)
    else:
        loglik_null = loglik

    std_err_all = np.sqrt(np.diag(covariance))
    z = float(norm.ppf(0.5 + confidence_level / 2.0))
    intercept = float(theta[0])
    beta = np.asarray(theta[1 : 1 + p], dtype=float)
    log_sigma = float(theta[-1])
    sigma = float(np.exp(log_sigma))
    shape = 1.0 / sigma
    se_intercept = float(std_err_all[0])
    se_beta = np.asarray(std_err_all[1 : 1 + p], dtype=float)
    se_log_sigma = float(std_err_all[-1])
    se_sigma = sigma * se_log_sigma
    se_shape = shape * se_log_sigma
    lr = max(2.0 * (loglik - loglik_null), 0.0)
    lr_df = max(p, 0)
    lr_p = 1.0 if lr_df == 0 else float(chi2.sf(lr, lr_df))

    return WeibullAFTResult(
        intercept=intercept,
        coefficients=beta,
        log_sigma=log_sigma,
        sigma=sigma,
        shape=shape,
        std_err_intercept=se_intercept,
        std_err=se_beta,
        std_err_log_sigma=se_log_sigma,
        std_err_sigma=se_sigma,
        std_err_shape=se_shape,
        ci_lower_intercept=intercept - z * se_intercept,
        ci_upper_intercept=intercept + z * se_intercept,
        ci_lower=beta - z * se_beta,
        ci_upper=beta + z * se_beta,
        ci_lower_log_sigma=log_sigma - z * se_log_sigma,
        ci_upper_log_sigma=log_sigma + z * se_log_sigma,
        acceleration_factors=np.exp(beta),
        acceleration_factor_ci_lower=np.exp(beta - z * se_beta),
        acceleration_factor_ci_upper=np.exp(beta + z * se_beta),
        log_likelihood=float(loglik),
        log_likelihood_null=float(loglik_null),
        likelihood_ratio_statistic=float(lr),
        likelihood_ratio_p_value=lr_p,
        n_observations=int(durations.size),
        n_events=int(np.count_nonzero(events)),
        n_iterations=iteration,
        converged=True,
        feature_names=names,
    )
