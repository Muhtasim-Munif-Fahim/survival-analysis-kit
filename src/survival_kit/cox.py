"""Cox proportional hazards estimator via the Breslow partial likelihood."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import chi2, norm

from .utils import step_value, validate_durations_events


@dataclass
class CoxPHResult:
    """Fitted Cox proportional hazards model for right-censored data."""

    coefficients: np.ndarray
    hazard_ratios: np.ndarray
    std_err: np.ndarray
    ci_lower: np.ndarray
    ci_upper: np.ndarray
    log_partial_likelihood: float
    log_partial_likelihood_null: float
    likelihood_ratio_statistic: float
    likelihood_ratio_p_value: float
    n_observations: int
    n_events: int
    n_iterations: int
    converged: bool
    feature_names: tuple
    baseline_time: np.ndarray
    baseline_cumhazard: np.ndarray

    @property
    def z_scores(self):
        return self.coefficients / self.std_err

    @property
    def p_values(self):
        return 2.0 * norm.sf(np.abs(self.z_scores))

    @property
    def hazard_ratio_ci_lower(self):
        return np.exp(self.ci_lower)

    @property
    def hazard_ratio_ci_upper(self):
        return np.exp(self.ci_upper)

    def linear_predictor(self, covariates):
        """Risk scores ``X @ beta``; larger values predict earlier events."""
        X = as_covariate_matrix(covariates, n_features=self.coefficients.size)
        return X @ self.coefficients

    def relative_hazard(self, covariates):
        """Hazard ratio ``exp(X @ beta)`` relative to a zero covariate vector."""
        return np.exp(self.linear_predictor(covariates))

    def baseline_at(self, times):
        """Breslow baseline cumulative hazard just after each time in ``times``."""
        return step_value(self.baseline_time, self.baseline_cumhazard, times, outside=0.0)

    def cumulative_hazard_at(self, times, covariates):
        """Cumulative hazard ``H0(t) * exp(X @ beta)`` at ``times``.

        ``covariates`` is a single observation (length ``p`` or shape ``(1, p)``).
        """
        X = as_covariate_matrix(covariates, n_features=self.coefficients.size)
        if X.shape[0] != 1:
            raise ValueError("cumulative_hazard_at expects a single observation")
        return self.baseline_at(times) * float(np.exp(X[0] @ self.coefficients))

    def survival_at(self, times, covariates):
        """PH survival ``exp(-H(t | X))`` at ``times`` for a single observation."""
        return np.exp(-self.cumulative_hazard_at(times, covariates))


def as_covariate_matrix(covariates, n_features, n_observations=None):
    """Coerce covariates to a finite ``(n, p)`` float matrix."""
    X = np.asarray(covariates, dtype=float)
    if n_features < 1:
        raise ValueError("at least one covariate is required")
    if X.ndim == 1:
        if n_features == 1:
            X = X.reshape(-1, 1)
        elif X.shape[0] == n_features:
            X = X.reshape(1, -1)
        else:
            raise ValueError("covariates must be shaped (n,) for one feature or (p,) for one row")
    if X.ndim != 2:
        raise ValueError("covariates must be 1- or 2-dimensional")
    if X.shape[1] != n_features:
        raise ValueError("covariates must have one column per coefficient")
    if n_observations is not None and X.shape[0] != n_observations:
        raise ValueError("covariates must have one row per observation")
    if X.shape[0] == 0:
        raise ValueError("at least one observation is required")
    if not np.all(np.isfinite(X)):
        raise ValueError("covariates must contain only finite values")
    return X


def breslow_partial_likelihood(beta, durations, events, covariates):
    """Breslow partial log-likelihood, score, information, and baseline jumps.

    The information matrix is ``-H``, the observed information of the partial
    log-likelihood. Unique event times with no deaths are skipped; tied deaths
    at a time ``t`` contribute ``s(t) @ beta - d(t) * log(sum_{R(t)} exp(x @ beta))``.
    """
    beta = np.asarray(beta, dtype=float).reshape(-1)
    X = np.asarray(covariates, dtype=float)
    p = beta.size
    eta = X @ beta
    if not np.all(np.isfinite(eta)):
        inf_score = np.full(p, np.nan)
        inf_info = np.full((p, p), np.nan)
        return -np.inf, inf_score, inf_info, np.array([]), np.array([])

    loglik = 0.0
    score = np.zeros(p)
    information = np.zeros((p, p))
    event_times = []
    increments = []

    for t in np.unique(durations[events]):
        at_risk = durations >= t
        died = at_risk & (durations == t) & events
        d_k = int(np.count_nonzero(died))
        if d_k == 0:
            continue
        s_k = X[died].sum(axis=0)
        eta_risk = eta[at_risk]
        offset = float(np.max(eta_risk))
        weights = np.exp(eta_risk - offset)
        X_risk = X[at_risk]
        s0 = float(weights.sum())
        if not np.isfinite(s0) or s0 <= 0.0:
            inf_score = np.full(p, np.nan)
            inf_info = np.full((p, p), np.nan)
            return -np.inf, inf_score, inf_info, np.array([]), np.array([])
        s1 = weights @ X_risk
        s2 = (X_risk * weights[:, None]).T @ X_risk
        mean = s1 / s0
        loglik += float(s_k @ beta - d_k * (np.log(s0) + offset))
        score += s_k - d_k * mean
        information += d_k * (s2 / s0 - np.outer(mean, mean))
        event_times.append(float(t))
        increments.append(d_k / (s0 * np.exp(offset)))

    return (
        float(loglik),
        score,
        information,
        np.asarray(event_times, dtype=float),
        np.asarray(increments, dtype=float),
    )


def fit_cox_ph(
    durations,
    events,
    covariates,
    *,
    feature_names=None,
    confidence_level=0.95,
    max_iter=100,
    tol=1e-8,
):
    """Maximize the Breslow partial likelihood of a Cox PH model.

    ``covariates`` is an ``(n, p)`` matrix, or a length-``n`` vector for a
    single covariate. There is no intercept: a constant shift is absorbed into
    the unspecified baseline hazard. Hazard ratios are ``exp(beta)`` for a
    one-unit increase in the corresponding column.
    """
    durations, events = validate_durations_events(durations, events)
    if not np.any(events):
        raise ValueError("at least one event is required")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie strictly between 0 and 1")
    if max_iter < 1:
        raise ValueError("max_iter must be a positive integer")
    if tol <= 0:
        raise ValueError("tol must be positive")

    X_probe = np.asarray(covariates, dtype=float)
    n_features = 1 if X_probe.ndim == 1 else X_probe.shape[1]
    X = as_covariate_matrix(covariates, n_features=n_features, n_observations=durations.size)
    p = X.shape[1]
    if np.any(np.ptp(X, axis=0) == 0.0):
        raise ValueError("each covariate must vary across observations")

    if feature_names is None:
        names = tuple(f"x{i}" for i in range(p))
    else:
        names = tuple(feature_names)
        if len(names) != p:
            raise ValueError("feature_names must match the number of covariates")

    beta = np.zeros(p)
    loglik, score, information, event_times, increments = breslow_partial_likelihood(
        beta, durations, events, X
    )
    loglik_null = float(loglik)
    converged = np.max(np.abs(score)) < tol
    iteration = 0

    while not converged and iteration < max_iter:
        if (
            not np.all(np.isfinite(information))
            or np.linalg.matrix_rank(information, tol=1e-10) < p
        ):
            raise ValueError("observed information is singular; check for collinear covariates")
        try:
            delta = np.linalg.solve(information, score)
        except np.linalg.LinAlgError as exc:
            raise ValueError("observed information is singular; check for collinear covariates") from exc
        if not np.all(np.isfinite(delta)):
            raise ValueError("Newton step is not finite")

        step = 1.0
        accepted = False
        trial_state = None
        while step >= 1e-12:
            trial = beta + step * delta
            trial_ll, trial_score, trial_info, trial_times, trial_inc = breslow_partial_likelihood(
                trial, durations, events, X
            )
            if np.isfinite(trial_ll) and trial_ll + 1e-12 >= loglik:
                trial_state = (trial, trial_ll, trial_score, trial_info, trial_times, trial_inc)
                accepted = True
                break
            step *= 0.5
        if not accepted:
            raise ValueError("line search failed to increase the partial likelihood")

        beta, loglik, score, information, event_times, increments = trial_state
        iteration += 1
        if np.max(np.abs(delta)) * step < tol:
            converged = True

    if not converged:
        raise ValueError(f"Newton-Raphson failed to converge in {max_iter} iterations")

    try:
        covariance = np.linalg.inv(information)
    except np.linalg.LinAlgError as exc:
        raise ValueError("observed information is singular; check for collinear covariates") from exc
    if not np.all(np.isfinite(covariance)) or np.any(np.diag(covariance) <= 0.0):
        raise ValueError("observed information is singular; check for collinear covariates")

    std_err = np.sqrt(np.diag(covariance))
    z = float(norm.ppf(0.5 + confidence_level / 2.0))
    lr = max(2.0 * (loglik - loglik_null), 0.0)

    return CoxPHResult(
        coefficients=beta,
        hazard_ratios=np.exp(beta),
        std_err=std_err,
        ci_lower=beta - z * std_err,
        ci_upper=beta + z * std_err,
        log_partial_likelihood=float(loglik),
        log_partial_likelihood_null=loglik_null,
        likelihood_ratio_statistic=float(lr),
        likelihood_ratio_p_value=float(chi2.sf(lr, p)),
        n_observations=int(durations.size),
        n_events=int(np.count_nonzero(events)),
        n_iterations=iteration,
        converged=True,
        feature_names=names,
        baseline_time=event_times,
        baseline_cumhazard=np.cumsum(increments),
    )
