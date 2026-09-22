"""Fine-Gray proportional subdistribution hazards for competing risks.

Nelson-Aalen cumulative hazard estimation already lives in
:mod:`survival_kit.nelson_aalen`. This module is the regression companion to
the Aalen-Johansen CIF and Gray's test: a weighted Cox model for one cause's
subdistribution hazard.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import chi2, norm

from .competing_risks import _censoring_left_limit
from .cox import as_covariate_matrix
from .kaplan_meier import fit_kaplan_meier
from .utils import step_value, validate_event_types


@dataclass
class FineGrayResult:
    """Fitted Fine-Gray subdistribution hazard model for one cause."""

    coefficients: np.ndarray
    subdistribution_hazard_ratios: np.ndarray
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
    cause: int
    baseline_time: np.ndarray
    baseline_cumhazard: np.ndarray

    @property
    def z_scores(self):
        return self.coefficients / self.std_err

    @property
    def p_values(self):
        return 2.0 * norm.sf(np.abs(self.z_scores))

    @property
    def subdistribution_hazard_ratio_ci_lower(self):
        return np.exp(self.ci_lower)

    @property
    def subdistribution_hazard_ratio_ci_upper(self):
        return np.exp(self.ci_upper)

    def linear_predictor(self, covariates):
        """Risk scores ``X @ beta``. Larger values raise the subdistribution hazard."""
        X = as_covariate_matrix(covariates, n_features=self.coefficients.size)
        return X @ self.coefficients

    def relative_subdistribution_hazard(self, covariates):
        """Subdistribution hazard ratio ``exp(X @ beta)`` versus a zero covariate row."""
        return np.exp(self.linear_predictor(covariates))

    def baseline_at(self, times):
        """Baseline cumulative subdistribution hazard just after each time in ``times``."""
        return step_value(self.baseline_time, self.baseline_cumhazard, times, outside=0.0)

    def cumulative_subdistribution_hazard_at(self, times, covariates):
        """Cumulative subdistribution hazard ``Λ0(t) * exp(X @ beta)``.

        ``covariates`` is a single observation (length ``p`` or shape ``(1, p)``).
        """
        X = as_covariate_matrix(covariates, n_features=self.coefficients.size)
        if X.shape[0] != 1:
            raise ValueError("cumulative_subdistribution_hazard_at expects a single observation")
        return self.baseline_at(times) * float(np.exp(X[0] @ self.coefficients))

    def cumulative_incidence_at(self, times, covariates):
        """Model CIF ``1 - exp(-Λ(t | X))`` at ``times`` for a single observation."""
        return 1.0 - np.exp(-self.cumulative_subdistribution_hazard_at(times, covariates))


@dataclass
class _FineGrayTables:
    covariates: np.ndarray
    cause: int
    times: np.ndarray
    weights: np.ndarray
    died_counts: np.ndarray
    died_sum: np.ndarray


def _weight_matrix(durations, event_types, cause, times, censor_curve):
    """IPCW subdistribution weights at each cause-specific event time.

    Rows follow ``_fine_gray_weights``: weight 1 while ``duration >= t``, and
    ``G(t-) / G(T_i-)`` after a competing event. ``G`` is the Kaplan-Meier
    estimator that treats censoring as the event.
    """
    n = durations.shape[0]
    m = times.shape[0]
    weights = (durations.reshape(1, n) >= times.reshape(m, 1)).astype(float)
    competing = (event_types > 0) & (event_types != cause)
    if not np.any(competing):
        return weights
    comp = np.flatnonzero(competing)
    g_times = np.atleast_1d(_censoring_left_limit(censor_curve, times))
    g_comp = np.atleast_1d(_censoring_left_limit(censor_curve, durations[comp]))
    later = times.reshape(m, 1) > durations[comp].reshape(1, -1)
    ratio = np.zeros((m, comp.size), dtype=float)
    positive = g_comp > 0.0
    ratio[:, positive] = g_times.reshape(m, 1) / g_comp[positive]
    weights[:, comp] = np.where(later, ratio, weights[:, comp])
    return weights


def _prepare_tables(durations, event_types, covariates, cause):
    cause_event = event_types == cause
    times = np.unique(durations[cause_event])
    censor_curve = fit_kaplan_meier(durations, event_types == 0)
    if times.size == 0:
        weights = np.zeros((0, durations.shape[0]), dtype=float)
        died_counts = np.zeros(0, dtype=float)
        died_sum = np.zeros((0, covariates.shape[1]), dtype=float)
    else:
        weights = _weight_matrix(durations, event_types, cause, times, censor_curve)
        rows = np.flatnonzero(cause_event)
        index = np.searchsorted(times, durations[rows])
        died_counts = np.bincount(index, minlength=times.size).astype(float)
        died_sum = np.column_stack(
            [
                np.bincount(index, weights=covariates[rows, j], minlength=times.size)
                for j in range(covariates.shape[1])
            ]
        )
    return _FineGrayTables(
        covariates=covariates,
        cause=int(cause),
        times=times,
        weights=weights,
        died_counts=died_counts,
        died_sum=died_sum,
    )


def _empty_likelihood(n_features):
    return (
        0.0,
        np.zeros(n_features),
        np.zeros((n_features, n_features)),
        np.array([], dtype=float),
        np.array([], dtype=float),
    )


def _failed_likelihood(n_features):
    return (
        -np.inf,
        np.full(n_features, np.nan),
        np.full((n_features, n_features), np.nan),
        np.array([], dtype=float),
        np.array([], dtype=float),
    )


def _partial_from_tables(beta, tables):
    """Weighted Breslow partial likelihood treating IPCW weights as fixed."""
    beta = np.asarray(beta, dtype=float).reshape(-1)
    X = tables.covariates
    p = beta.size
    if tables.times.size == 0:
        return _empty_likelihood(p)

    eta = X @ beta
    if not np.all(np.isfinite(eta)):
        return _failed_likelihood(p)

    weights = tables.weights
    masked = np.where(weights > 0.0, eta.reshape(1, -1), -np.inf)
    offset = np.max(masked, axis=1)
    shifted = np.where(weights > 0.0, eta.reshape(1, -1) - offset.reshape(-1, 1), -np.inf)
    relative = np.exp(np.clip(shifted, -745.0, 0.0))
    weighted = weights * relative
    s0 = weighted.sum(axis=1)
    if (
        not np.all(np.isfinite(offset))
        or not np.all(np.isfinite(s0))
        or np.any(s0 <= 0.0)
    ):
        return _failed_likelihood(p)

    s1 = weighted @ X
    mean = s1 / s0.reshape(-1, 1)
    factor = tables.died_counts / s0
    subject_mass = factor @ weighted
    information = (X * subject_mass.reshape(-1, 1)).T @ X
    information -= (mean * tables.died_counts.reshape(-1, 1)).T @ mean
    information = 0.5 * (information + information.T)
    log_s0 = np.log(s0) + offset
    loglik = float(np.sum(tables.died_sum @ beta - tables.died_counts * log_s0))
    score = tables.died_sum.sum(axis=0) - (tables.died_counts.reshape(-1, 1) * mean).sum(axis=0)
    increments = np.exp(np.log(tables.died_counts) - np.log(s0) - offset)
    return (
        loglik,
        score,
        information,
        tables.times,
        increments,
    )


def fine_gray_partial_likelihood(beta, durations, event_types, covariates, cause=1):
    """Weighted Breslow partial log-likelihood of a Fine-Gray model.

    Returns ``(loglik, score, information, event_times, increments)``. The
    information matrix is ``-H``, the observed information of the weighted
    partial log-likelihood with IPCW weights held fixed. Tied cause-``cause``
    times use the Breslow denominator.
    """
    durations, event_types = validate_event_types(durations, event_types)
    cause = int(cause)
    if cause < 1:
        raise ValueError("cause must be a positive integer")
    beta = np.asarray(beta, dtype=float).reshape(-1)
    X = as_covariate_matrix(covariates, n_features=beta.size, n_observations=durations.size)
    tables = _prepare_tables(durations, event_types, X, cause)
    return _partial_from_tables(beta, tables)


def _require_cause(cause):
    cause = int(cause)
    if cause < 1:
        raise ValueError("cause must be a positive integer")
    return cause


def fit_fine_gray(
    durations,
    event_types,
    covariates,
    cause=1,
    *,
    feature_names=None,
    confidence_level=0.95,
    max_iter=100,
    tol=1e-8,
):
    """Fit the Fine-Gray proportional subdistribution hazard model for ``cause``.

    ``event_types`` use ``0`` for right-censoring and positive integers for
    mutually exclusive causes. The partial likelihood is the Breslow Cox
    likelihood on the subdistribution risk set: subjects who fail of another
    cause stay at risk with weight ``G(t-) / G(T_i-)``, where ``G`` is the
    Kaplan-Meier estimator of the censoring distribution. There is no
    intercept. ``exp(beta)`` is the per-unit subdistribution hazard ratio, and
    the model CIF is ``1 - exp(-Λ0(t) exp(x @ beta))``.

    Standard errors invert the weighted partial-likelihood information and
    treat the censoring weights as fixed. With no competing events the weights
    equal one on the ordinary risk set, and the fit matches :func:`fit_cox_ph`.
    """
    durations, event_types = validate_event_types(durations, event_types)
    cause = _require_cause(cause)
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie strictly between 0 and 1")
    if max_iter < 1:
        raise ValueError("max_iter must be a positive integer")
    if tol <= 0:
        raise ValueError("tol must be positive")
    if not np.any(event_types == cause):
        raise ValueError("at least one event of the requested cause is required")

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

    tables = _prepare_tables(durations, event_types, X, cause)
    beta = np.zeros(p)
    loglik, score, information, event_times, increments = _partial_from_tables(beta, tables)
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
            raise ValueError(
                "observed information is singular; check for collinear covariates"
            ) from exc
        if not np.all(np.isfinite(delta)):
            raise ValueError("Newton step is not finite")

        step = 1.0
        accepted = False
        trial_state = None
        while step >= 1e-12:
            trial = beta + step * delta
            trial_ll, trial_score, trial_info, trial_times, trial_inc = _partial_from_tables(
                trial, tables
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
        raise ValueError(
            "observed information is singular; check for collinear covariates"
        ) from exc
    if not np.all(np.isfinite(covariance)) or np.any(np.diag(covariance) <= 0.0):
        raise ValueError("observed information is singular; check for collinear covariates")

    std_err = np.sqrt(np.diag(covariance))
    z = float(norm.ppf(0.5 + confidence_level / 2.0))
    lr = max(2.0 * (loglik - loglik_null), 0.0)
    return FineGrayResult(
        coefficients=beta,
        subdistribution_hazard_ratios=np.exp(beta),
        std_err=std_err,
        ci_lower=beta - z * std_err,
        ci_upper=beta + z * std_err,
        log_partial_likelihood=float(loglik),
        log_partial_likelihood_null=loglik_null,
        likelihood_ratio_statistic=float(lr),
        likelihood_ratio_p_value=float(chi2.sf(lr, p)),
        n_observations=int(durations.size),
        n_events=int(np.count_nonzero(event_types == cause)),
        n_iterations=iteration,
        converged=True,
        feature_names=names,
        cause=cause,
        baseline_time=event_times,
        baseline_cumhazard=np.cumsum(increments),
    )
