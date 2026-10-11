"""Residual diagnostics and the Grambsch-Therneau proportional-hazards test.

Given a fitted :class:`~survival_kit.cox.CoxPHResult` and the data it was fit
on, this module computes the classical Cox model residuals

* **Schoenfeld** residuals ``r_k = x_(k) - xbar(t_k)``, one row per death,
  where ``xbar`` is the risk-weighted covariate mean at the death time
  (Breslow handling of ties, matching :func:`~survival_kit.cox.fit_cox_ph`);
* **scaled Schoenfeld** residuals ``r*_k = beta + D * V r_k`` (``D`` deaths,
  ``V`` the model covariance), whose smoothed trend over time estimates a
  time-varying coefficient ``beta_j(t)``;
* **martingale** residuals ``M_i = delta_i - H0(t_i) exp(x_i beta)``;
* **deviance** residuals ``sign(M_i) sqrt(-2 [M_i + delta_i log(delta_i - M_i)]``,

and the Grambsch-Therneau (1994) score-type test of proportional hazards:
for a transform ``g`` of time, per-covariate statistics

    T_j = (sum_k (g_k - gbar) r*_kj)^2 / (D V_jj sum_k (g_k - gbar)^2)

are compared to chi-square(1), and the global statistic

    T = (sum_k (g_k - gbar) r_k)' V (sum_k (g_k - gbar) r_k) D / sum_k (g_k - gbar)^2

to chi-square(p). This is the approximation used by R's ``survival::cox.zph``
before version 3.0 and by ``lifelines.statistics.proportional_hazard_test``.

References
----------
Schoenfeld, D. (1982). Partial residuals for the proportional hazards
regression model. *Biometrika* 69(1), 239-241.

Grambsch, P. M. and Therneau, T. M. (1994). Proportional hazards tests and
diagnostics based on weighted residuals. *Biometrika* 81(3), 515-526.

Therneau, T. M., Grambsch, P. M. and Fleming, T. R. (1990). Martingale-based
residuals for survival models. *Biometrika* 77(1), 147-160.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import chi2

from .cox import as_covariate_matrix, breslow_partial_likelihood
from .kaplan_meier import fit_kaplan_meier
from .utils import step_value, validate_durations_events

TIME_TRANSFORMS = ("km", "rank", "identity", "log")


@dataclass
class SchoenfeldResiduals:
    """Schoenfeld residuals, one row per observed death (sorted by time)."""

    times: np.ndarray
    index: np.ndarray
    residuals: np.ndarray
    scaled: np.ndarray
    feature_names: tuple


@dataclass
class ProportionalHazardsTest:
    """Grambsch-Therneau test of the proportional-hazards assumption."""

    feature_names: tuple
    transform: str
    correlations: np.ndarray
    statistics: np.ndarray
    p_values: np.ndarray
    global_statistic: float
    global_df: int
    global_p_value: float
    n_events: int

    def violations(self, alpha=0.05):
        """Names of covariates whose per-variable p-value is below ``alpha``."""
        return tuple(
            name for name, p in zip(self.feature_names, self.p_values) if p < alpha
        )

    def summary_rows(self):
        """``(name, rho, chisq, p)`` rows followed by a ``GLOBAL`` row."""
        rows = [
            (name, float(rho), float(stat), float(p))
            for name, rho, stat, p in zip(
                self.feature_names, self.correlations, self.statistics, self.p_values
            )
        ]
        rows.append(("GLOBAL", float("nan"), self.global_statistic, self.global_p_value))
        return rows


def _prepare(fit, durations, events, covariates):
    durations, events = validate_durations_events(durations, events)
    p = fit.coefficients.size
    X = as_covariate_matrix(covariates, n_features=p, n_observations=durations.size)
    if int(np.count_nonzero(events)) != fit.n_events or durations.size != fit.n_observations:
        raise ValueError("data do not match the sample the Cox model was fit on")
    return durations, events, X


def _covariance(fit, durations, events, X):
    _, _, information, _, _ = breslow_partial_likelihood(fit.coefficients, durations, events, X)
    return np.linalg.inv(information)


def schoenfeld_residuals(fit, durations, events, covariates):
    """Schoenfeld and scaled Schoenfeld residuals for a fitted Cox model.

    Returns one row per death sorted by time (ties keep input order). The
    unscaled residuals of a converged fit sum to zero column-wise (the score
    equations); the scaled residuals are ``beta + D * V r_k``.
    """
    durations, events, X = _prepare(fit, durations, events, covariates)
    beta = fit.coefficients
    eta = X @ beta
    death_idx = np.flatnonzero(events)
    order = np.lexsort((death_idx, durations[death_idx]))
    death_idx = death_idx[order]

    residuals = np.empty((death_idx.size, beta.size))
    means = {}
    for row, i in enumerate(death_idx):
        t = durations[i]
        if t not in means:
            at_risk = durations >= t
            eta_risk = eta[at_risk]
            weights = np.exp(eta_risk - np.max(eta_risk))
            means[t] = weights @ X[at_risk] / weights.sum()
        residuals[row] = X[i] - means[t]

    covariance = _covariance(fit, durations, events, X)
    scaled = beta + death_idx.size * residuals @ covariance
    return SchoenfeldResiduals(
        times=durations[death_idx],
        index=death_idx,
        residuals=residuals,
        scaled=scaled,
        feature_names=tuple(fit.feature_names),
    )


def martingale_residuals(fit, durations, events, covariates):
    """Martingale residuals ``delta_i - H0(t_i) exp(x_i beta)`` (input order)."""
    durations, events, X = _prepare(fit, durations, events, covariates)
    cumhaz = step_value(fit.baseline_time, fit.baseline_cumhazard, durations, outside=0.0)
    return events.astype(float) - cumhaz * np.exp(X @ fit.coefficients)


def deviance_residuals(fit, durations, events, covariates):
    """Deviance residuals: a symmetrized transform of the martingale residuals."""
    _, events_bool = validate_durations_events(durations, events)
    martingale = martingale_residuals(fit, durations, events, covariates)
    delta = events_bool.astype(float)
    remaining = delta - martingale
    log_term = np.zeros_like(martingale)
    positive = (delta > 0) & (remaining > 0)
    log_term[positive] = delta[positive] * np.log(remaining[positive])
    return np.sign(martingale) * np.sqrt(np.maximum(-2.0 * (martingale + log_term), 0.0))


def transform_event_times(times, durations, events, transform="km"):
    """Transform death times for the Grambsch-Therneau test.

    ``km`` is ``1 - S(t-)`` from the pooled left-continuous Kaplan-Meier curve
    (R's default), ``rank`` the rank among deaths (1..D, ties broken by order),
    ``identity`` the raw time and ``log`` its logarithm.
    """
    times = np.asarray(times, dtype=float)
    if transform == "identity":
        return times.copy()
    if transform == "log":
        if np.any(times <= 0):
            raise ValueError("log transform requires positive event times")
        return np.log(times)
    if transform == "rank":
        return np.arange(1, times.size + 1, dtype=float)
    if transform == "km":
        curve = fit_kaplan_meier(durations, events)
        knots = np.asarray(curve.time, dtype=float)
        surv = np.asarray(curve.survival, dtype=float)
        idx = np.searchsorted(knots, times, side="left") - 1
        before = np.where(idx >= 0, surv[np.clip(idx, 0, None)], 1.0)
        return 1.0 - before
    raise ValueError(f"transform must be one of {TIME_TRANSFORMS}")


def proportional_hazards_test(fit, durations, events, covariates, *, transform="km"):
    """Grambsch-Therneau test that each Cox coefficient is constant over time.

    Small p-values flag covariates whose effect drifts with (transformed)
    time; ``correlations`` give the direction (Pearson correlation between
    ``g(t)`` and the scaled Schoenfeld residuals).
    """
    if transform not in TIME_TRANSFORMS:
        raise ValueError(f"transform must be one of {TIME_TRANSFORMS}")
    durations_v, events_v, X = _prepare(fit, durations, events, covariates)
    resid = schoenfeld_residuals(fit, durations_v, events_v, X)
    n_deaths = resid.residuals.shape[0]
    if n_deaths < 2:
        raise ValueError("at least two events are required for the PH test")
    g = transform_event_times(resid.times, durations_v, events_v, transform)
    centered = g - g.mean()
    ss = float(centered @ centered)
    if ss <= 0.0:
        raise ValueError("transformed event times are constant; the PH test is undefined")

    covariance = _covariance(fit, durations_v, events_v, X)
    scaled_minus_beta = n_deaths * resid.residuals @ covariance
    test = centered @ scaled_minus_beta
    stats = test**2 / (np.diag(covariance) * n_deaths * ss)
    p_values = chi2.sf(stats, 1)

    correlations = np.empty(test.size)
    for j in range(test.size):
        col = scaled_minus_beta[:, j]
        sd = np.std(col)
        correlations[j] = 0.0 if sd == 0 else float(np.corrcoef(centered, col)[0, 1])

    raw = centered @ resid.residuals
    global_stat = float(raw @ covariance @ raw) * n_deaths / ss
    p = test.size
    return ProportionalHazardsTest(
        feature_names=tuple(fit.feature_names),
        transform=transform,
        correlations=correlations,
        statistics=stats,
        p_values=p_values,
        global_statistic=global_stat,
        global_df=p,
        global_p_value=float(chi2.sf(global_stat, p)),
        n_events=n_deaths,
    )
