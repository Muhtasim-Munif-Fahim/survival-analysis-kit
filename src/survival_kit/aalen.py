"""Aalen's additive hazards model with time-varying cumulative coefficients.

The Cox model is multiplicative: covariates scale a common baseline hazard.
Aalen's additive hazards model is the additive counterpart

    h(t | X) = beta_0(t) + beta(t)^T X

where every coefficient may vary freely with time. Estimation accumulates
least-squares increments at each event time (Aalen 1989; Martinussen and
Scheike 2006). The reported curves are the *cumulative* coefficients

    B_j(t) = integral_0^t beta_j(s) ds

which are the natural identifiable quantities: a positive slope for ``B_j``
means covariate ``j`` raises the hazard around that time.

This module fits the ordinary-least-squares additive model with an intercept
column and returns the cumulative coefficient paths. Pointwise covariance of
the increments is accumulated under the usual martingale estimator.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from .cox import as_covariate_matrix
from .utils import step_value, validate_durations_events


@dataclass
class AalenAdditiveResult:
    """Fitted Aalen additive hazards model for right-censored data."""

    time: np.ndarray
    cumulative_coefficients: np.ndarray  # shape (n_times, 1 + p), col 0 = intercept
    std_err: np.ndarray  # same shape
    ci_lower: np.ndarray
    ci_upper: np.ndarray
    n_observations: int
    n_events: int
    feature_names: tuple  # includes intercept name at index 0

    def cumulative_at(self, times, covariate_index=None):
        """Cumulative coefficient path(s) just after each time in ``times``.

        ``covariate_index`` selects a column (``0`` = intercept). ``None``
        returns every column with shape ``(len(times), 1 + p)``.
        """
        if covariate_index is None:
            cols = [
                step_value(self.time, self.cumulative_coefficients[:, j], times, outside=0.0)
                for j in range(self.cumulative_coefficients.shape[1])
            ]
            return np.column_stack(cols)
        j = int(covariate_index)
        return step_value(self.time, self.cumulative_coefficients[:, j], times, outside=0.0)

    def cumulative_hazard_at(self, times, covariates):
        """Additive cumulative hazard ``B(t)^T [1, X]`` for one observation."""
        X = as_covariate_matrix(
            covariates, n_features=self.cumulative_coefficients.shape[1] - 1
        )
        if X.shape[0] != 1:
            raise ValueError("cumulative_hazard_at expects a single observation")
        design = np.concatenate([[1.0], X[0]])
        B = self.cumulative_at(times)  # (n_times, 1+p)
        return B @ design


def fit_aalen_additive(
    durations,
    events,
    covariates,
    *,
    feature_names=None,
    confidence_level=0.95,
    ridge=1e-8,
):
    """Fit Aalen's additive hazards model by least-squares increments.

    ``covariates`` is an ``(n, p)`` matrix (or length-``n`` for one covariate).
    An intercept column is always included as the first cumulative coefficient.
    ``ridge`` is added to the diagonal of ``X_risk^T X_risk`` for numerical
    stability when the risk set is nearly collinear.
    """
    durations, events = validate_durations_events(durations, events)
    if not np.any(events):
        raise ValueError("at least one event is required")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie strictly between 0 and 1")
    if ridge < 0.0:
        raise ValueError("ridge must be nonnegative")

    X_probe = np.asarray(covariates, dtype=float)
    n_features = 1 if X_probe.ndim == 1 else X_probe.shape[1]
    X = as_covariate_matrix(covariates, n_features=n_features, n_observations=durations.size)
    n, p = X.shape
    # Design with intercept.
    Z = np.column_stack([np.ones(n), X])
    q = p + 1

    if feature_names is None:
        names = ("intercept",) + tuple(f"x{i}" for i in range(p))
    else:
        names = ("intercept",) + tuple(feature_names)
        if len(names) != q:
            raise ValueError("feature_names must match the number of covariates")

    event_times = np.unique(durations[events])
    n_times = event_times.size
    increments = np.zeros((n_times, q), dtype=float)
    increment_var = np.zeros((n_times, q), dtype=float)

    for k, t in enumerate(event_times):
        at_risk = durations >= t
        died = at_risk & (durations == t) & events
        d_k = int(np.count_nonzero(died))
        if d_k == 0:
            continue
        Z_risk = Z[at_risk]
        n_risk = Z_risk.shape[0]
        if n_risk < q:
            # Under-determined risk set: skip this jump.
            continue
        gram = Z_risk.T @ Z_risk
        if ridge > 0.0:
            gram = gram + ridge * np.eye(q)
        try:
            gram_inv = np.linalg.inv(gram)
        except np.linalg.LinAlgError:
            continue
        # Least-squares increment: (Z'Z)^{-1} Z' dN. With ties, dN is the
        # sum of event indicators at t among the risk set.
        dN = died[at_risk].astype(float)
        delta = gram_inv @ (Z_risk.T @ dN)
        if not np.all(np.isfinite(delta)):
            continue
        increments[k] = delta
        # Diagonal of martingale variance contribution:
        # (Z'Z)^{-1} Z' diag(dN) Z (Z'Z)^{-1}, simplified when dN in {0,1}.
        # For ties use the event rows only.
        Z_events = Z[died]
        mid = Z_events.T @ Z_events
        var_mat = gram_inv @ mid @ gram_inv
        increment_var[k] = np.maximum(np.diag(var_mat), 0.0)

    cumulative = np.cumsum(increments, axis=0)
    variance = np.cumsum(increment_var, axis=0)
    std_err = np.sqrt(variance)
    z = float(norm.ppf(0.5 + confidence_level / 2.0))

    return AalenAdditiveResult(
        time=event_times,
        cumulative_coefficients=cumulative,
        std_err=std_err,
        ci_lower=cumulative - z * std_err,
        ci_upper=cumulative + z * std_err,
        n_observations=int(n),
        n_events=int(np.count_nonzero(events)),
        feature_names=names,
    )
