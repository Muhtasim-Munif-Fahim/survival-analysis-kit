"""Cox proportional hazards with time-varying covariates (counting process).

The standard Cox PH model treats covariates as fixed at baseline. When a
covariate can change during follow-up (treatment switches, biomarkers, etc.),
the natural representation is the *counting-process* / start-stop format
(Andersen and Gill 1982; Therneau and Grambsch 2000): each subject contributes
one or more intervals ``(start_i, stop_i]`` with a constant covariate vector on
that interval and an event indicator at ``stop_i``.

Risk sets at an event time ``t`` contain every interval with
``start < t <= stop``. The Breslow partial likelihood, score, and observed
information are otherwise identical to the time-fixed Cox model, so this
module reuses the same Newton-Raphson fit and returns a ``CoxPHResult``.

References
----------
Andersen, P. K. and Gill, R. D. (1982). Cox's regression model for counting
processes: a large sample study. *Annals of Statistics* 10:1100–1120.
Therneau, T. M. and Grambsch, P. M. (2000). *Modeling Survival Data:
Extending the Cox Model*. Springer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import chi2, norm

from .cox import CoxPHResult, as_covariate_matrix


def validate_counting_process(start, stop, events):
    """Coerce and validate start-stop intervals with event indicators.

    Returns float ``start``, float ``stop``, and a boolean event mask.
    Intervals must satisfy ``0 <= start < stop`` with finite endpoints.
    """
    start = np.asarray(start, dtype=float)
    stop = np.asarray(stop, dtype=float)
    events = np.asarray(events).astype(bool)
    if start.ndim != 1 or stop.ndim != 1 or events.ndim != 1:
        raise ValueError("start, stop, and events must be one-dimensional")
    if not (start.shape[0] == stop.shape[0] == events.shape[0]):
        raise ValueError("start, stop, and events must have the same length")
    if start.size == 0:
        raise ValueError("at least one interval is required")
    if not (np.all(np.isfinite(start)) and np.all(np.isfinite(stop))):
        raise ValueError("start and stop must contain only finite values")
    if np.any(start < 0.0):
        raise ValueError("start times must be non-negative")
    if np.any(stop <= start):
        raise ValueError("each interval must satisfy start < stop")
    return start, stop, events


def breslow_partial_likelihood_tvc(beta, start, stop, events, covariates):
    """Breslow partial log-likelihood for counting-process Cox data.

    At each unique event time ``t``, the risk set is ``{i : start_i < t <= stop_i}``.
    Tied deaths at ``t`` use the Breslow approximation. Returns the log-likelihood,
    score, observed information, event times, and Breslow baseline increments.
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

    for t in np.unique(stop[events]):
        at_risk = (start < t) & (stop >= t)
        died = at_risk & (stop == t) & events
        d_k = int(np.count_nonzero(died))
        if d_k == 0:
            continue
        s_k = X[died].sum(axis=0)
        eta_risk = eta[at_risk]
        if eta_risk.size == 0:
            continue
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


def fit_cox_tvc(
    start,
    stop,
    events,
    covariates,
    *,
    feature_names=None,
    confidence_level=0.95,
    max_iter=100,
    tol=1e-8,
):
    """Maximize the Breslow partial likelihood of a Cox model with TVC.

    ``start``, ``stop``, and ``events`` describe counting-process intervals.
    ``covariates`` is an ``(n_intervals, p)`` matrix (or length-``n`` for one
    covariate) held constant on each interval. There is no intercept: a
    constant shift is absorbed into the unspecified baseline hazard.
    """
    start, stop, events = validate_counting_process(start, stop, events)
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
    X = as_covariate_matrix(covariates, n_features=n_features, n_observations=start.size)
    p = X.shape[1]
    if np.any(np.ptp(X, axis=0) == 0.0):
        raise ValueError("each covariate must vary across intervals")

    if feature_names is None:
        names = tuple(f"x{i}" for i in range(p))
    else:
        names = tuple(feature_names)
        if len(names) != p:
            raise ValueError("feature_names must match the number of covariates")

    beta = np.zeros(p)
    loglik, score, information, event_times, increments = breslow_partial_likelihood_tvc(
        beta, start, stop, events, X
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
            trial_ll, trial_score, trial_info, trial_times, trial_inc = (
                breslow_partial_likelihood_tvc(trial, start, stop, events, X)
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
        n_observations=int(start.size),
        n_events=int(np.count_nonzero(events)),
        n_iterations=iteration,
        converged=True,
        feature_names=names,
        baseline_time=event_times,
        baseline_cumhazard=np.cumsum(increments),
    )


def expand_to_counting_process(durations, events, covariates):
    """Convert time-fixed observations to a single start-stop interval each.

    Useful for checking that ``fit_cox_tvc`` recovers ``fit_cox_ph`` on
    baseline-only data: every subject becomes ``(0, duration]``.
    """
    durations = np.asarray(durations, dtype=float)
    events = np.asarray(events).astype(bool)
    if durations.ndim != 1 or events.ndim != 1:
        raise ValueError("durations and events must be one-dimensional")
    if durations.size != events.size:
        raise ValueError("durations and events must have the same length")
    if durations.size == 0:
        raise ValueError("at least one observation is required")
    if np.any(durations <= 0.0):
        raise ValueError("durations must be positive for counting-process expansion")
    X_probe = np.asarray(covariates, dtype=float)
    n_features = 1 if X_probe.ndim == 1 else X_probe.shape[1]
    X = as_covariate_matrix(covariates, n_features=n_features, n_observations=durations.size)
    start = np.zeros(durations.size, dtype=float)
    return start, durations.copy(), events.copy(), X


@dataclass
class TimeVaryingSurvivalData:
    """Synthetic counting-process sample with one binary time-varying covariate."""

    start: np.ndarray
    stop: np.ndarray
    events: np.ndarray
    covariates: np.ndarray
    subject_id: np.ndarray


def generate_tvc_data(
    n_subjects,
    beta,
    *,
    shape=1.2,
    scale=4.0,
    switch_time=2.0,
    switch_spread=1.0,
    never_switch_fraction=0.25,
    censor_fraction=0.2,
    seed=0,
):
    """Draw a counting-process sample with a binary treatment that switches on.

    Each subject starts untreated (``x=0``). Subject-specific switch times are
    drawn around ``switch_time`` (spread ``switch_spread``); a fraction
    ``never_switch_fraction`` never receive treatment. Varying switch times are
    essential: a common calendar switch for everyone is collinear with the
    baseline hazard and not identifiable in a Cox model.

    Latent event times under the Weibull PH hazard
    ``h(t|x) = shape/scale * (t/scale)^{shape-1} * exp(beta * x)`` are drawn by
    inverting the cumulative hazard piecewise across each subject's switch.
    Independent exponential censoring targets approximately ``censor_fraction``.
    """
    if n_subjects < 2:
        raise ValueError("n_subjects must be at least 2")
    beta = float(np.asarray(beta, dtype=float).reshape(-1)[0])
    if shape <= 0 or scale <= 0:
        raise ValueError("shape and scale must be positive")
    if switch_time <= 0:
        raise ValueError("switch_time must be positive")
    if switch_spread < 0:
        raise ValueError("switch_spread must be non-negative")
    if not 0.0 <= never_switch_fraction < 1.0:
        raise ValueError("never_switch_fraction must lie in [0, 1)")
    if not 0.0 <= censor_fraction < 1.0:
        raise ValueError("censor_fraction must lie in [0, 1)")

    rng = np.random.default_rng(seed)
    # Subject-specific switch times (inf = never switch).
    switches = np.full(n_subjects, np.inf)
    will_switch = rng.random(n_subjects) >= never_switch_fraction
    n_switch = int(np.count_nonzero(will_switch))
    if n_switch:
        raw = switch_time + switch_spread * (rng.random(n_switch) - 0.5)
        switches[will_switch] = np.maximum(raw, 1e-6)

    # Cumulative hazard for untreated Weibull: H0(t) = (t/scale)^shape
    # With covariate path x(t)=0 on (0,s], x(t)=1 on (s,inf):
    # H(t) = H0(t) for t<=s; H(t) = H0(s) + exp(beta)*(H0(t)-H0(s)) for t>s.
    u = rng.random(n_subjects)
    target = -np.log(np.maximum(u, 1e-12))
    event_times = np.empty(n_subjects)
    for i in range(n_subjects):
        s = float(switches[i])
        if not np.isfinite(s):
            event_times[i] = scale * (target[i] ** (1.0 / shape))
            continue
        h0_switch = (s / scale) ** shape
        if target[i] <= h0_switch:
            event_times[i] = scale * (target[i] ** (1.0 / shape))
        else:
            remaining = (target[i] - h0_switch) / np.exp(beta)
            event_times[i] = scale * ((h0_switch + remaining) ** (1.0 / shape))

    if censor_fraction <= 0.0:
        censor_times = np.full(n_subjects, np.inf)
    else:
        # Calibrate exponential censoring without consuming the main draws twice.
        rate_lo, rate_hi = 1e-4, 10.0
        for _ in range(40):
            rate = 0.5 * (rate_lo + rate_hi)
            trial = rng.exponential(1.0 / rate, n_subjects)
            frac = float(np.mean(trial < event_times))
            if frac < censor_fraction:
                rate_lo = rate
            else:
                rate_hi = rate
        censor_times = rng.exponential(1.0 / max(0.5 * (rate_lo + rate_hi), 1e-8), n_subjects)

    observed = np.minimum(event_times, censor_times)
    died = event_times <= censor_times

    starts, stops, evs, xs, ids = [], [], [], [], []
    for i in range(n_subjects):
        t = float(observed[i])
        e = bool(died[i])
        s = float(switches[i])
        if (not np.isfinite(s)) or t <= s:
            starts.append(0.0)
            stops.append(t)
            evs.append(e)
            xs.append([0.0])
            ids.append(i)
        else:
            starts.append(0.0)
            stops.append(s)
            evs.append(False)
            xs.append([0.0])
            ids.append(i)
            starts.append(s)
            stops.append(t)
            evs.append(e)
            xs.append([1.0])
            ids.append(i)

    return TimeVaryingSurvivalData(
        start=np.asarray(starts, dtype=float),
        stop=np.asarray(stops, dtype=float),
        events=np.asarray(evs, dtype=bool),
        covariates=np.asarray(xs, dtype=float),
        subject_id=np.asarray(ids, dtype=int),
    )
