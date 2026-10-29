"""Mixture cure model: logistic incidence with a Weibull AFT latency.

Some populations contain a *cured* (long-term survivor) fraction that will
never experience the event; the Kaplan-Meier curve then levels off at a
plateau instead of decaying to zero. The mixture cure model (Boag 1949;
Farewell 1982; Sy & Taylor 2000) writes the population survival as

    S_pop(t | x, z) = (1 - p(z)) + p(z) * S_u(t | x)

where ``p(z) = expit(gamma_0 + z @ gamma)`` is the probability of being
*uncured* (the incidence part) and ``S_u`` is the survival of the uncured
(the latency part), here a Weibull AFT
``log T = mu + x @ beta + sigma * W`` with a standard Gumbel (minimum) ``W``.

The full right-censored log-likelihood

    events:   log p_i + log f_u(t_i | x_i)
    censored: log(1 - p_i + p_i * S_u(t_i | x_i))

is maximised directly (BFGS on the analytic gradient, several starts), and
Wald standard errors come from a central finite-difference Hessian of that
gradient. Identifiability of the cure fraction needs sufficient follow-up:
the latency distribution must have essentially finished by the last observed
time, which shows up as a long KM plateau with only censored subjects.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, log_expit
from scipy.stats import chi2, norm

from .aft import _as_design, fit_weibull_aft
from .cox import as_covariate_matrix
from .utils import validate_durations_events

_MAX_Z = 50.0


@dataclass
class MixtureCureResult:
    """Fitted logistic-Weibull mixture cure model.

    ``incidence_*`` are the logistic coefficients for being **uncured**
    (``odds_ratios = exp(incidence_coefficients)``); ``latency_*`` are the
    Weibull AFT coefficients for the uncured (time ratios ``exp(beta)``).
    """

    incidence_intercept: float
    incidence_coefficients: np.ndarray
    latency_intercept: float
    latency_coefficients: np.ndarray
    sigma: float
    shape: float
    std_err_incidence_intercept: float
    std_err_incidence: np.ndarray
    std_err_latency_intercept: float
    std_err_latency: np.ndarray
    std_err_log_sigma: float
    covariance: np.ndarray
    log_likelihood: float
    log_likelihood_no_cure: float
    aic: float
    aic_no_cure: float
    n_observations: int
    n_events: int
    converged: bool
    n_iterations: int
    incidence_names: tuple
    latency_names: tuple
    confidence_level: float

    # -- inference ----------------------------------------------------
    @property
    def odds_ratios(self):
        return np.exp(self.incidence_coefficients)

    @property
    def time_ratios(self):
        return np.exp(self.latency_coefficients)

    @property
    def incidence_p_values(self):
        return 2.0 * norm.sf(np.abs(self.incidence_coefficients / self.std_err_incidence))

    @property
    def latency_p_values(self):
        return 2.0 * norm.sf(np.abs(self.latency_coefficients / self.std_err_latency))

    @property
    def cure_fraction(self):
        """Cure probability at a zero incidence covariate vector."""
        return float(1.0 - expit(self.incidence_intercept))

    def cure_fraction_ci(self):
        """Wald interval for the baseline cure fraction (logit scale, back-transformed)."""
        z = float(norm.ppf(0.5 + self.confidence_level / 2.0))
        a, se = self.incidence_intercept, self.std_err_incidence_intercept
        return float(1.0 - expit(a + z * se)), float(1.0 - expit(a - z * se))

    @property
    def cure_vs_no_cure_lr(self):
        """``2 (l_cure - l_weibull)``. Under H0 (no cure) the cure fraction sits on
        the boundary, so the reference law is ``0.5 chi2_0 + 0.5 chi2_1``."""
        return max(2.0 * (self.log_likelihood - self.log_likelihood_no_cure), 0.0)

    @property
    def cure_vs_no_cure_p_value(self):
        return float(0.5 * chi2.sf(self.cure_vs_no_cure_lr, 1)) if self.cure_vs_no_cure_lr > 0 else 1.0

    # -- prediction ---------------------------------------------------
    def _Z(self, z):
        q = self.incidence_coefficients.size
        if q == 0:
            if z is not None:
                raise ValueError("this fit has no incidence covariates")
            return None
        return as_covariate_matrix(z, n_features=q)

    def _X(self, x):
        p = self.latency_coefficients.size
        if p == 0:
            if x is not None:
                raise ValueError("this fit has no latency covariates")
            return None
        return as_covariate_matrix(x, n_features=p)

    def uncured_probability(self, incidence_covariates=None):
        Z = self._Z(incidence_covariates)
        eta = self.incidence_intercept + (0.0 if Z is None else Z @ self.incidence_coefficients)
        return np.atleast_1d(expit(eta)).astype(float)

    def cure_probability(self, incidence_covariates=None):
        return 1.0 - self.uncured_probability(incidence_covariates)

    def _location(self, x):
        X = self._X(x)
        m = self.latency_intercept + (0.0 if X is None else X @ self.latency_coefficients)
        return np.atleast_1d(np.asarray(m, dtype=float))

    def uncured_survival(self, times, latency_covariates=None):
        """``S_u(t | x)``; returns ``(n_rows, n_times)`` (rows from the covariates)."""
        t = _as_times(times)
        m = self._location(latency_covariates)
        return _weibull_sf(t[None, :], m[:, None], self.sigma)

    def survival(self, times, incidence_covariates=None, latency_covariates=None):
        """Population survival ``1 - p + p S_u``, shape ``(n_rows, n_times)``."""
        pu = self.uncured_probability(incidence_covariates)
        su = self.uncured_survival(times, latency_covariates)
        pu, su = np.broadcast_arrays(pu[:, None], su)
        return 1.0 - pu + pu * su

    def posterior_cure_probability(self, durations, events, incidence_covariates=None,
                                   latency_covariates=None):
        """``P(cured | data)``: 0 for subjects with an event, ``(1-p) / S_pop(t)`` if censored."""
        durations, events = validate_durations_events(durations, events)
        n = durations.size
        pu = np.broadcast_to(self.uncured_probability(incidence_covariates), (n,))
        m = np.broadcast_to(self._location(latency_covariates), (n,))
        su = _weibull_sf(durations, m, self.sigma)
        post = (1.0 - pu) / (1.0 - pu + pu * su)
        return np.where(events, 0.0, post)


def _as_times(times):
    t = np.atleast_1d(np.asarray(times, dtype=float))
    if np.any(t < 0) or not np.all(np.isfinite(t)):
        raise ValueError("times must be finite and non-negative")
    return t


def _weibull_sf(t, m, sigma):
    t, m = np.broadcast_arrays(np.asarray(t, float), np.asarray(m, float))
    out = np.ones(t.shape)
    pos = t > 0
    z = np.clip((np.log(t[pos]) - m[pos]) / sigma, -_MAX_Z, _MAX_Z)
    out[pos] = np.exp(-np.exp(z))
    return out


def mixture_cure_log_likelihood(params, log_times, events, Z, X):
    """Log-likelihood and gradient. ``params = [g0, gamma..., mu, beta..., log_sigma]``."""
    q, p = Z.shape[1], X.shape[1]
    g = params[: q + 1]
    b = params[q + 1 : q + p + 2]
    log_sigma = params[-1]
    sigma = np.exp(log_sigma)
    eta = g[0] + Z @ g[1:]
    m = b[0] + X @ b[1:]
    z = np.clip((log_times - m) / sigma, -_MAX_Z, _MAX_Z)
    ez = np.exp(z)
    pu = expit(eta)
    log_pu = log_expit(eta)
    d = events

    ll_event = log_pu + z - ez - log_sigma - log_times
    su = np.exp(-ez)
    # log(1 - p + p S) computed stably: log((1-p) + p S)
    denom = (1.0 - pu) + pu * su
    denom = np.maximum(denom, 1e-300)
    ll_cens = np.log(denom)
    ll = float(np.sum(np.where(d, ll_event, ll_cens)))

    d_eta = np.where(d, 1.0 - pu, pu * (1.0 - pu) * (su - 1.0) / denom)
    d_z = np.where(d, 1.0 - ez, -pu * su * ez / denom)
    d_m = -d_z / sigma
    d_logs = np.where(d, -1.0, 0.0) - d_z * z
    grad = np.concatenate([
        [d_eta.sum()], Z.T @ d_eta, [d_m.sum()], X.T @ d_m, [d_logs.sum()],
    ])
    return ll, grad


def _numeric_information(params, args, step=1e-5):
    k = params.size
    H = np.empty((k, k))
    for j in range(k):
        h = step * max(1.0, abs(params[j]))
        e = np.zeros(k)
        e[j] = h
        gp = mixture_cure_log_likelihood(params + e, *args)[1]
        gm = mixture_cure_log_likelihood(params - e, *args)[1]
        H[:, j] = (gp - gm) / (2 * h)
    return -(H + H.T) / 2.0


def fit_mixture_cure(
    durations,
    events,
    incidence_covariates=None,
    latency_covariates=None,
    *,
    incidence_names=None,
    latency_names=None,
    confidence_level=0.95,
    max_iter=500,
    tol=1e-8,
    n_starts=3,
):
    """Fit a logistic-incidence / Weibull-latency mixture cure model by maximum likelihood.

    ``incidence_covariates`` (``Z``) drive the probability of being uncured,
    ``latency_covariates`` (``X``) the Weibull AFT time of the uncured. Either
    may be omitted (intercept only); they may also be the same matrix.
    The result also reports the plain Weibull AFT fit (no cure), its AIC and a
    boundary-corrected likelihood-ratio test for the presence of a cure fraction.
    """
    durations, events = validate_durations_events(durations, events)
    if np.any(durations <= 0.0):
        raise ValueError("durations must be strictly positive")
    n = durations.size
    if not np.any(events):
        raise ValueError("at least one event is required")
    if np.all(events):
        raise ValueError("a cure model needs censored observations")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie strictly between 0 and 1")
    if max_iter < 1 or tol <= 0 or n_starts < 1:
        raise ValueError("max_iter and n_starts must be positive and tol > 0")

    Z, z_default = _as_design(incidence_covariates, n)
    X, x_default = _as_design(latency_covariates, n)
    for M in (Z, X):
        if M.shape[1] and np.any(np.ptp(M, axis=0) == 0.0):
            raise ValueError("each covariate must vary across observations")
    z_names = z_default if incidence_names is None else tuple(incidence_names)
    x_names = x_default if latency_names is None else tuple(latency_names)
    if len(z_names) != Z.shape[1] or len(x_names) != X.shape[1]:
        raise ValueError("names must match the number of covariates")
    q, p = Z.shape[1], X.shape[1]
    log_t = np.log(durations)
    args = (log_t, events, Z, X)

    # latency start: Weibull AFT on the events only (they are all uncured)
    ev = events
    try:
        lat = fit_weibull_aft(durations[ev], np.ones(ev.sum(), bool), X[ev] if p else None)
        b0 = np.r_[lat.intercept, lat.coefficients, lat.log_sigma]
    except ValueError:
        b0 = np.r_[log_t[ev].mean(), np.zeros(p), np.log(max(log_t[ev].std(), 0.1))]
    nofit = fit_weibull_aft(durations, events, X if p else None)

    def obj(theta):
        ll, g = mixture_cure_log_likelihood(theta, *args)
        return -ll, -g

    best = None
    frac = events.mean()
    for p_unc in np.linspace(min(0.98, frac + 0.05), 0.98, n_starts) if n_starts > 1 else [min(0.98, frac + 0.1)]:
        theta0 = np.r_[np.log(p_unc / (1 - p_unc)), np.zeros(q), b0]
        res = minimize(obj, theta0, jac=True, method="BFGS",
                       options={"maxiter": max_iter, "gtol": tol * n})
        if best is None or res.fun < best.fun:
            best = res
    theta = best.x
    ll = -float(best.fun)
    info = _numeric_information(theta, args)
    try:
        cov = np.linalg.inv(info)
    except np.linalg.LinAlgError as exc:
        raise ValueError("information matrix is singular; the cure fraction may be unidentified") from exc
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    k = theta.size
    return MixtureCureResult(
        incidence_intercept=float(theta[0]),
        incidence_coefficients=theta[1 : q + 1].copy(),
        latency_intercept=float(theta[q + 1]),
        latency_coefficients=theta[q + 2 : q + p + 2].copy(),
        sigma=float(np.exp(theta[-1])),
        shape=float(np.exp(-theta[-1])),
        std_err_incidence_intercept=float(se[0]),
        std_err_incidence=se[1 : q + 1].copy(),
        std_err_latency_intercept=float(se[q + 1]),
        std_err_latency=se[q + 2 : q + p + 2].copy(),
        std_err_log_sigma=float(se[-1]),
        covariance=cov,
        log_likelihood=ll,
        log_likelihood_no_cure=float(nofit.log_likelihood),
        aic=2 * k - 2 * ll,
        aic_no_cure=2 * (p + 2) - 2 * float(nofit.log_likelihood),
        n_observations=int(n),
        n_events=int(events.sum()),
        converged=bool(best.success or np.max(np.abs(best.jac)) < 1e-4 * n),
        n_iterations=int(best.nit),
        incidence_names=z_names,
        latency_names=x_names,
        confidence_level=float(confidence_level),
    )


def generate_cure_data(n=500, cure_fraction=0.3, shape=1.5, scale=1.0, censor_max=None,
                       incidence_coef=0.0, latency_coef=0.0, random_state=None):
    """Simulate mixture-cure data with one standard-normal covariate ``x``.

    ``P(uncured | x) = expit(logit(1 - cure_fraction) + incidence_coef * x)``;
    uncured times are Weibull with ``log T = log(scale) + latency_coef * x + W / shape``.
    Censoring is uniform on ``[0, censor_max]`` (default: 3x the uncured 99th percentile).
    Returns ``(durations, events, x, cured)``.
    """
    if not 0.0 < cure_fraction < 1.0:
        raise ValueError("cure_fraction must lie strictly between 0 and 1")
    rng = np.random.default_rng(random_state)
    x = rng.normal(size=n)
    pu = expit(np.log((1 - cure_fraction) / cure_fraction) + incidence_coef * x)
    cured = rng.random(n) >= pu
    w = np.log(-np.log(rng.random(n)))  # standard Gumbel (minimum)
    t_unc = np.exp(np.log(scale) + latency_coef * x + w / shape)
    if censor_max is None:
        censor_max = 3.0 * scale * (-np.log(0.01)) ** (1.0 / shape)
    c = rng.uniform(0, censor_max, n)
    t = np.where(cured, np.inf, t_unc)
    durations = np.minimum(t, c)
    durations = np.maximum(durations, 1e-8)
    events = t <= c
    return durations, events, x, cured
