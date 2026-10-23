"""Elastic-net penalized Cox regression with a regularization path and cross-validation.

``fit_cox_ph`` maximises the Breslow partial likelihood without a penalty.
That breaks down with many or collinear covariates.  Following glmnet's Cox
family (Simon, Friedman, Hastie and Tibshirani, JSS 2011), this module minimises

    -(1/n) * ell(beta) + lambda * [ alpha * ||beta||_1 + (1 - alpha)/2 * ||beta||_2^2 ]

where ``ell`` is the Breslow partial log-likelihood, ``alpha = 1`` is the lasso
and ``alpha = 0`` is ridge.  Covariates are standardised internally by default
(population SD, as in glmnet), and coefficients are reported on the original scale.

The solver is a proximal Newton method.  Each outer step minimises the exact
second-order model of ``-ell/n`` plus the penalty by cyclic coordinate
descent, then backtracks on the true objective.  Iteration stops when the
KKT conditions hold to ``tol``.  The likelihood, score and information use
sorted cumulative sums (``O(n p^2)``).

``penalized_cox_path`` fits a decreasing log-spaced ``lambda`` grid with warm
starts, starting from ``lambda_max`` (the smallest ``lambda`` with all
coefficients zero when ``alpha > 0``).  ``cv_penalized_cox`` chooses ``lambda``
by the cross-validated partial likelihood of Verweij and van Houwelingen (1993).
For each fold it computes ``ell_full(beta_-k) - ell_-k(beta_-k)`` and returns
``lambda_min`` and the one-standard-error ``lambda_1se``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .cox import as_covariate_matrix
from .utils import validate_durations_events


def _prepare(durations, events, covariates):
    durations, events = validate_durations_events(durations, events)
    if not np.any(events):
        raise ValueError("at least one event is required")
    X_probe = np.asarray(covariates, dtype=float)
    p = 1 if X_probe.ndim == 1 else X_probe.shape[1]
    X = as_covariate_matrix(covariates, n_features=p, n_observations=durations.size)
    return durations, events.astype(bool), X


def cox_loglik_sorted(beta, durations, events, X, *, need_info=True):
    """Breslow partial log-likelihood, score and information using cumulative sums.

    Gives the same values as :func:`survival_kit.cox.breslow_partial_likelihood`,
    without the per-time loop.
    """
    beta = np.asarray(beta, dtype=float).reshape(-1)
    order = np.argsort(-durations, kind="stable")
    t = durations[order]
    d = events[order].astype(float)
    Xs = X[order]
    eta = Xs @ beta
    offset = float(np.max(eta))
    w = np.exp(eta - offset)
    # last index of each block of equal times (descending order): risk set = prefix up to it
    last = np.r_[np.flatnonzero(t[1:] != t[:-1]), t.size - 1]
    first = np.r_[0, last[:-1] + 1]
    S0 = np.cumsum(w)[last]
    S1 = np.cumsum(w[:, None] * Xs, axis=0)[last]
    D = np.add.reduceat(d, first)
    DX = np.add.reduceat(d[:, None] * Xs, first, axis=0)
    keep = D > 0
    S0, S1, D, DX = S0[keep], S1[keep], D[keep], DX[keep]
    mean = S1 / S0[:, None]
    loglik = float(np.sum(DX @ beta) - np.sum(D * (np.log(S0) + offset)))
    score = DX.sum(axis=0) - (D[:, None] * mean).sum(axis=0)
    if not need_info:
        return loglik, score, None
    S2 = np.cumsum(w[:, None, None] * Xs[:, :, None] * Xs[:, None, :], axis=0)[last][keep]
    info = np.einsum("k,kij->ij", D, S2 / S0[:, None, None]) - np.einsum("k,ki,kj->ij", D, mean, mean)
    return loglik, score, info


def _penalty(beta, lam, alpha):
    return lam * (alpha * np.abs(beta).sum() + 0.5 * (1 - alpha) * beta @ beta)


def _soft(z, g):
    return np.sign(z) * np.maximum(np.abs(z) - g, 0.0)


def _kkt_violation(beta, grad, lam, alpha):
    """Max violation of the stationarity conditions (grad = (1/n) * score)."""
    r = grad - lam * (1 - alpha) * beta
    nz = beta != 0
    v = np.zeros_like(beta)
    v[nz] = np.abs(r[nz] - lam * alpha * np.sign(beta[nz]))
    v[~nz] = np.maximum(np.abs(r[~nz]) - lam * alpha, 0.0)
    return float(v.max()) if v.size else 0.0


def _solve(durations, events, Z, lam, alpha, beta0, max_iter, tol):
    n, p = Z.shape
    beta = beta0.copy()
    ll, sc, info = cox_loglik_sorted(beta, durations, events, Z)
    obj = -ll / n + _penalty(beta, lam, alpha)
    for it in range(1, max_iter + 1):
        g = sc / n
        H = info / n
        if _kkt_violation(beta, g, lam, alpha) < tol:
            return beta, it - 1, True
        # coordinate descent on  -g'd + 0.5 d'Hd + pen(beta + d)
        b = beta.copy()
        Hb = np.zeros(p)  # H @ (b - beta)
        diag = np.diag(H) + lam * (1 - alpha)
        for _ in range(500):
            max_change = 0.0
            for j in range(p):
                if diag[j] <= 0:
                    continue
                # partial residual gradient at coordinate j, excluding j's own term
                rj = g[j] - Hb[j] + H[j, j] * (b[j] - beta[j]) + H[j, j] * beta[j]
                new = _soft(rj, lam * alpha) / diag[j]
                delta = new - b[j]
                if delta != 0.0:
                    Hb += H[:, j] * delta
                    b[j] = new
                    max_change = max(max_change, abs(delta) * np.sqrt(diag[j]))
            if max_change < tol * 1e-2:
                break
        direction = b - beta
        step = 1.0
        while True:
            trial = beta + step * direction
            tll, tsc, tinfo = cox_loglik_sorted(trial, durations, events, Z)
            tobj = -tll / n + _penalty(trial, lam, alpha)
            if np.isfinite(tobj) and tobj <= obj + 1e-14:
                break
            step *= 0.5
            if step < 1e-10:
                return beta, it, _kkt_violation(beta, sc / n, lam, alpha) < 10 * tol
        beta, ll, sc, info, obj = trial, tll, tsc, tinfo, tobj
    return beta, max_iter, _kkt_violation(beta, sc / n, lam, alpha) < tol


@dataclass
class PenalizedCoxResult:
    """Elastic-net Cox fit at a single ``lambda``."""

    coefficients: np.ndarray
    lambda_: float
    alpha: float
    log_partial_likelihood: float
    n_observations: int
    n_events: int
    n_iterations: int
    converged: bool
    feature_names: tuple

    @property
    def hazard_ratios(self):
        return np.exp(self.coefficients)

    @property
    def active_set(self):
        return tuple(name for name, b in zip(self.feature_names, self.coefficients) if b != 0)

    def linear_predictor(self, covariates):
        X = as_covariate_matrix(covariates, n_features=self.coefficients.size)
        return X @ self.coefficients


@dataclass
class PenalizedCoxPath:
    """Coefficients along a decreasing ``lambda`` grid (original covariate scale)."""

    lambdas: np.ndarray
    coefficients: np.ndarray  # (n_lambdas, p)
    alpha: float
    log_partial_likelihood: np.ndarray
    n_nonzero: np.ndarray
    feature_names: tuple
    converged: np.ndarray


@dataclass
class PenalizedCoxCVResult:
    """Cross-validated partial-likelihood deviance along the path."""

    path: PenalizedCoxPath
    cv_deviance: np.ndarray
    cv_deviance_se: np.ndarray
    lambda_min: float
    lambda_1se: float
    index_min: int
    index_1se: int
    fold_ids: np.ndarray = field(repr=False)

    @property
    def coefficients_min(self):
        return self.path.coefficients[self.index_min]

    @property
    def coefficients_1se(self):
        return self.path.coefficients[self.index_1se]


def _check_params(alpha, standardize=True):
    if not (0.0 <= alpha <= 1.0):
        raise ValueError("alpha must lie in [0, 1]")


def _standardize(X, standardize):
    if not standardize:
        return X, np.zeros(X.shape[1]), np.ones(X.shape[1])
    center = X.mean(axis=0)
    scale = X.std(axis=0)
    if np.any(scale == 0):
        raise ValueError("each covariate must vary across observations")
    return (X - center) / scale, center, scale


def _names(feature_names, p):
    if feature_names is None:
        return tuple(f"x{i}" for i in range(p))
    names = tuple(feature_names)
    if len(names) != p:
        raise ValueError("feature_names must match the number of covariates")
    return names


def penalized_cox_lambda_max(durations, events, covariates, *, alpha=1.0, standardize=True):
    """Smallest ``lambda`` at which every coefficient is zero (``alpha > 0``)."""
    _check_params(alpha)
    if alpha == 0:
        raise ValueError("lambda_max is infinite for ridge (alpha = 0)")
    durations, events, X = _prepare(durations, events, covariates)
    Z, _, _ = _standardize(X, standardize)
    _, sc, _ = cox_loglik_sorted(np.zeros(Z.shape[1]), durations, events, Z, need_info=False)
    return float(np.max(np.abs(sc)) / Z.shape[0] / alpha)


def fit_penalized_cox(
    durations,
    events,
    covariates,
    *,
    lambda_,
    alpha=1.0,
    standardize=True,
    feature_names=None,
    max_iter=200,
    tol=1e-8,
    _warm=None,
):
    """Elastic-net Cox regression at a single ``lambda_`` (see module docstring)."""
    _check_params(alpha)
    if not (lambda_ >= 0) or not np.isfinite(lambda_):
        raise ValueError("lambda_ must be a non-negative finite number")
    if max_iter < 1 or tol <= 0:
        raise ValueError("max_iter must be positive and tol positive")
    durations, events, X = _prepare(durations, events, covariates)
    n, p = X.shape
    names = _names(feature_names, p)
    Z, center, scale = _standardize(X, standardize)
    beta0 = np.zeros(p) if _warm is None else _warm * scale
    bz, iters, conv = _solve(durations, events, Z, float(lambda_), float(alpha), beta0, max_iter, tol)
    coef = bz / scale
    ll, _, _ = cox_loglik_sorted(coef, durations, events, X, need_info=False)
    return PenalizedCoxResult(
        coefficients=coef,
        lambda_=float(lambda_),
        alpha=float(alpha),
        log_partial_likelihood=ll,
        n_observations=n,
        n_events=int(events.sum()),
        n_iterations=iters,
        converged=bool(conv),
        feature_names=names,
    )


def penalized_cox_path(
    durations,
    events,
    covariates,
    *,
    alpha=1.0,
    lambdas=None,
    n_lambdas=50,
    lambda_min_ratio=None,
    standardize=True,
    feature_names=None,
    max_iter=200,
    tol=1e-8,
):
    """Warm-started elastic-net Cox fits along a decreasing ``lambda`` grid."""
    _check_params(alpha)
    durations, events, X = _prepare(durations, events, covariates)
    n, p = X.shape
    names = _names(feature_names, p)
    if lambdas is None:
        if n_lambdas < 1:
            raise ValueError("n_lambdas must be positive")
        a = max(alpha, 1e-3)  # ridge: glmnet's convention of using alpha = 0.001 for the grid
        lmax = penalized_cox_lambda_max(durations, events, X, alpha=a, standardize=standardize)
        ratio = lambda_min_ratio if lambda_min_ratio is not None else (1e-4 if n > p else 1e-2)
        lambdas = lmax * np.logspace(0, np.log10(ratio), n_lambdas)
    lambdas = np.sort(np.asarray(lambdas, dtype=float))[::-1]
    if lambdas.size == 0 or np.any(lambdas < 0) or not np.all(np.isfinite(lambdas)):
        raise ValueError("lambdas must be non-negative and finite")
    coefs = np.zeros((lambdas.size, p))
    lls = np.zeros(lambdas.size)
    conv = np.zeros(lambdas.size, dtype=bool)
    warm = None
    for i, lam in enumerate(lambdas):
        fit = fit_penalized_cox(
            durations, events, X, lambda_=lam, alpha=alpha, standardize=standardize,
            feature_names=names, max_iter=max_iter, tol=tol, _warm=warm,
        )
        coefs[i], lls[i], conv[i] = fit.coefficients, fit.log_partial_likelihood, fit.converged
        warm = fit.coefficients
    return PenalizedCoxPath(
        lambdas=lambdas,
        coefficients=coefs,
        alpha=float(alpha),
        log_partial_likelihood=lls,
        n_nonzero=np.count_nonzero(coefs, axis=1),
        feature_names=names,
        converged=conv,
    )


def cv_penalized_cox(
    durations,
    events,
    covariates,
    *,
    alpha=1.0,
    n_folds=5,
    lambdas=None,
    n_lambdas=50,
    lambda_min_ratio=None,
    standardize=True,
    feature_names=None,
    random_state=None,
    max_iter=200,
    tol=1e-7,
):
    """K-fold cross-validated partial likelihood (Verweij & van Houwelingen) along the path."""
    durations, events, X = _prepare(durations, events, covariates)
    n = durations.size
    if not (2 <= int(n_folds) <= n):
        raise ValueError("n_folds must be between 2 and n_samples")
    path = penalized_cox_path(
        durations, events, X, alpha=alpha, lambdas=lambdas, n_lambdas=n_lambdas,
        lambda_min_ratio=lambda_min_ratio, standardize=standardize,
        feature_names=feature_names, max_iter=max_iter, tol=tol,
    )
    rng = np.random.default_rng(random_state)
    folds = rng.permutation(np.arange(n) % int(n_folds))
    dev = np.zeros((int(n_folds), path.lambdas.size))
    for k in range(int(n_folds)):
        tr = folds != k
        if not np.any(events[tr]):
            raise ValueError("a training fold has no events; use fewer folds")
        fold_path = penalized_cox_path(
            durations[tr], events[tr], X[tr], alpha=alpha, lambdas=path.lambdas,
            standardize=standardize, max_iter=max_iter, tol=tol,
        )
        for i, b in enumerate(fold_path.coefficients):
            ll_full, _, _ = cox_loglik_sorted(b, durations, events, X, need_info=False)
            ll_tr, _, _ = cox_loglik_sorted(b, durations[tr], events[tr], X[tr], need_info=False)
            dev[k, i] = -2.0 * (ll_full - ll_tr)
    # glmnet reports the mean deviance per event; weight folds by their number of events
    w = np.array([events[folds == k].sum() for k in range(int(n_folds))], dtype=float)
    w = np.where(w > 0, w, 1e-12)
    per_event = dev / w[:, None]
    mean = np.average(per_event, axis=0, weights=w)
    var = np.average((per_event - mean) ** 2, axis=0, weights=w) / max(int(n_folds) - 1, 1)
    se = np.sqrt(var)
    i_min = int(np.argmin(mean))
    i_1se = int(np.flatnonzero(mean <= mean[i_min] + se[i_min])[0])
    return PenalizedCoxCVResult(
        path=path,
        cv_deviance=mean,
        cv_deviance_se=se,
        lambda_min=float(path.lambdas[i_min]),
        lambda_1se=float(path.lambdas[i_1se]),
        index_min=i_min,
        index_1se=i_1se,
        fold_ids=folds,
    )
