"""Tests for elastic-net penalized Cox regression."""

import numpy as np
import pytest
from scipy.optimize import minimize

from survival_kit.cox import breslow_partial_likelihood, fit_cox_ph
from survival_kit.penalized_cox import (
    cox_loglik_sorted,
    cv_penalized_cox,
    fit_penalized_cox,
    penalized_cox_lambda_max,
    penalized_cox_path,
)


def _sparse_data(n=300, p=10, seed=0, ties=True):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    beta = np.r_[1.0, -0.8, 0.5, np.zeros(p - 3)]
    T = rng.exponential(1.0 / np.exp(X @ beta))
    C = rng.exponential(2.0, n)
    d = np.minimum(T, C)
    if ties:
        d = np.round(d, 2) + 0.01
    return d, T <= C, X, beta


def _kkt(coef_std, d, e, Z, lam, alpha):
    _, score, _ = cox_loglik_sorted(coef_std, d, e, Z)
    g = score / len(d) - lam * (1 - alpha) * coef_std
    nz = coef_std != 0
    return g, nz


def test_sorted_likelihood_matches_loop_implementation():
    d, e, X, _ = _sparse_data(seed=1)
    rng = np.random.default_rng(3)
    for _ in range(3):
        b = rng.normal(scale=0.3, size=X.shape[1])
        ll, sc, info, _, _ = breslow_partial_likelihood(b, d, e, X)
        ll2, sc2, info2 = cox_loglik_sorted(b, d, e, X)
        assert ll2 == pytest.approx(ll, abs=1e-9)
        np.testing.assert_allclose(sc2, sc, atol=1e-9)
        np.testing.assert_allclose(info2, info, atol=1e-8)


def test_zero_penalty_equals_unpenalized_cox():
    d, e, X, _ = _sparse_data(seed=2)
    ref = fit_cox_ph(d, e, X)
    for standardize in (True, False):
        fit = fit_penalized_cox(d, e, X, lambda_=0.0, standardize=standardize)
        assert fit.converged
        np.testing.assert_allclose(fit.coefficients, ref.coefficients, atol=1e-7)
        assert fit.log_partial_likelihood == pytest.approx(ref.log_partial_likelihood, abs=1e-8)


@pytest.mark.parametrize("alpha", [1.0, 0.5])
def test_kkt_conditions(alpha):
    d, e, X, _ = _sparse_data(seed=3)
    lam = 0.05
    fit = fit_penalized_cox(d, e, X, lambda_=lam, alpha=alpha, standardize=False, tol=1e-10)
    g, nz = _kkt(fit.coefficients, d, e, X, lam, alpha)
    assert nz.any() and (~nz).any()
    np.testing.assert_allclose(g[nz], lam * alpha * np.sign(fit.coefficients[nz]), atol=1e-8)
    assert np.all(np.abs(g[~nz]) <= lam * alpha + 1e-8)


def test_ridge_matches_generic_optimizer():
    d, e, X, _ = _sparse_data(n=150, p=5, seed=4)
    lam, n = 0.2, len(d)

    def obj(b):
        ll, sc, _ = cox_loglik_sorted(b, d, e, X, need_info=False)
        return -ll / n + 0.5 * lam * b @ b, -sc / n + lam * b

    ref = minimize(obj, np.zeros(5), jac=True, method="BFGS", options={"gtol": 1e-12}).x
    fit = fit_penalized_cox(d, e, X, lambda_=lam, alpha=0.0, standardize=False)
    np.testing.assert_allclose(fit.coefficients, ref, atol=1e-6)
    assert np.all(fit.coefficients != 0)


def test_lambda_max_zeroes_everything_and_just_below_activates_one():
    d, e, X, _ = _sparse_data(seed=5)
    lmax = penalized_cox_lambda_max(d, e, X)
    assert np.all(fit_penalized_cox(d, e, X, lambda_=lmax * 1.0001).coefficients == 0)
    just_below = fit_penalized_cox(d, e, X, lambda_=lmax * 0.98)
    assert np.count_nonzero(just_below.coefficients) == 1
    assert just_below.active_set == ("x0",)  # the strongest signal enters first
    with pytest.raises(ValueError, match="ridge"):
        penalized_cox_lambda_max(d, e, X, alpha=0.0)


def test_standardization_is_scale_equivariant():
    d, e, X, _ = _sparse_data(seed=6)
    scale = np.array([1.0, 10.0, 0.1, 3.0, 1.0, 1.0, 5.0, 1.0, 1.0, 0.5])
    a = fit_penalized_cox(d, e, X, lambda_=0.03, alpha=0.7)
    b = fit_penalized_cox(d, e, X * scale + 4.0, lambda_=0.03, alpha=0.7)
    np.testing.assert_allclose(b.coefficients * scale, a.coefficients, atol=1e-7)
    np.testing.assert_allclose(
        a.linear_predictor(X) - a.linear_predictor(X).mean(),
        b.linear_predictor(X * scale + 4.0) - b.linear_predictor(X * scale + 4.0).mean(),
        atol=1e-6,
    )


def test_path_shrinks_and_converges():
    d, e, X, _ = _sparse_data(seed=7)
    path = penalized_cox_path(d, e, X, n_lambdas=30)
    assert path.converged.all()
    assert np.all(np.diff(path.lambdas) < 0)
    assert path.n_nonzero[0] == 0 and path.n_nonzero[-1] == X.shape[1]
    l1 = np.abs(path.coefficients).sum(axis=1)
    assert l1[-1] > l1[len(l1) // 2] > l1[0]
    assert np.all(np.diff(path.log_partial_likelihood) >= -1e-8)
    ref = fit_cox_ph(d, e, X).coefficients
    np.testing.assert_allclose(path.coefficients[-1], ref, atol=0.02)


def test_cv_recovers_true_support():
    d, e, X, beta = _sparse_data(n=400, seed=8)
    cv = cv_penalized_cox(d, e, X, n_folds=5, random_state=0, n_lambdas=40)
    assert cv.lambda_1se >= cv.lambda_min
    support = np.flatnonzero(cv.coefficients_1se)
    assert set(np.flatnonzero(beta)) <= set(support)
    assert len(support) <= 5
    assert np.all(np.sign(cv.coefficients_1se[:3]) == np.sign(beta[:3]))
    # the null model and tiny lambda are both worse than the optimum
    assert cv.cv_deviance[cv.index_min] < cv.cv_deviance[0]
    assert np.all(cv.cv_deviance_se >= 0)
    assert np.ptp(np.bincount(cv.fold_ids)) <= 1


def test_cv_seed_reproducible():
    d, e, X, _ = _sparse_data(n=200, seed=9)
    a = cv_penalized_cox(d, e, X, random_state=1, n_lambdas=10)
    b = cv_penalized_cox(d, e, X, random_state=1, n_lambdas=10)
    np.testing.assert_array_equal(a.cv_deviance, b.cv_deviance)


def test_handles_more_features_than_samples():
    rng = np.random.default_rng(10)
    n, p = 60, 100
    X = rng.normal(size=(n, p))
    T = rng.exponential(1.0 / np.exp(1.5 * X[:, 0]))
    e = np.ones(n, dtype=bool)
    path = penalized_cox_path(T, e, X, n_lambdas=15)
    assert path.converged.all()
    assert path.n_nonzero[-1] < n
    first = np.flatnonzero(path.n_nonzero > 0)[0]
    assert np.flatnonzero(path.coefficients[first]).tolist() == [0]


def test_validation():
    d, e, X, _ = _sparse_data(n=50)
    with pytest.raises(ValueError):
        fit_penalized_cox(d, e, X, lambda_=-1.0)
    with pytest.raises(ValueError):
        fit_penalized_cox(d, e, X, lambda_=0.1, alpha=1.5)
    with pytest.raises(ValueError, match="event"):
        fit_penalized_cox(d, np.zeros(50, dtype=bool), X, lambda_=0.1)
    with pytest.raises(ValueError, match="vary"):
        fit_penalized_cox(d, e, np.c_[X, np.ones(50)], lambda_=0.1)
    with pytest.raises(ValueError, match="feature_names"):
        fit_penalized_cox(d, e, X, lambda_=0.1, feature_names=["a"])
    with pytest.raises(ValueError, match="n_folds"):
        cv_penalized_cox(d, e, X, n_folds=1)
