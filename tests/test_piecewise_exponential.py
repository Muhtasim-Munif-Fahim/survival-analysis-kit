"""Tests for the piecewise exponential PH model."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import minimize

from survival_kit import (
    fit_cox_ph,
    fit_piecewise_exponential,
    generate_ph_data,
    piecewise_exponential_data,
)


def _data(n=600, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 2))
    beta = np.array([0.7, -0.4])
    # true baseline: 0.2 on [0, 1), 0.6 on [1, 3), 0.3 afterwards
    rates, cuts = np.array([0.2, 0.6, 0.3]), np.array([1.0, 3.0])
    u = rng.exponential(size=n) / np.exp(X @ beta)
    H_cuts = np.concatenate([[0.0], np.cumsum(rates[:-1] * np.diff(cuts, prepend=0.0))])
    k = np.searchsorted(H_cuts, u, side="right") - 1
    lower = np.concatenate([[0.0], cuts])
    T = lower[k] + (u - H_cuts[k]) / rates[k]
    C = rng.uniform(0, 8, size=n)
    return np.minimum(T, C), T <= C, X, beta, rates, cuts


def _brute_loglik(theta, t, e, X, cuts):
    K = cuts.size + 1
    lam, beta = np.exp(theta[:K]), theta[K:]
    lower = np.concatenate([[0.0], cuts])
    upper = np.concatenate([cuts, [np.inf]])
    E = np.clip(np.minimum(t[:, None], upper) - lower, 0, None)
    idx = np.searchsorted(cuts, t, side="right")
    eta = X @ beta
    return float(np.sum(e * (np.log(lam[idx]) + eta)) - np.sum((E @ lam) * np.exp(eta)))


def test_recovers_truth_and_matches_brute_force_mle():
    t, e, X, beta, rates, cuts = _data()
    res = fit_piecewise_exponential(t, e, X, breakpoints=cuts, feature_names=["a", "b"])
    assert res.converged
    np.testing.assert_allclose(res.coefficients, beta, atol=0.15)
    np.testing.assert_allclose(res.baseline_hazards, rates, rtol=0.3)
    theta0 = np.concatenate([np.log(res.baseline_hazards), res.coefficients])
    assert res.log_likelihood == pytest.approx(_brute_loglik(theta0, t, e, X, cuts), rel=1e-10)
    opt = minimize(lambda th: -_brute_loglik(th, t, e, X, cuts), np.zeros(5), method="BFGS",
                   options={"gtol": 1e-8})
    np.testing.assert_allclose(theta0, opt.x, atol=1e-4)
    assert res.log_likelihood >= -opt.fun - 1e-6
    # standard errors match the inverse numerical Hessian of the full likelihood
    h = 1e-4
    Hn = np.zeros((5, 5))
    for i in range(5):
        for j in range(5):
            ei, ej = np.eye(5)[i] * h, np.eye(5)[j] * h
            f = lambda th: _brute_loglik(th, t, e, X, cuts)
            Hn[i, j] = (f(theta0 + ei + ej) - f(theta0 + ei - ej) - f(theta0 - ei + ej) + f(theta0 - ei - ej)) / (4 * h * h)
    se_num = np.sqrt(np.diag(np.linalg.inv(-Hn)))
    np.testing.assert_allclose(res.std_err, se_num[3:], rtol=1e-3)
    np.testing.assert_allclose(res.std_err_log_hazards, se_num[:3], rtol=1e-3)
    assert res.lr_p_value < 1e-10
    assert "a: coef=" in res.summary()


def test_poisson_person_period_equivalence():
    from sklearn.linear_model import PoissonRegressor

    t, e, X, _, _, cuts = _data(300, seed=1)
    pp = piecewise_exponential_data(t, e, X, breakpoints=cuts)
    assert pp["event"].sum() == e.sum()
    np.testing.assert_allclose(np.bincount(pp["id"], weights=pp["exposure"]), t)
    dummies = np.eye(cuts.size + 1)[pp["interval"]]
    design = np.hstack([dummies, pp["covariates"]])
    ok = pp["exposure"] > 0
    glm = PoissonRegressor(alpha=0.0, fit_intercept=False, max_iter=1000, tol=1e-10)
    glm.fit(design[ok], pp["event"][ok] / pp["exposure"][ok], sample_weight=pp["exposure"][ok])
    res = fit_piecewise_exponential(t, e, X, breakpoints=cuts)
    np.testing.assert_allclose(glm.coef_[3:], res.coefficients, atol=1e-4)
    np.testing.assert_allclose(np.exp(glm.coef_[:3]), res.baseline_hazards, rtol=1e-3)


def test_many_intervals_approach_cox():
    data = generate_ph_data(500, [0.5, -0.8], censor_fraction=0.3, seed=3)
    t, e, X = data.durations, data.events, data.covariates
    cox = fit_cox_ph(t, e, X)
    coarse = fit_piecewise_exponential(t, e, X, breakpoints=1)
    fine = fit_piecewise_exponential(t, e, X, breakpoints=60)
    err_fine = np.abs(fine.coefficients - cox.coefficients).max()
    err_coarse = np.abs(coarse.coefficients - cox.coefficients).max()
    assert err_fine < 0.03
    assert err_fine < err_coarse


def test_no_covariates_occurrence_exposure_and_survival():
    t = np.array([0.5, 1.5, 2.0, 2.5, 4.0, 5.0])
    e = np.array([1, 1, 0, 1, 0, 1], dtype=bool)
    res = fit_piecewise_exponential(t, e, breakpoints=[1.0, 3.0])
    # interval exposures: [0,1): 0.5+5*1 ; [1,3): 0.5+1+1.5+2*2 ; [3,inf): 1+2
    np.testing.assert_allclose(res.exposure_per_interval, [5.5, 7.0, 3.0])
    np.testing.assert_allclose(res.baseline_hazards, [1 / 5.5, 2 / 7.0, 1 / 3.0])
    assert res.coefficients.size == 0 and res.lr_p_value == 1.0
    H = res.cumulative_hazard([0.5, 1.0, 2.0, 10.0])[0]
    np.testing.assert_allclose(H, [0.5 / 5.5, 1 / 5.5, 1 / 5.5 + 2 / 7.0, 1 / 5.5 + 4 / 7.0 + 7 / 3.0])
    med = res.median_survival_time()[0]
    assert res.survival_function([med])[0, 0] == pytest.approx(0.5)
    np.testing.assert_allclose(res.hazard([0.2, 1.0, 9.0])[0], res.baseline_hazards)
    assert res.intervals == [(0.0, 1.0), (1.0, 3.0), (3.0, np.inf)]


def test_zero_event_interval_and_quantile_inf():
    t = np.array([0.5, 0.7, 2.0, 2.5, 3.0])
    e = np.array([1, 1, 0, 0, 0], dtype=bool)
    res = fit_piecewise_exponential(t, e, breakpoints=[1.0])
    assert res.baseline_hazards[1] == 0.0 and np.isnan(res.std_err_log_hazards[1])
    # S plateaus at exp(-2/(0.5+0.7+3)) > 0.5 → median is infinite
    assert np.isinf(res.median_survival_time()[0])
    assert np.isfinite(res.quantile_survival_time(0.7)[0])


def test_quantile_breakpoints_and_prediction_shapes():
    t, e, X, *_ = _data(200, seed=5)
    res = fit_piecewise_exponential(t, e, X, breakpoints=4)
    assert res.n_intervals == 4 and res.events_per_interval.sum() == e.sum()
    assert np.all(np.abs(res.events_per_interval - e.sum() / 4) <= 2)
    S = res.survival_function([0.5, 1.0, 2.0], X[:3])
    assert S.shape == (3, 3) and np.all(np.diff(S, axis=1) <= 0)
    with pytest.raises(ValueError):
        res.survival_function([1.0])


def test_validation():
    t, e, X, *_ = _data(50)
    with pytest.raises(ValueError):
        fit_piecewise_exponential(t, np.zeros_like(e), X)
    with pytest.raises(ValueError):
        fit_piecewise_exponential(t, e, X, breakpoints=[2.0, 1.0])
    with pytest.raises(ValueError):
        fit_piecewise_exponential(t, e, X, breakpoints=[-1.0])
    with pytest.raises(ValueError):
        fit_piecewise_exponential(t, e, X, breakpoints=0)
    with pytest.raises(ValueError):
        fit_piecewise_exponential(t, e, X, feature_names=["only_one"])
