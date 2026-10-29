"""Tests for the logistic-Weibull mixture cure model."""

import numpy as np
import pytest
from scipy.optimize import approx_fprime

from survival_kit import fit_kaplan_meier, fit_mixture_cure, generate_cure_data
from survival_kit.cure import mixture_cure_log_likelihood


def test_gradient_matches_finite_differences():
    d, e, x, _ = generate_cure_data(150, 0.3, random_state=2, incidence_coef=0.7, latency_coef=-0.4)
    Z = X = x[:, None]
    args = (np.log(d), e, Z, X)
    theta = np.array([0.4, 0.3, 0.1, -0.2, np.log(0.8)])
    ll, g = mixture_cure_log_likelihood(theta, *args)
    num = approx_fprime(theta, lambda t: mixture_cure_log_likelihood(t, *args)[0], 1e-6)
    assert np.allclose(g, num, rtol=1e-4, atol=1e-4)
    assert np.isfinite(ll)


def test_recovers_parameters_with_covariates():
    d, e, x, _ = generate_cure_data(3000, 0.35, shape=1.5, incidence_coef=1.0,
                                    latency_coef=0.5, random_state=0)
    r = fit_mixture_cure(d, e, x, x, incidence_names=["x"], latency_names=["x"])
    assert r.converged
    assert r.incidence_intercept == pytest.approx(np.log(0.65 / 0.35), abs=0.15)
    assert r.incidence_coefficients[0] == pytest.approx(1.0, abs=0.2)
    assert r.latency_intercept == pytest.approx(0.0, abs=0.08)
    assert r.latency_coefficients[0] == pytest.approx(0.5, abs=0.08)
    assert r.shape == pytest.approx(1.5, abs=0.12)
    assert np.all(r.std_err_incidence > 0) and np.all(r.std_err_latency > 0)
    # the cure model is decisively better than a plain Weibull AFT
    assert r.log_likelihood > r.log_likelihood_no_cure
    assert r.aic < r.aic_no_cure and r.cure_vs_no_cure_p_value < 1e-10
    assert r.odds_ratios[0] == pytest.approx(np.exp(r.incidence_coefficients[0]))


def test_intercept_only_cure_fraction_matches_km_plateau():
    d, e, _, _ = generate_cure_data(1500, 0.3, random_state=1)
    r = fit_mixture_cure(d, e)
    lo, hi = r.cure_fraction_ci()
    assert lo < 0.3 < hi and lo < r.cure_fraction < hi
    km = fit_kaplan_meier(d, e)
    t_last = d[e].max()
    km_plateau = float(np.asarray(km.survival)[np.asarray(km.time) <= t_last][-1])
    assert r.cure_fraction == pytest.approx(km_plateau, abs=0.03)
    s = r.survival([0.0, 1.0, 50.0])
    assert s.shape == (1, 3)
    assert s[0, 0] == pytest.approx(1.0) and s[0, 2] == pytest.approx(r.cure_fraction, abs=1e-6)
    assert np.all(np.diff(s[0]) <= 0)


def test_posterior_cure_probability():
    d, e, x, cured = generate_cure_data(1500, 0.4, incidence_coef=0.8, random_state=3)
    r = fit_mixture_cure(d, e, x, None)
    post = r.posterior_cure_probability(d, e, x[:, None])
    assert np.all(post[e] == 0.0)
    prior = r.cure_probability(x[:, None])
    assert np.all(post[~e] >= prior[~e] - 1e-12)
    # long censored follow-up => almost surely cured
    late = (~e) & (d > np.quantile(d, 0.8))
    assert post[late].mean() > 0.95
    # posterior separates the truly cured among censored subjects
    assert post[~e & cured].mean() > post[~e & ~cured].mean() + 0.3


def test_no_cure_population_gives_small_cure_fraction():
    rng = np.random.default_rng(5)
    t = rng.weibull(1.3, 800) * 2.0
    c = rng.uniform(0, 12, 800)
    d, e = np.minimum(t, c), t <= c
    r = fit_mixture_cure(d, e)
    assert r.cure_fraction < 0.03
    assert r.log_likelihood >= r.log_likelihood_no_cure - 1e-4
    assert r.cure_vs_no_cure_p_value > 0.01


def test_validation():
    d, e, x, _ = generate_cure_data(100, 0.3, random_state=0)
    with pytest.raises(ValueError):
        fit_mixture_cure(d, np.ones_like(e))
    with pytest.raises(ValueError):
        fit_mixture_cure(d, np.zeros_like(e))
    with pytest.raises(ValueError):
        fit_mixture_cure(np.r_[0.0, d[1:]], e)
    with pytest.raises(ValueError):
        fit_mixture_cure(d, e, np.ones(100))
    with pytest.raises(ValueError):
        fit_mixture_cure(d, e, x, incidence_names=["a", "b"])
    with pytest.raises(ValueError):
        generate_cure_data(10, cure_fraction=1.0)
    r = fit_mixture_cure(d, e)
    with pytest.raises(ValueError):
        r.uncured_probability(x)
