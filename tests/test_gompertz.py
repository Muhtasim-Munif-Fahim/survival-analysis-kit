"""Tests for the Gompertz proportional-hazards model."""

import numpy as np
import pytest
from scipy.optimize import approx_fprime

from survival_kit import fit_gompertz, generate_gompertz_data, gompertz_log_likelihood


def test_gradient_matches_finite_differences():
    t, e, X = generate_gompertz_data(120, 0.1, 0.4, [0.5, -0.3], random_state=0)
    theta = np.array([np.log(0.08), 0.35, 0.4, -0.2])
    _, g = gompertz_log_likelihood(theta, t, e, X)
    num = approx_fprime(theta, lambda th: gompertz_log_likelihood(th, t, e, X)[0], 1e-7)
    assert np.allclose(g, num, rtol=1e-4, atol=1e-3)


def test_gradient_near_zero_shape_is_continuous():
    t = np.array([0.5, 1.0, 2.0, 3.0])
    e = np.array([1, 0, 1, 1])
    ll_pos, g_pos = gompertz_log_likelihood([0.0, 1e-9], t, e)
    ll_zero, g_zero = gompertz_log_likelihood([0.0, 0.0], t, e)
    assert ll_pos == pytest.approx(ll_zero, abs=1e-6)
    assert np.allclose(g_pos, g_zero, atol=1e-6)


def test_recovers_parameters():
    t, e, X = generate_gompertz_data(4000, 0.05, 0.3, [0.7, -0.5], random_state=1)
    fit = fit_gompertz(t, e, X, feature_names=["age", "dose"])
    assert fit.converged
    assert fit.b == pytest.approx(0.05, rel=0.15)
    assert fit.g == pytest.approx(0.3, abs=0.05)
    assert np.allclose(fit.coefficients, [0.7, -0.5], atol=0.08)
    assert np.all(fit.hazard_ratio_ci_lower < np.exp([0.7, -0.5]))
    assert np.all(fit.hazard_ratio_ci_upper > np.exp([0.7, -0.5]))
    assert fit.shape_lr_p_value < 1e-6
    assert fit.feature_names == ("age", "dose")


def test_exponential_data_gives_flat_shape():
    rng = np.random.default_rng(2)
    t = rng.exponential(2.0, size=3000)
    e = np.ones_like(t, dtype=bool)
    fit = fit_gompertz(t, e)
    assert abs(fit.g) < 3 * fit.std_err_g + 1e-3
    assert fit.b == pytest.approx(0.5, rel=0.1)
    # exponential MLE is closed form
    assert fit.log_likelihood_exponential == pytest.approx(3000 * np.log(3000 / t.sum()) - 3000)


def test_prediction_methods_consistent():
    t, e, _ = generate_gompertz_data(2000, 0.02, 0.5, random_state=3)
    fit = fit_gompertz(t, e)
    med = fit.median_survival_time()[0]
    assert fit.survival_function([med])[0, 0] == pytest.approx(0.5, abs=1e-8)
    times = np.linspace(0.0, 5.0, 6)
    S = fit.survival_function(times)
    assert S[0, 0] == pytest.approx(1.0)
    assert np.all(np.diff(S[0]) <= 0)
    assert fit.hazard([0.0])[0, 0] == pytest.approx(fit.b)


def test_negative_shape_has_cure_fraction():
    t, e, _ = generate_gompertz_data(4000, 0.5, -0.8, censor_time=30.0, random_state=4)
    fit = fit_gompertz(t, e)
    assert fit.g < 0
    expected = np.exp(0.5 / -0.8)
    assert fit.cure_fraction()[0] == pytest.approx(expected, abs=0.04)
    assert np.isinf(fit.quantile_survival_time(0.3)[0])


def test_validation():
    with pytest.raises(ValueError):
        fit_gompertz([1.0, 2.0], [0, 0])
    with pytest.raises(ValueError):
        fit_gompertz([1.0, 2.0, 3.0], [1, 1, 0], np.ones((3, 2)), feature_names=["a"])
    fit = fit_gompertz(*generate_gompertz_data(200, random_state=5, coefficients=[0.2])[:3])
    with pytest.raises(ValueError):
        fit.survival_function([1.0])
