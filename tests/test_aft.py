"""Tests for the Weibull accelerated failure time estimator."""

import numpy as np
import pytest
from scipy.optimize import minimize
from scipy.stats import weibull_min

from survival_kit.aft import fit_weibull_aft, weibull_aft_log_likelihood
from survival_kit.cox import fit_cox_ph
from survival_kit.synth import generate_ph_data, generate_survival_data


def _empty_design(n):
    return np.zeros((n, 0), dtype=float)


def test_log_likelihood_on_a_tiny_uncensored_table():
    log_times = np.log([1.0, 2.0, 3.0])
    events = np.array([True, True, True])
    loglik, score, information = weibull_aft_log_likelihood(
        [0.0, 0.0], log_times, events, _empty_design(3)
    )
    # At mu=0, sigma=1: z=log t, u=t, event loglik = -t, total -6.
    assert loglik == pytest.approx(-6.0)
    assert score[0] == pytest.approx(3.0)
    assert score[1] == pytest.approx(-3.0 + np.log(2.0) + 2.0 * np.log(3.0))
    assert information.shape == (2, 2)
    assert np.all(np.isfinite(information))


def test_log_likelihood_with_one_censoring_on_a_known_table():
    durations = np.array([1.0, 2.0, 3.0, 4.0])
    events = np.array([True, True, False, True])
    loglik, _, _ = weibull_aft_log_likelihood(
        [0.0, 0.0], np.log(durations), events, _empty_design(4)
    )
    # Events contribute -t; the censored t=3 contributes -u = -3.
    assert loglik == pytest.approx(-10.0)


def test_score_matches_central_differences():
    durations = np.array([1.0, 2.0, 4.0, 7.0, 9.0])
    events = np.array([True, True, False, True, True])
    X = np.array([[0.0, 1.0], [1.0, 0.0], [1.0, 1.0], [0.0, 0.0], [1.0, -1.0]])
    theta = np.array([0.2, 0.3, -0.2, -0.1])
    loglik, score, _ = weibull_aft_log_likelihood(theta, np.log(durations), events, X)
    eps = 1e-6
    numeric = np.empty(theta.size)
    for j in range(theta.size):
        bump = np.zeros(theta.size)
        bump[j] = eps
        plus, _, _ = weibull_aft_log_likelihood(theta + bump, np.log(durations), events, X)
        minus, _, _ = weibull_aft_log_likelihood(theta - bump, np.log(durations), events, X)
        numeric[j] = (plus - minus) / (2.0 * eps)
    assert np.allclose(score, numeric, rtol=1e-5, atol=1e-5)
    assert np.isfinite(loglik)


def test_information_matches_numerical_hessian():
    durations = np.array([0.8, 1.5, 2.2, 3.0, 5.5])
    events = np.array([True, True, False, True, True])
    X = np.array([[0.0], [1.0], [0.5], [-1.0], [0.25]])
    theta = np.array([0.1, -0.4, 0.05])
    _, _, information = weibull_aft_log_likelihood(theta, np.log(durations), events, X)
    eps = 1e-5
    numeric = np.empty((theta.size, theta.size))
    for j in range(theta.size):
        bump = np.zeros(theta.size)
        bump[j] = eps
        _, plus, _ = weibull_aft_log_likelihood(theta + bump, np.log(durations), events, X)
        _, minus, _ = weibull_aft_log_likelihood(theta - bump, np.log(durations), events, X)
        numeric[:, j] = -(plus - minus) / (2.0 * eps)
    assert np.allclose(information, numeric, rtol=1e-4, atol=1e-4)


def test_mle_matches_independent_maximizer_on_a_tiny_sample():
    durations = np.array([1.0, 2.0, 3.0])
    events = np.array([True, True, True])
    X = _empty_design(3)

    def negative_loglik(theta):
        loglik, _, _ = weibull_aft_log_likelihood(theta, np.log(durations), events, X)
        return -loglik

    def negative_score(theta):
        _, score, _ = weibull_aft_log_likelihood(theta, np.log(durations), events, X)
        return -score

    closed = minimize(
        negative_loglik, np.array([0.5, 0.0]), jac=negative_score, method="BFGS"
    )
    fit = fit_weibull_aft(durations, events)
    assert closed.success
    assert fit.intercept == pytest.approx(closed.x[0], rel=1e-5, abs=1e-5)
    assert fit.log_sigma == pytest.approx(closed.x[1], rel=1e-5, abs=1e-5)
    assert fit.log_likelihood == pytest.approx(-closed.fun, rel=1e-8)
    _, score, _ = weibull_aft_log_likelihood(
        [fit.intercept, fit.log_sigma], np.log(durations), events, X
    )
    assert np.allclose(score, 0.0, atol=1e-6)


def test_uncensored_mle_matches_scipy_weibull_min():
    times = np.array([0.5, 0.9, 1.2, 1.8, 2.4, 3.1, 4.0, 6.2])
    fit = fit_weibull_aft(times, np.ones(times.size, bool))
    shape, loc, scale = weibull_min.fit(times, floc=0)
    assert loc == pytest.approx(0.0)
    assert fit.shape == pytest.approx(shape, rel=1e-4)
    assert fit.scale == pytest.approx(scale, rel=1e-4)


def test_recovers_weibull_shape_and_scale_without_covariates():
    rng = np.random.default_rng(11)
    shape, scale = 1.4, 3.0
    times = scale * rng.weibull(shape, size=4000)
    fit = fit_weibull_aft(times, np.ones(times.size, bool))
    assert fit.converged
    assert fit.shape == pytest.approx(shape, rel=0.08)
    assert fit.scale == pytest.approx(scale, rel=0.08)
    assert abs(fit.shape - shape) < 3.5 * fit.std_err_shape
    assert abs(fit.intercept - np.log(scale)) < 3.5 * fit.std_err_intercept


def test_two_arm_time_ratio_matches_scale_ratio():
    shape = 1.3
    ratio = 2.0
    scale = 5.0
    data = generate_survival_data(
        5000, shape=shape, scale=scale, group_scale_ratio=ratio, seed=17
    )
    treated = (data.groups == "treatment").astype(float)
    fit = fit_weibull_aft(data.durations, data.events, treated)
    assert fit.acceleration_factors[0] == pytest.approx(ratio, rel=0.12)
    assert fit.acceleration_factor_ci_lower[0] < ratio < fit.acceleration_factor_ci_upper[0]
    assert fit.shape == pytest.approx(shape, rel=0.12)
    assert fit.intercept == pytest.approx(np.log(scale), abs=0.12)


def test_aft_coefficients_match_negative_ph_over_shape():
    true_ph = np.array([0.8, -0.5])
    shape, scale = 1.4, 3.0
    data = generate_ph_data(
        4000, true_ph, shape=shape, scale=scale, censor_fraction=0.0, seed=11
    )
    fit = fit_weibull_aft(data.durations, data.events, data.covariates)
    expected_aft = -true_ph / shape
    assert np.allclose(fit.coefficients, expected_aft, atol=0.12)
    assert np.all(np.abs(fit.coefficients - expected_aft) < 3.5 * fit.std_err)
    assert np.allclose(fit.ph_coefficients, true_ph, atol=0.15)


def test_recovers_coefficients_under_right_censoring():
    true_ph = np.array([0.6, -0.35])
    shape = 1.2
    data = generate_ph_data(
        5000, true_ph, shape=shape, scale=4.0, censor_fraction=0.25, seed=13
    )
    fit = fit_weibull_aft(data.durations, data.events, data.covariates)
    assert 0.18 < data.censor_fraction < 0.32
    expected_aft = -true_ph / shape
    assert np.allclose(fit.coefficients, expected_aft, atol=0.15)
    assert np.all(np.abs(fit.coefficients - expected_aft) < 3.5 * fit.std_err)


def test_ph_coefficients_agree_with_cox_on_weibull_data():
    true_ph = np.array([0.7])
    data = generate_ph_data(2500, true_ph, shape=1.5, scale=2.5, seed=19)
    aft = fit_weibull_aft(data.durations, data.events, data.covariates)
    cox = fit_cox_ph(data.durations, data.events, data.covariates)
    assert aft.ph_coefficients[0] == pytest.approx(cox.coefficients[0], rel=0.15)


def test_acceleration_factors_are_exp_coefficients():
    data = generate_ph_data(400, [0.4], seed=3)
    fit = fit_weibull_aft(data.durations, data.events, data.covariates)
    assert np.allclose(fit.acceleration_factors, np.exp(fit.coefficients))
    assert np.allclose(fit.acceleration_factor_ci_lower, np.exp(fit.ci_lower))
    assert np.allclose(fit.acceleration_factor_ci_upper, np.exp(fit.ci_upper))


def test_null_likelihood_and_lr_test_for_unrelated_covariate():
    rng = np.random.default_rng(19)
    durations = rng.exponential(2.0, 250)
    events = np.ones(250, bool)
    noise = rng.normal(size=250)
    fit = fit_weibull_aft(durations, events, noise)
    intercept_only = fit_weibull_aft(durations, events)
    assert fit.p_values[0] > 0.01
    assert fit.likelihood_ratio_p_value > 0.01
    assert fit.log_likelihood_null == pytest.approx(intercept_only.log_likelihood, rel=1e-6)
    assert fit.log_likelihood == pytest.approx(fit.log_likelihood_null, abs=1.0)


def test_intercept_only_lr_statistic_is_zero():
    rng = np.random.default_rng(23)
    durations = rng.exponential(1.5, 80)
    fit = fit_weibull_aft(durations, np.ones(80, bool))
    assert fit.coefficients.size == 0
    assert fit.likelihood_ratio_statistic == pytest.approx(0.0)
    assert fit.likelihood_ratio_p_value == pytest.approx(1.0)
    assert fit.feature_names == ()


def test_survival_at_is_one_at_time_zero():
    data = generate_ph_data(200, [0.3, -0.2], seed=31)
    fit = fit_weibull_aft(data.durations, data.events, data.covariates)
    s0 = fit.survival_at(0.0, np.zeros(2))
    assert s0 == pytest.approx(1.0)
    later = fit.survival_at(float(np.median(data.durations)), np.zeros(2))
    assert 0.0 < later < 1.0
    assert fit.cumulative_hazard_at(0.0, np.zeros(2)) == pytest.approx(0.0)


def test_survival_matches_weibull_formula():
    durations = np.array([1.0, 2.0, 3.0, 4.0])
    fit = fit_weibull_aft(durations, np.ones(4, bool))
    t = 2.5
    expected = np.exp(-((t / fit.scale) ** fit.shape))
    assert fit.survival_at(t) == pytest.approx(expected)
    assert fit.median() == pytest.approx(fit.scale * (np.log(2.0) ** (1.0 / fit.shape)))


def test_time_ratio_is_one_at_the_origin():
    data = generate_ph_data(120, [0.2, 0.1], seed=37)
    fit = fit_weibull_aft(data.durations, data.events, data.covariates)
    assert fit.time_ratio(np.zeros(2))[0] == pytest.approx(1.0)
    doubled = fit.time_ratio(np.array([[1.0, 0.0], [2.0, 0.0]]))
    assert doubled[1] == pytest.approx(doubled[0] ** 2, rel=1e-12)


def test_linear_predictor_adds_intercept():
    data = generate_ph_data(80, [0.4, -0.2], seed=41)
    fit = fit_weibull_aft(
        data.durations, data.events, data.covariates, feature_names=("age", "marker")
    )
    eta = fit.linear_predictor(data.covariates)
    assert np.allclose(eta, fit.intercept + data.covariates @ fit.coefficients)
    assert fit.feature_names == ("age", "marker")
    assert fit.n_observations == 80
    assert fit.n_events == int(data.events.sum())


def test_higher_confidence_level_never_narrows_the_interval():
    data = generate_ph_data(200, [0.5], seed=43)
    narrow = fit_weibull_aft(
        data.durations, data.events, data.covariates, confidence_level=0.90
    )
    wide = fit_weibull_aft(
        data.durations, data.events, data.covariates, confidence_level=0.99
    )
    assert wide.ci_upper[0] - wide.ci_lower[0] >= narrow.ci_upper[0] - narrow.ci_lower[0]
    assert wide.ci_upper_intercept - wide.ci_lower_intercept >= (
        narrow.ci_upper_intercept - narrow.ci_lower_intercept
    )


def test_quantile_inverts_survival():
    rng = np.random.default_rng(47)
    times = rng.weibull(1.2, size=200)
    fit = fit_weibull_aft(times, np.ones(200, bool))
    p = 0.3
    t_p = fit.quantile(p)
    assert fit.survival_at(t_p) == pytest.approx(1.0 - p, rel=1e-10)


def test_invalid_inputs_raise():
    with pytest.raises(ValueError, match="strictly positive"):
        fit_weibull_aft([0.0, 2.0], [True, True])
    with pytest.raises(ValueError, match="at least one event"):
        fit_weibull_aft([1.0, 2.0], [False, False])
    with pytest.raises(ValueError, match="vary"):
        fit_weibull_aft([1.0, 2.0, 3.0], [True, True, True], [1.0, 1.0, 1.0])
    with pytest.raises(ValueError, match="collinear"):
        X = np.column_stack([np.arange(6.0), 2.0 * np.arange(6.0)])
        fit_weibull_aft(np.arange(1.0, 7.0), np.ones(6, bool), X)
    with pytest.raises(ValueError, match="finite"):
        fit_weibull_aft([1.0, 2.0], [True, True], [0.0, np.nan])
    with pytest.raises(ValueError, match="feature_names"):
        fit_weibull_aft(
            [1.0, 2.0, 3.0], [True, True, True], [0.0, 1.0, 0.0], feature_names=("a", "b")
        )
    with pytest.raises(ValueError, match="confidence_level"):
        fit_weibull_aft([1.0, 2.0, 3.0], [True, True, True], confidence_level=1.0)
    with pytest.raises(ValueError):
        fit_weibull_aft([1.0, 2.0], [True], [0.0, 1.0])
