"""Tests for the Cox proportional hazards estimator."""

import numpy as np
import pytest
from scipy.optimize import minimize_scalar

from survival_kit.concordance import concordance_index
from survival_kit.cox import breslow_partial_likelihood, fit_cox_ph
from survival_kit.synth import generate_ph_data, generate_survival_data


def test_partial_likelihood_at_zero_matches_hand_computation():
    durations = np.array([1.0, 2.0, 3.0])
    events = np.array([True, True, True])
    X = np.array([0.0, 1.0, 0.0])
    loglik, score, information, times, increments = breslow_partial_likelihood(
        [0.0], durations, events, X.reshape(-1, 1)
    )
    assert loglik == pytest.approx(-np.log(6.0))
    assert np.allclose(times, [1.0, 2.0, 3.0])
    assert increments[0] == pytest.approx(1.0 / 3.0)


def test_partial_likelihood_with_tied_deaths_uses_breslow():
    durations = np.array([1.0, 1.0, 3.0])
    events = np.array([True, True, True])
    X = np.array([[0.0], [1.0], [0.0]])
    loglik, _, _, _, _ = breslow_partial_likelihood([0.0], durations, events, X)
    # Two deaths at t=1 with three at risk, then one death of the last subject.
    assert loglik == pytest.approx(-2.0 * np.log(3.0))


def test_mle_matches_scalar_maximizer_on_a_tiny_sample():
    durations = np.array([1.0, 2.0, 3.0])
    events = np.array([True, True, True])
    X = np.array([0.0, 1.0, 0.0])

    def negative_loglik(beta):
        return -(beta - np.log(2.0 + np.exp(beta)) - np.log(1.0 + np.exp(beta)))

    closed = minimize_scalar(negative_loglik, bounds=(-5.0, 5.0), method="bounded")
    fit = fit_cox_ph(durations, events, X)
    assert fit.coefficients[0] == pytest.approx(closed.x, rel=1e-5)
    assert fit.log_partial_likelihood == pytest.approx(-closed.fun, rel=1e-8)
    _, score, _, _, _ = breslow_partial_likelihood(
        fit.coefficients, durations, events, X.reshape(-1, 1)
    )
    assert score[0] == pytest.approx(0.0, abs=1e-8)


def test_score_matches_central_differences():
    durations = np.array([1.0, 2.0, 4.0, 7.0, 9.0])
    events = np.array([True, True, False, True, True])
    X = np.array([[0.0, 1.0], [1.0, 0.0], [1.0, 1.0], [0.0, 0.0], [1.0, -1.0]])
    beta = np.array([0.3, -0.2])
    loglik, score, _, _, _ = breslow_partial_likelihood(beta, durations, events, X)
    eps = 1e-6
    numeric = np.empty(2)
    for j in range(2):
        bump = np.zeros(2)
        bump[j] = eps
        plus, _, _, _, _ = breslow_partial_likelihood(beta + bump, durations, events, X)
        minus, _, _, _, _ = breslow_partial_likelihood(beta - bump, durations, events, X)
        numeric[j] = (plus - minus) / (2.0 * eps)
    assert np.allclose(score, numeric, rtol=1e-5, atol=1e-5)
    assert np.isfinite(loglik)


def test_recovers_weibull_ph_coefficients():
    true = np.array([0.8, -0.5])
    data = generate_ph_data(
        4000, true, shape=1.4, scale=3.0, censor_fraction=0.0, seed=11
    )
    fit = fit_cox_ph(data.durations, data.events, data.covariates)
    assert fit.converged
    assert np.allclose(fit.coefficients, true, atol=0.12)
    assert np.all(np.abs(fit.coefficients - true) < 3.0 * fit.std_err)


def test_recovers_coefficients_under_right_censoring():
    true = np.array([0.6, -0.35])
    data = generate_ph_data(
        5000, true, shape=1.2, scale=4.0, censor_fraction=0.25, seed=13
    )
    fit = fit_cox_ph(data.durations, data.events, data.covariates)
    assert 0.18 < data.censor_fraction < 0.32
    assert np.allclose(fit.coefficients, true, atol=0.15)
    assert np.all(np.abs(fit.coefficients - true) < 3.5 * fit.std_err)


def test_two_arm_weibull_hazard_ratio_matches_scale_ratio():
    shape = 1.3
    ratio = 2.0
    data = generate_survival_data(
        6000, shape=shape, scale=5.0, group_scale_ratio=ratio, seed=17
    )
    treated = (data.groups == "treatment").astype(float)
    fit = fit_cox_ph(data.durations, data.events, treated)
    expected_hr = ratio ** (-shape)
    assert fit.hazard_ratios[0] == pytest.approx(expected_hr, rel=0.12)
    assert fit.hazard_ratio_ci_lower[0] < expected_hr < fit.hazard_ratio_ci_upper[0]


def test_hazard_ratios_are_exp_coefficients():
    data = generate_ph_data(400, [0.4], seed=3)
    fit = fit_cox_ph(data.durations, data.events, data.covariates)
    assert np.allclose(fit.hazard_ratios, np.exp(fit.coefficients))
    assert np.allclose(fit.hazard_ratio_ci_lower, np.exp(fit.ci_lower))
    assert np.allclose(fit.hazard_ratio_ci_upper, np.exp(fit.ci_upper))


def test_null_likelihood_and_lr_test_for_unrelated_covariate():
    rng = np.random.default_rng(19)
    durations = rng.exponential(2.0, 250)
    events = np.ones(250, bool)
    noise = rng.normal(size=250)
    fit = fit_cox_ph(durations, events, noise)
    assert fit.p_values[0] > 0.01
    assert fit.likelihood_ratio_p_value > 0.01
    assert fit.log_partial_likelihood == pytest.approx(
        fit.log_partial_likelihood_null, abs=1.0
    )


def test_informative_scores_rank_events():
    data = generate_ph_data(800, [1.1, -0.7], shape=1.1, seed=23)
    fit = fit_cox_ph(data.durations, data.events, data.covariates)
    c_hat = concordance_index(
        data.durations, data.events, fit.linear_predictor(data.covariates)
    )
    assert c_hat.c_index > 0.65


def test_baseline_cumulative_hazard_is_non_decreasing():
    data = generate_ph_data(300, [0.5], censor_fraction=0.2, seed=29)
    fit = fit_cox_ph(data.durations, data.events, data.covariates)
    assert fit.baseline_time.size == np.unique(data.durations[data.events]).size
    assert np.all(np.diff(fit.baseline_cumhazard) >= -1e-12)
    assert fit.baseline_at(fit.baseline_time[0] - 1e-9) == 0.0
    assert fit.baseline_at(fit.baseline_time[0]) == pytest.approx(fit.baseline_cumhazard[0])


def test_survival_at_is_one_before_the_first_event():
    data = generate_ph_data(200, [0.3, -0.2], seed=31)
    fit = fit_cox_ph(data.durations, data.events, data.covariates)
    s0 = fit.survival_at(0.0, np.zeros(2))
    assert s0 == pytest.approx(1.0)
    later = fit.survival_at(float(np.median(data.durations)), np.zeros(2))
    assert 0.0 < later < 1.0


def test_relative_hazard_is_one_at_the_origin():
    data = generate_ph_data(120, [0.2, 0.1], seed=37)
    fit = fit_cox_ph(data.durations, data.events, data.covariates)
    assert fit.relative_hazard(np.zeros(2))[0] == pytest.approx(1.0)
    doubled = fit.relative_hazard(np.array([[1.0, 0.0], [2.0, 0.0]]))
    assert doubled[1] == pytest.approx(doubled[0] ** 2, rel=1e-12)


def test_feature_names_round_trip():
    data = generate_ph_data(80, [0.4, -0.2], seed=41)
    fit = fit_cox_ph(
        data.durations, data.events, data.covariates, feature_names=("age", "marker")
    )
    assert fit.feature_names == ("age", "marker")
    assert fit.n_observations == 80
    assert fit.n_events == int(data.events.sum())


def test_higher_confidence_level_never_narrows_the_interval():
    data = generate_ph_data(200, [0.5], seed=43)
    narrow = fit_cox_ph(data.durations, data.events, data.covariates, confidence_level=0.90)
    wide = fit_cox_ph(data.durations, data.events, data.covariates, confidence_level=0.99)
    assert np.all(wide.ci_upper - wide.ci_lower >= narrow.ci_upper - narrow.ci_lower)


def test_invalid_inputs_raise():
    with pytest.raises(ValueError, match="at least one event"):
        fit_cox_ph([1.0, 2.0], [False, False], [0.0, 1.0])
    with pytest.raises(ValueError, match="vary"):
        fit_cox_ph([1.0, 2.0, 3.0], [True, True, True], [1.0, 1.0, 1.0])
    with pytest.raises(ValueError, match="collinear"):
        X = np.column_stack([np.arange(6.0), 2.0 * np.arange(6.0)])
        fit_cox_ph(np.arange(1.0, 7.0), np.ones(6, bool), X)
    with pytest.raises(ValueError, match="finite"):
        fit_cox_ph([1.0, 2.0], [True, True], [0.0, np.nan])
    with pytest.raises(ValueError, match="feature_names"):
        fit_cox_ph([1.0, 2.0, 3.0], [True, True, True], [0.0, 1.0, 0.0], feature_names=("a", "b"))
    with pytest.raises(ValueError, match="confidence_level"):
        fit_cox_ph([1.0, 2.0, 3.0], [True, True, True], [0.0, 1.0, 0.0], confidence_level=1.0)
    with pytest.raises(ValueError):
        fit_cox_ph([1.0, 2.0], [True], [0.0, 1.0])
