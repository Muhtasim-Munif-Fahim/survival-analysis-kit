"""Tests for Cox PH with time-varying covariates (counting process)."""

from __future__ import annotations

import numpy as np
import pytest

from survival_kit.cox import breslow_partial_likelihood, fit_cox_ph
from survival_kit.cox_tvc import (
    breslow_partial_likelihood_tvc,
    expand_to_counting_process,
    fit_cox_tvc,
    generate_tvc_data,
    validate_counting_process,
)
from survival_kit.synth import generate_ph_data


def test_validate_rejects_non_positive_length_intervals():
    with pytest.raises(ValueError, match="start < stop"):
        validate_counting_process([0.0, 1.0], [1.0, 1.0], [True, False])


def test_validate_rejects_negative_start():
    with pytest.raises(ValueError, match="non-negative"):
        validate_counting_process([-0.1], [1.0], [True])


def test_validate_rejects_length_mismatch():
    with pytest.raises(ValueError, match="same length"):
        validate_counting_process([0.0, 0.0], [1.0], [True, False])


def test_partial_likelihood_matches_fixed_cox_on_expanded_data():
    durations = np.array([1.0, 2.0, 4.0, 7.0])
    events = np.array([True, True, False, True])
    X = np.array([[0.0], [1.0], [1.0], [0.0]])
    start, stop, ev, Xcp = expand_to_counting_process(durations, events, X)
    beta = np.array([0.4])
    ll_fixed, score_fixed, info_fixed, _, _ = breslow_partial_likelihood(
        beta, durations, events, X
    )
    ll_tvc, score_tvc, info_tvc, _, _ = breslow_partial_likelihood_tvc(
        beta, start, stop, ev, Xcp
    )
    assert ll_tvc == pytest.approx(ll_fixed)
    assert np.allclose(score_tvc, score_fixed)
    assert np.allclose(info_tvc, info_fixed)


def test_fit_matches_fixed_cox_when_covariates_are_baseline_only():
    data = generate_ph_data(80, [0.7, -0.4], censor_fraction=0.15, seed=3)
    fixed = fit_cox_ph(data.durations, data.events, data.covariates)
    start, stop, ev, X = expand_to_counting_process(
        data.durations, data.events, data.covariates
    )
    tvc = fit_cox_tvc(start, stop, ev, X, feature_names=("x0", "x1"))
    assert tvc.converged
    assert np.allclose(tvc.coefficients, fixed.coefficients, rtol=1e-8, atol=1e-10)
    assert tvc.log_partial_likelihood == pytest.approx(fixed.log_partial_likelihood)
    assert tvc.feature_names == ("x0", "x1")


def test_score_matches_central_differences():
    start = np.array([0.0, 0.0, 1.0, 0.0, 2.0])
    stop = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    events = np.array([True, False, True, True, True])
    X = np.array([[0.0, 1.0], [1.0, 0.0], [1.0, 1.0], [0.0, 0.0], [1.0, -1.0]])
    beta = np.array([0.25, -0.15])
    _, score, _, _, _ = breslow_partial_likelihood_tvc(beta, start, stop, events, X)
    eps = 1e-6
    numeric = np.empty(2)
    for j in range(2):
        bump = np.zeros(2)
        bump[j] = eps
        plus, _, _, _, _ = breslow_partial_likelihood_tvc(beta + bump, start, stop, events, X)
        minus, _, _, _, _ = breslow_partial_likelihood_tvc(beta - bump, start, stop, events, X)
        numeric[j] = (plus - minus) / (2.0 * eps)
    assert np.allclose(score, numeric, rtol=1e-5, atol=1e-5)


def test_hand_computation_two_interval_risk_set():
    """Subject A dies at t=1 untreated; B is at risk untreated then treated."""
    # Intervals: A:(0,1] x=0 event; B:(0,1] x=0 censor-at-switch; B:(1,2] x=1 event
    start = np.array([0.0, 0.0, 1.0])
    stop = np.array([1.0, 1.0, 2.0])
    events = np.array([True, False, True])
    X = np.array([[0.0], [0.0], [1.0]])
    # At beta=0: event t=1 risk={A,B first}, contrib -log(2); event t=2 risk={B second}, -log(1)
    ll, score, info, times, inc = breslow_partial_likelihood_tvc([0.0], start, stop, events, X)
    assert times.tolist() == [1.0, 2.0]
    assert ll == pytest.approx(-np.log(2.0))
    # At t=1 both at risk have x=0; at t=2 the sole at-risk interval has x=1.
    # Score contributions cancel to zero at beta=0.
    assert score[0] == pytest.approx(0.0)
    assert info[0, 0] == pytest.approx(0.0)
    assert inc[0] == pytest.approx(0.5)
    assert inc[1] == pytest.approx(1.0)


def test_recovers_known_tvc_coefficient():
    true_beta = 0.9
    data = generate_tvc_data(
        3000, true_beta, shape=1.3, scale=3.5, switch_time=1.5,
        switch_spread=1.5, never_switch_fraction=0.3, censor_fraction=0.1, seed=21
    )
    fit = fit_cox_tvc(data.start, data.stop, data.events, data.covariates)
    assert fit.converged
    assert fit.coefficients[0] == pytest.approx(true_beta, abs=0.15)
    assert abs(fit.coefficients[0] - true_beta) < 3.0 * fit.std_err[0]


def test_likelihood_ratio_rejects_null_when_effect_present():
    data = generate_tvc_data(800, 1.2, censor_fraction=0.15, seed=5)
    fit = fit_cox_tvc(data.start, data.stop, data.events, data.covariates)
    assert fit.likelihood_ratio_p_value < 0.01
    assert fit.hazard_ratios[0] > 1.0


def test_rejects_constant_covariate():
    start = np.array([0.0, 0.0, 0.0])
    stop = np.array([1.0, 2.0, 3.0])
    events = np.array([True, True, True])
    X = np.ones((3, 1))
    with pytest.raises(ValueError, match="vary"):
        fit_cox_tvc(start, stop, events, X)


def test_rejects_no_events():
    start = np.array([0.0, 0.0])
    stop = np.array([1.0, 2.0])
    events = np.array([False, False])
    X = np.array([[0.0], [1.0]])
    with pytest.raises(ValueError, match="at least one event"):
        fit_cox_tvc(start, stop, events, X)


def test_feature_names_length_checked():
    start = np.array([0.0, 0.0])
    stop = np.array([1.0, 2.0])
    events = np.array([True, True])
    X = np.array([[0.0], [1.0]])
    with pytest.raises(ValueError, match="feature_names"):
        fit_cox_tvc(start, stop, events, X, feature_names=("a", "b"))


def test_wald_intervals_cover_zero_under_null():
    data = generate_tvc_data(600, 0.0, censor_fraction=0.2, seed=9)
    fit = fit_cox_tvc(data.start, data.stop, data.events, data.covariates)
    assert fit.ci_lower[0] < 0.0 < fit.ci_upper[0]


def test_generate_tvc_produces_valid_intervals():
    data = generate_tvc_data(40, 0.5, switch_time=1.0, seed=1)
    assert data.start.size == data.stop.size == data.events.size
    assert data.covariates.shape[0] == data.start.size
    assert np.all(data.stop > data.start)
    assert np.any(data.events)
    # Subjects who survive past the switch contribute two rows.
    assert data.start.size > 40 or np.all(data.stop <= 1.0)


def test_expand_rejects_non_positive_durations():
    with pytest.raises(ValueError, match="positive"):
        expand_to_counting_process([0.0, 1.0], [True, False], [[0.0], [1.0]])


def test_baseline_cumhazard_monotone():
    data = generate_tvc_data(200, 0.6, seed=2)
    fit = fit_cox_tvc(data.start, data.stop, data.events, data.covariates)
    assert fit.baseline_time.size >= 1
    assert np.all(np.diff(fit.baseline_cumhazard) >= -1e-12)
