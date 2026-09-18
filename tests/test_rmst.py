"""Tests for restricted mean survival time and the two-group difference."""

import numpy as np
import pytest
from scipy.stats import norm

from survival_kit.kaplan_meier import fit_kaplan_meier
from survival_kit.rmst import (
    default_truncation_time,
    restricted_mean_survival_time,
    rmst_difference_test,
)
from survival_kit.synth import generate_survival_data


# Same sample as the Kaplan-Meier fixture: events at 3, 5, 7.
DURATIONS = [3, 5, 5, 7, 9]
EVENTS = [True, False, True, True, False]


def test_hand_computed_rmst_and_greenwood_standard_error():
    result = restricted_mean_survival_time(DURATIONS, EVENTS, tau=8.0)
    # Area: 1*3 + 0.8*2 + 0.6*2 + 0.3*1 = 6.1
    assert result.rmst == pytest.approx(6.1)
    assert result.rmtl == pytest.approx(1.9)
    assert result.tau == 8.0
    # Remaining areas A(3)=3.1, A(5)=1.5, A(7)=0.3 with Greenwood
    # increments 1/20, 1/12, 1/2.
    expected_var = 3.1**2 * 0.05 + 1.5**2 / 12.0 + 0.3**2 * 0.5
    assert result.std_err == pytest.approx(np.sqrt(expected_var))
    z = float(norm.ppf(0.975))
    assert result.ci_lower == pytest.approx(max(6.1 - z * result.std_err, 0.0))
    assert result.ci_upper == pytest.approx(min(6.1 + z * result.std_err, 8.0))
    assert result.n_observations == 5
    assert result.n_events == 3


def test_rmst_matches_area_under_fitted_km_curve():
    curve = fit_kaplan_meier(DURATIONS, EVENTS)
    result = restricted_mean_survival_time(DURATIONS, EVENTS, tau=8.0)
    assert curve.restricted_mean(8.0) == pytest.approx(result.rmst)
    assert curve.restricted_mean(0.0) == 0.0
    assert curve.restricted_mean(3.0) == pytest.approx(3.0)


def test_tau_at_an_event_time_excludes_the_jump():
    result = restricted_mean_survival_time(DURATIONS, EVENTS, tau=3.0)
    assert result.rmst == pytest.approx(3.0)
    assert result.std_err == pytest.approx(0.0)


def test_tau_between_event_times():
    result = restricted_mean_survival_time(DURATIONS, EVENTS, tau=4.0)
    assert result.rmst == pytest.approx(3.8)
    assert result.std_err == pytest.approx(np.sqrt(0.8**2 * 0.05))


def test_all_censored_sample_has_rmst_equal_to_tau():
    result = restricted_mean_survival_time([1.0, 2.0, 3.0], [False, False, False], tau=2.5)
    assert result.rmst == pytest.approx(2.5)
    assert result.std_err == 0.0
    assert result.n_events == 0
    assert result.rmtl == pytest.approx(0.0)


def test_everyone_dying_together_gives_zero_variance_after_the_drop():
    result = restricted_mean_survival_time([2.0, 2.0, 2.0], [True, True, True], tau=2.0)
    assert result.rmst == pytest.approx(2.0)
    assert result.std_err == pytest.approx(0.0)
    assert result.ci_lower == pytest.approx(2.0)
    assert result.ci_upper == pytest.approx(2.0)
    # The KM curve itself may be integrated past the last event when S=0.
    curve = fit_kaplan_meier([2.0, 2.0, 2.0], [True, True, True])
    assert curve.restricted_mean(5.0) == pytest.approx(2.0)


def test_default_tau_is_the_last_follow_up_time():
    result = restricted_mean_survival_time(DURATIONS, EVENTS)
    assert result.tau == pytest.approx(9.0)
    assert default_truncation_time(DURATIONS) == pytest.approx(9.0)


def test_uncensored_exponential_rmst_matches_closed_form():
    rng = np.random.default_rng(11)
    scale = 4.0
    tau = 3.0
    times = rng.exponential(scale, 8000)
    result = restricted_mean_survival_time(times, np.ones(times.size, bool), tau=tau)
    expected = scale * (1.0 - np.exp(-tau / scale))
    assert result.rmst == pytest.approx(expected, rel=0.02)
    assert abs(result.rmst - expected) < 3.0 * result.std_err


def test_synthetic_censoring_still_tracks_exponential_rmst():
    data = generate_survival_data(
        4000,
        shape=1.0,
        scale=3.0,
        group_scale_ratio=1.0,
        censor_fraction=0.25,
        seed=19,
    )
    tau = 4.0
    result = restricted_mean_survival_time(data.durations, data.events, tau=tau)
    expected = 3.0 * (1.0 - np.exp(-tau / 3.0))
    assert result.rmst == pytest.approx(expected, rel=0.05)


def test_two_group_hand_computed_difference_and_standard_error():
    a_dur, a_ev = [1.0, 2.0, 3.0], [True, True, True]
    b_dur, b_ev = [2.0, 3.0, 4.0], [True, True, True]
    result = rmst_difference_test(a_dur, a_ev, b_dur, b_ev, tau=3.0)
    assert result.rmst_a == pytest.approx(2.0)
    assert result.rmst_b == pytest.approx(8.0 / 3.0)
    assert result.difference == pytest.approx(2.0 - 8.0 / 3.0)
    expected_se = np.sqrt(2.0 / 9.0 + 2.0 / 27.0)
    assert result.std_err == pytest.approx(expected_se)
    assert result.std_err_a == pytest.approx(np.sqrt(2.0 / 9.0))
    assert result.std_err_b == pytest.approx(np.sqrt(2.0 / 27.0))
    assert result.z_score == pytest.approx(result.difference / expected_se)
    assert result.p_value == pytest.approx(2.0 * norm.sf(abs(result.z_score)))


def test_swapping_groups_negates_the_difference_and_keeps_the_p_value():
    a_dur, a_ev = [1.0, 2.0, 3.0], [True, True, True]
    b_dur, b_ev = [2.0, 3.0, 4.0], [True, True, True]
    forward = rmst_difference_test(a_dur, a_ev, b_dur, b_ev, tau=3.0)
    backward = rmst_difference_test(b_dur, b_ev, a_dur, a_ev, tau=3.0)
    assert backward.difference == pytest.approx(-forward.difference)
    assert backward.std_err == pytest.approx(forward.std_err)
    assert backward.p_value == pytest.approx(forward.p_value)


def test_identical_samples_show_no_rmst_difference():
    result = rmst_difference_test(
        [1.0, 2.0, 4.0], [True, True, True],
        [1.0, 2.0, 4.0], [True, True, True],
        tau=4.0,
    )
    assert result.difference == pytest.approx(0.0)
    assert result.p_value == pytest.approx(1.0)
    assert result.z_score == pytest.approx(0.0)


def test_separated_synthetic_arms_reject_the_null():
    data = generate_survival_data(
        600,
        shape=1.0,
        scale=2.0,
        group_scale_ratio=3.0,
        censor_fraction=0.2,
        seed=7,
    )
    control = data.groups == "control"
    treated = data.groups == "treatment"
    tau = default_truncation_time(data.durations[control], data.durations[treated])
    result = rmst_difference_test(
        data.durations[treated],
        data.events[treated],
        data.durations[control],
        data.events[control],
        tau=tau,
    )
    expected_t = 6.0 * (1.0 - np.exp(-tau / 6.0))
    expected_c = 2.0 * (1.0 - np.exp(-tau / 2.0))
    assert result.rmst_a == pytest.approx(expected_t, rel=0.08)
    assert result.rmst_b == pytest.approx(expected_c, rel=0.08)
    assert result.difference > 0.0
    assert result.p_value < 1e-6


def test_same_rate_synthetic_arms_fail_to_reject_the_null():
    data = generate_survival_data(
        500,
        shape=1.0,
        scale=3.0,
        group_scale_ratio=1.0,
        censor_fraction=0.15,
        seed=23,
    )
    control = data.groups == "control"
    treated = data.groups == "treatment"
    result = rmst_difference_test(
        data.durations[control],
        data.events[control],
        data.durations[treated],
        data.events[treated],
    )
    assert result.p_value > 0.01


def test_two_group_default_tau_is_the_earlier_last_follow_up():
    result = rmst_difference_test(
        [1.0, 2.0, 5.0], [True, True, True],
        [1.5, 3.0, 4.0], [True, True, True],
    )
    assert result.tau == pytest.approx(4.0)


def test_invalid_inputs_raise():
    with pytest.raises(ValueError, match="positive"):
        restricted_mean_survival_time(DURATIONS, EVENTS, tau=0.0)
    with pytest.raises(ValueError, match="positive"):
        restricted_mean_survival_time(DURATIONS, EVENTS, tau=-1.0)
    with pytest.raises(ValueError, match="last follow-up"):
        restricted_mean_survival_time(DURATIONS, EVENTS, tau=100.0)
    with pytest.raises(ValueError, match="confidence_level"):
        restricted_mean_survival_time(DURATIONS, EVENTS, tau=4.0, confidence_level=1.0)
    with pytest.raises(ValueError, match="last follow-up"):
        rmst_difference_test(
            [1.0, 2.0], [True, True], [1.0, 3.0], [True, True], tau=5.0
        )
    with pytest.raises(ValueError):
        fit_kaplan_meier(DURATIONS, EVENTS).restricted_mean(-1.0)
