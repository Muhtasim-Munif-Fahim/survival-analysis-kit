"""Tests for the log-rank tests."""

import numpy as np
import pytest

from survival_kit.logrank import log_rank_test, log_rank_test_groups


A_DUR, A_EV = [1, 2, 3], [True, True, True]
B_DUR, B_EV = [2, 3, 4], [True, True, True]


@pytest.fixture()
def hand_result():
    return log_rank_test(A_DUR, A_EV, B_DUR, B_EV)


def test_hand_computed_statistic_and_p_value(hand_result):
    assert hand_result.statistic == pytest.approx(1.2831, rel=1e-3)
    assert 0.25 < hand_result.p_value < 0.26
    assert hand_result.degrees_of_freedom == 1


def test_observed_and_expected_deaths_are_reported(hand_result):
    assert np.allclose(hand_result.observed_deaths, [3.0, 3.0])
    assert hand_result.expected_deaths.sum() == pytest.approx(6.0)
    assert hand_result.expected_deaths[0] == pytest.approx(1.9666667, rel=1e-6)
    assert hand_result.expected_deaths[1] == pytest.approx(4.0333333, rel=1e-6)


def test_swapping_groups_preserves_statistic_and_p_value():
    forward = log_rank_test(A_DUR, A_EV, B_DUR, B_EV)
    backward = log_rank_test(B_DUR, B_EV, A_DUR, A_EV)
    assert backward.statistic == pytest.approx(forward.statistic)
    assert backward.p_value == pytest.approx(forward.p_value)


def test_identical_samples_yield_zero_separation():
    result = log_rank_test(A_DUR, A_EV, list(A_DUR), list(A_EV))
    assert result.statistic < 1e-12
    assert result.p_value > 0.99


def test_subject_censored_before_all_deaths_contributes_nothing():
    result = log_rank_test([1.0], [False], [5.0, 6.0], [True, True])
    assert result.statistic == 0.0
    assert result.p_value == 1.0
    assert result.observed_deaths[0] == 0.0
    assert result.expected_deaths[0] == 0.0


def test_separated_exponential_samples_reject_the_null():
    rng = np.random.default_rng(11)
    a = rng.exponential(scale=1.0, size=150)
    b = rng.exponential(scale=3.0, size=150)
    result = log_rank_test(a, np.ones(150, bool), b, np.ones(150, bool))
    assert result.statistic > 10.0
    assert result.p_value < 1e-3


def test_same_rate_samples_fail_to_reject_the_null():
    rng = np.random.default_rng(23)
    a = rng.exponential(scale=2.0, size=150)
    b = rng.exponential(scale=2.0, size=150)
    result = log_rank_test(a, np.ones(150, bool), b, np.ones(150, bool))
    assert result.p_value > 0.01


def test_k_group_interface_matches_two_sample_interface():
    via_groups = log_rank_test_groups(
        np.concatenate([A_DUR, B_DUR]).astype(float),
        np.array(A_EV + B_EV),
        np.array(["a"] * 3 + ["b"] * 3),
    )
    via_pair = log_rank_test(A_DUR, A_EV, B_DUR, B_EV)
    assert via_groups.statistic == pytest.approx(via_pair.statistic)
    assert via_groups.p_value == pytest.approx(via_pair.p_value)
    assert np.allclose(via_groups.observed_deaths, via_pair.observed_deaths)


def test_three_identical_groups_show_no_separation():
    rng = np.random.default_rng(31)
    durations = np.concatenate([rng.exponential(2.0, 120) for _ in range(3)])
    events = np.ones(durations.size, bool)
    groups = np.repeat(["c1", "c2", "c3"], 120)
    result = log_rank_test_groups(durations, events, groups)
    assert result.degrees_of_freedom == 2
    assert result.p_value > 0.01


def test_three_separated_groups_reject_the_null():
    rng = np.random.default_rng(37)
    durations = np.concatenate([rng.exponential(s, 120) for s in (0.5, 1.5, 4.0)])
    events = np.ones(durations.size, bool)
    groups = np.repeat(["low", "mid", "high"], 120)
    result = log_rank_test_groups(durations, events, groups)
    assert result.degrees_of_freedom == 2
    assert result.p_value < 1e-6


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        log_rank_test_groups([1, 2], [True, True], ["only"])
    with pytest.raises(ValueError):
        log_rank_test_groups([1, 2], [True, False], ["a", "b", "c"])
    with pytest.raises(ValueError):
        log_rank_test([1, 2], [True, False], [3], [True, False])