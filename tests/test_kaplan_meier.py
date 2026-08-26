"""Tests for the Kaplan-Meier estimator."""

import numpy as np
import pytest

from survival_kit.kaplan_meier import fit_kaplan_meier


DURATIONS = [3, 5, 5, 7, 9]
EVENTS = [True, False, True, True, False]


@pytest.fixture()
def curve():
    return fit_kaplan_meier(DURATIONS, EVENTS)


@pytest.fixture()
def linear_curve():
    return fit_kaplan_meier(DURATIONS, EVENTS, ci_method="linear")


def test_all_censored_keeps_survival_at_one():
    curve = fit_kaplan_meier([1.0, 2.0, 3.0], [False, False, False])
    assert curve.time.size == 0
    assert curve.at(0.0) == 1.0
    assert curve.at(10.0) == 1.0
    assert curve.median() is None


def test_uncensored_data_matches_empirical_survival():
    curve = fit_kaplan_meier([1, 2, 2, 4, 7], [True] * 5)
    assert np.allclose(curve.time, [1, 2, 4, 7])
    assert np.allclose(curve.survival, [0.8, 0.4, 0.2, 0.0])


def test_known_curve_values_with_censoring(curve):
    assert np.allclose(curve.time, [3, 5, 7])
    assert np.allclose(curve.survival, [0.8, 0.6, 0.3])


def test_greenwood_standard_errors(curve):
    expected = [0.1788854382, 0.2190890230, 0.2387467278]
    assert np.allclose(curve.std_err, expected, rtol=1e-8)


def test_linear_confidence_limits(linear_curve):
    assert linear_curve.ci_lower[0] == pytest.approx(0.449391, abs=1e-5)
    assert linear_curve.ci_upper[0] == 1.0
    assert linear_curve.ci_upper[-1] == pytest.approx(0.767935, abs=1e-5)


def test_log_log_limits_match_hand_computation(curve):
    assert curve.ci_lower[0] == pytest.approx(0.2039, abs=1e-3)
    assert curve.ci_upper[0] == pytest.approx(0.9692, abs=1e-3)


def test_log_log_limits_bracket_estimate_inside_unit_interval(curve):
    assert np.all(curve.ci_lower >= 0.0)
    assert np.all(curve.ci_upper <= 1.0)
    assert np.all(curve.ci_lower <= curve.survival)
    assert np.all(curve.ci_upper >= curve.survival)


def test_log_log_band_is_narrower_than_linear_in_the_tail(curve, linear_curve):
    log_log_width = curve.ci_upper[-1] - curve.ci_lower[-1]
    linear_width = linear_curve.ci_upper[-1] - linear_curve.ci_lower[-1]
    assert log_log_width < linear_width


def test_step_semantics(curve):
    assert curve.at(2.999) == 1.0
    assert curve.at(3.0) == 0.8
    assert np.allclose(curve.at([3.0, 5.5, 7.0]), [0.8, 0.6, 0.3])
    assert curve.at(100.0) == pytest.approx(0.3)


def test_median_detection_with_censoring(curve):
    assert curve.median() == 7.0


def test_median_for_uncensored_sample():
    curve = fit_kaplan_meier([1, 2, 2, 4, 7], [True] * 5)
    assert curve.median() == 2.0


def test_tied_event_and_censoring_times_pool_into_one_risk_set():
    curve = fit_kaplan_meier([5, 5, 5, 5], [True, False, True, False])
    assert np.allclose(curve.time, [5])
    assert np.allclose(curve.survival, [0.5])


def test_every_subject_dying_together_degenerates_gracefully():
    curve = fit_kaplan_meier([2, 2, 2], [True, True, True])
    assert curve.survival[-1] == 0.0
    assert np.isinf(curve.std_err[-1])
    assert curve.ci_lower[-1] == 0.0
    assert curve.ci_upper[-1] == 1.0


def test_higher_confidence_level_never_narrows_the_band():
    narrow = fit_kaplan_meier(DURATIONS, EVENTS, confidence_level=0.90)
    wide = fit_kaplan_meier(DURATIONS, EVENTS, confidence_level=0.99)
    assert np.all(wide.ci_upper - wide.ci_lower >= narrow.ci_upper - narrow.ci_lower)


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        fit_kaplan_meier([1, 2], [True])
    with pytest.raises(ValueError):
        fit_kaplan_meier([-1, 2], [True, True])
    with pytest.raises(ValueError):
        fit_kaplan_meier([], [])
    with pytest.raises(ValueError):
        fit_kaplan_meier([1, 2], [True, True], confidence_level=1.0)
    with pytest.raises(ValueError):
        fit_kaplan_meier([1, 2], [True, True], ci_method="wilson")