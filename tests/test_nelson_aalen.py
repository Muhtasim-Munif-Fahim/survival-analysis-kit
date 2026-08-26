"""Tests for the Nelson-Aalen cumulative hazard estimator."""

import numpy as np
import pytest

from survival_kit.kaplan_meier import fit_kaplan_meier
from survival_kit.nelson_aalen import fit_nelson_aalen


DURATIONS = [3, 5, 5, 7, 9]
EVENTS = [True, False, True, True, False]


@pytest.fixture()
def hazard():
    return fit_nelson_aalen(DURATIONS, EVENTS)


def test_known_cumulative_hazard_values(hazard):
    assert np.allclose(hazard.time, [3, 5, 7])
    assert np.allclose(hazard.cumhazard, [0.2, 0.45, 0.95])


def test_aalen_standard_errors_match_hand_computation(hazard):
    expected = [0.2, 0.3201562119, 0.5937171044]
    assert np.allclose(hazard.std_err, expected, rtol=1e-8)


def test_no_events_gives_zero_hazard():
    h = fit_nelson_aalen([1.0, 2.0], [False, False])
    assert h.time.size == 0
    assert h.at(50.0) == 0.0


def test_cumhazard_is_non_decreasing_on_censored_sample():
    rng = np.random.default_rng(5)
    durations = rng.exponential(2.0, 120)
    events = rng.random(120) < 0.8
    h = fit_nelson_aalen(durations, events)
    assert np.all(np.diff(h.cumhazard) >= 0.0)


def test_exp_of_negative_hazard_bounds_the_kaplan_meier_curve():
    rng = np.random.default_rng(7)
    latent = rng.weibull(1.4, 250) * 3.0
    censoring = rng.exponential(6.0, 250)
    durations = np.minimum(latent, censoring)
    events = latent <= censoring
    hazard = fit_nelson_aalen(durations, events)
    kaplan = fit_kaplan_meier(durations, events)
    assert np.all(np.exp(-hazard.cumhazard) >= kaplan.survival - 1e-12)


def test_confidence_limits_bracket_estimate_and_stay_nonnegative(hazard):
    assert np.all(hazard.ci_lower >= 0.0)
    assert np.all(hazard.ci_lower <= hazard.cumhazard)
    assert np.all(hazard.ci_upper >= hazard.cumhazard)


def test_greenwood_variant_dominates_aalen_variance():
    aalen = fit_nelson_aalen(DURATIONS, EVENTS, variance_method="aalen")
    greenwood = fit_nelson_aalen(DURATIONS, EVENTS, variance_method="greenwood")
    assert np.all(greenwood.std_err >= aalen.std_err - 1e-15)


def test_greenwood_handles_every_subject_dying_together():
    h = fit_nelson_aalen([2, 2, 2], [True, True, True], variance_method="greenwood")
    assert h.cumhazard[-1] == pytest.approx(1.0)
    assert np.isinf(h.std_err[-1])
    assert np.isinf(h.ci_upper[-1])


def test_step_semantics(hazard):
    assert hazard.at(2.999) == 0.0
    assert hazard.at(3.0) == pytest.approx(0.2)
    assert np.allclose(hazard.at([5.0, 100.0]), [0.45, 0.95])


def test_uncensored_exponential_latents_recover_the_true_cumulative_hazard():
    rate = 0.4
    rng = np.random.default_rng(13)
    durations = rng.exponential(1.0 / rate, 40000)
    events = np.ones(durations.size, bool)
    hazard = fit_nelson_aalen(durations, events)
    midpoint = float(np.quantile(durations, 0.5))
    assert hazard.at(midpoint) == pytest.approx(rate * midpoint, rel=0.03)


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        fit_nelson_aalen([1, 2], [True])
    with pytest.raises(ValueError):
        fit_nelson_aalen([-3, 1], [True, False])
    with pytest.raises(ValueError):
        fit_nelson_aalen([1, 2], [True, True], variance_method="breslowish")
    with pytest.raises(ValueError):
        fit_nelson_aalen([1, 2], [True, True], confidence_level=0.0)