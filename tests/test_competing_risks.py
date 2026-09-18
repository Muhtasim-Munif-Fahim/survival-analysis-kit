"""Tests for Aalen-Johansen cumulative incidence and Gray's test."""

import numpy as np
import pytest
from scipy.stats import norm

from survival_kit.competing_risks import (
    fit_all_cumulative_incidence,
    fit_cumulative_incidence,
    gray_test,
    gray_test_groups,
)
from survival_kit.kaplan_meier import fit_kaplan_meier
from survival_kit.logrank import log_rank_test
from survival_kit.synth import generate_competing_risks_data, generate_survival_data
from survival_kit.utils import observed_causes


# Five subjects: cause 1 at 1, cause 2 at 2, cause 1 at 3, censored at 4, cause 1 at 5.
DURATIONS = [1.0, 2.0, 3.0, 4.0, 5.0]
CAUSES = [1, 2, 1, 0, 1]


@pytest.fixture()
def cif_cause1():
    return fit_cumulative_incidence(DURATIONS, CAUSES, cause=1)


@pytest.fixture()
def cif_cause2():
    return fit_cumulative_incidence(DURATIONS, CAUSES, cause=2)


def test_hand_computed_cif_jumps(cif_cause1, cif_cause2):
    assert np.allclose(cif_cause1.time, [1.0, 3.0, 5.0])
    assert np.allclose(cif_cause1.incidence, [0.2, 0.4, 0.8])
    assert np.allclose(cif_cause2.time, [2.0])
    assert np.allclose(cif_cause2.incidence, [0.2])


def test_aalen_standard_errors_match_hand_computation(cif_cause1, cif_cause2):
    assert cif_cause1.std_err[0] == pytest.approx(np.sqrt(0.032))
    assert cif_cause1.std_err[1] == pytest.approx(np.sqrt(0.048))
    assert cif_cause2.std_err[0] == pytest.approx(np.sqrt(0.032))


def test_linear_confidence_limits_clip_to_the_unit_interval(cif_cause1):
    z = float(norm.ppf(0.975))
    assert cif_cause1.ci_lower[0] == pytest.approx(max(0.2 - z * cif_cause1.std_err[0], 0.0))
    assert cif_cause1.ci_upper[-1] == pytest.approx(min(0.8 + z * cif_cause1.std_err[-1], 1.0))
    assert np.all(cif_cause1.ci_lower >= 0.0)
    assert np.all(cif_cause1.ci_upper <= 1.0)
    assert np.all(cif_cause1.ci_lower <= cif_cause1.incidence)
    assert np.all(cif_cause1.ci_upper >= cif_cause1.incidence)


def test_cifs_plus_overall_survival_partition_one(cif_cause1, cif_cause2):
    kaplan = fit_kaplan_meier(DURATIONS, np.array(CAUSES) > 0)
    times = [1.0, 2.0, 3.0, 5.0]
    total = cif_cause1.at(times) + cif_cause2.at(times) + kaplan.at(times)
    assert np.allclose(total, 1.0)


def test_single_cause_cif_matches_one_minus_kaplan_meier():
    durations = [3, 5, 5, 7, 9]
    events = [True, False, True, True, False]
    kaplan = fit_kaplan_meier(durations, events)
    cif = fit_cumulative_incidence(durations, np.array(events, dtype=int), cause=1)
    assert np.allclose(cif.time, kaplan.time)
    assert np.allclose(cif.incidence, 1.0 - kaplan.survival)
    assert np.allclose(cif.std_err, kaplan.std_err)
    assert np.allclose(cif.overall_survival, kaplan.survival)


def test_naive_one_minus_km_overestimates_the_cause_specific_cif():
    cif = fit_cumulative_incidence(DURATIONS, CAUSES, cause=1)
    naive = fit_kaplan_meier(DURATIONS, np.array(CAUSES) == 1)
    times = cif.time
    assert np.all(1.0 - naive.at(times) >= cif.at(times) - 1e-12)
    assert (1.0 - naive.at(3.0)) > cif.at(3.0)


def test_step_semantics(cif_cause1):
    assert cif_cause1.at(0.999) == 0.0
    assert cif_cause1.at(1.0) == pytest.approx(0.2)
    assert np.allclose(cif_cause1.at([1.0, 2.5, 3.0]), [0.2, 0.2, 0.4])
    assert cif_cause1.at(100.0) == pytest.approx(0.8)


def test_all_censored_keeps_cif_at_zero():
    curve = fit_cumulative_incidence([1.0, 2.0, 3.0], [0, 0, 0], cause=1)
    assert curve.time.size == 0
    assert curve.n_events == 0
    assert curve.at(10.0) == 0.0


def test_unobserved_cause_is_identically_zero():
    curve = fit_cumulative_incidence(DURATIONS, CAUSES, cause=3)
    assert curve.time.size == 0
    assert curve.at(5.0) == 0.0


def test_cif_is_non_decreasing():
    rng = np.random.default_rng(5)
    durations = rng.exponential(2.0, 200)
    causes = rng.choice([0, 1, 2], size=200, p=[0.2, 0.5, 0.3])
    curve = fit_cumulative_incidence(durations, causes, cause=1)
    assert np.all(np.diff(curve.incidence) >= -1e-15)


def test_tied_mixed_causes_share_a_risk_set():
    curve1 = fit_cumulative_incidence([1.0, 1.0, 2.0], [1, 2, 1], cause=1)
    curve2 = fit_cumulative_incidence([1.0, 1.0, 2.0], [1, 2, 1], cause=2)
    assert np.allclose(curve1.time, [1.0, 2.0])
    assert np.allclose(curve1.incidence, [1.0 / 3.0, 2.0 / 3.0])
    assert np.allclose(curve2.incidence, [1.0 / 3.0])
    assert curve1.at(2.0) + curve2.at(2.0) == pytest.approx(1.0)


def test_fit_all_returns_every_observed_cause():
    curves = fit_all_cumulative_incidence(DURATIONS, CAUSES)
    assert tuple(curve.cause for curve in curves) == (1, 2)
    assert observed_causes(CAUSES) == (1, 2)


def test_uncensored_exponential_causes_recover_the_true_cif():
    data = generate_competing_risks_data(
        12000, cause_rates=(0.4, 0.2), censor_fraction=0.0, seed=11
    )
    curve = fit_cumulative_incidence(data.durations, data.event_types, cause=1)
    times = [0.5, 1.0, 2.0]
    expected = data.true_cif(times, cause=1)
    assert np.allclose(curve.at(times), expected, rtol=0.05)


def test_synthetic_censoring_still_tracks_the_exponential_cif():
    data = generate_competing_risks_data(
        8000, cause_rates=(0.5, 0.25), censor_fraction=0.3, seed=19
    )
    curve = fit_cumulative_incidence(data.durations, data.event_types, cause=1)
    expected = data.true_cif(1.5, cause=1)
    assert curve.at(1.5) == pytest.approx(expected, rel=0.08)


def test_higher_confidence_level_never_narrows_the_band():
    narrow = fit_cumulative_incidence(DURATIONS, CAUSES, cause=1, confidence_level=0.90)
    wide = fit_cumulative_incidence(DURATIONS, CAUSES, cause=1, confidence_level=0.99)
    assert np.all(wide.ci_upper - wide.ci_lower >= narrow.ci_upper - narrow.ci_lower - 1e-15)


def test_gray_test_matches_log_rank_when_there_is_one_cause():
    a_dur, a_ev = [1, 2, 3], [1, 1, 1]
    b_dur, b_ev = [2, 3, 4], [1, 1, 1]
    gray = gray_test(a_dur, a_ev, b_dur, b_ev, cause=1)
    log_rank = log_rank_test(a_dur, [True, True, True], b_dur, [True, True, True])
    assert gray.statistic == pytest.approx(log_rank.statistic)
    assert gray.p_value == pytest.approx(log_rank.p_value)
    assert gray.degrees_of_freedom == 1
    assert np.allclose(gray.observed, log_rank.observed_deaths)


def test_gray_test_matches_log_rank_with_censoring_and_no_competing_event():
    data = generate_survival_data(400, group_scale_ratio=2.0, censor_fraction=0.25, seed=8)
    event_types = data.events.astype(int)
    gray = gray_test_groups(data.durations, event_types, data.groups, cause=1)
    log_rank = log_rank_test_groups_from_data(data)
    assert gray.statistic == pytest.approx(log_rank.statistic, rel=1e-10)
    assert gray.p_value == pytest.approx(log_rank.p_value, rel=1e-10)


def log_rank_test_groups_from_data(data):
    from survival_kit.logrank import log_rank_test_groups

    return log_rank_test_groups(data.durations, data.events, data.groups)


def test_swapping_groups_preserves_the_gray_statistic():
    a_dur, a_ev = [1.0, 2.0, 4.0], [1, 2, 1]
    b_dur, b_ev = [1.5, 2.5, 5.0], [1, 1, 2]
    forward = gray_test(a_dur, a_ev, b_dur, b_ev, cause=1)
    backward = gray_test(b_dur, b_ev, a_dur, a_ev, cause=1)
    assert backward.statistic == pytest.approx(forward.statistic)
    assert backward.p_value == pytest.approx(forward.p_value)


def test_identical_samples_show_no_cif_difference():
    result = gray_test(
        [1.0, 2.0, 3.0], [1, 2, 1],
        [1.0, 2.0, 3.0], [1, 2, 1],
        cause=1,
    )
    assert result.statistic == pytest.approx(0.0)
    assert result.p_value == pytest.approx(1.0)


def test_separated_cause1_rates_reject_the_null():
    data = generate_competing_risks_data(
        900,
        cause_rates=(0.6, 0.25),
        group_rate_ratios=(0.25, 1.0),
        censor_fraction=0.2,
        seed=7,
    )
    result = gray_test_groups(data.durations, data.event_types, data.groups, cause=1)
    control = data.groups == "control"
    treated = data.groups == "treatment"
    t = 1.5
    assert data.true_cif(t, cause=1, group="control") > data.true_cif(
        t, cause=1, group="treatment"
    )
    control_cif = fit_cumulative_incidence(
        data.durations[control], data.event_types[control], cause=1
    )
    treated_cif = fit_cumulative_incidence(
        data.durations[treated], data.event_types[treated], cause=1
    )
    assert control_cif.at(t) > treated_cif.at(t)
    assert result.p_value < 1e-6


def test_same_rate_arms_fail_to_reject_the_gray_null():
    data = generate_competing_risks_data(
        700,
        cause_rates=(0.35, 0.25),
        group_rate_ratios=1.0,
        censor_fraction=0.15,
        seed=23,
    )
    result = gray_test_groups(data.durations, data.event_types, data.groups, cause=1)
    assert result.p_value > 0.01


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        fit_cumulative_incidence([1, 2], [1], cause=1)
    with pytest.raises(ValueError):
        fit_cumulative_incidence([-1, 2], [1, 2], cause=1)
    with pytest.raises(ValueError):
        fit_cumulative_incidence([1, 2], [1.5, 2], cause=1)
    with pytest.raises(ValueError):
        fit_cumulative_incidence([1, 2], [1, 2], cause=0)
    with pytest.raises(ValueError):
        fit_cumulative_incidence([1, 2], [1, 2], cause=1, confidence_level=1.0)
    with pytest.raises(ValueError):
        gray_test_groups([1.0, 2.0], [1, 1], ["a", "a"], cause=1)
    with pytest.raises(ValueError):
        gray_test_groups([1.0, 2.0], [1, 1], ["a"], cause=1)
