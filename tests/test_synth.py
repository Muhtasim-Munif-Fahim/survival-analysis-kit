"""Tests for the synthetic survival data generator."""

import numpy as np
import pytest

from survival_kit.synth import (
    exponential_competing_cif,
    generate_competing_risks_data,
    generate_ph_data,
    generate_survival_data,
    load_csv,
    save_csv,
)


def test_same_seed_reproduces_the_sample():
    first = generate_survival_data(80, seed=5)
    second = generate_survival_data(80, seed=5)
    other = generate_survival_data(80, seed=6)
    assert np.array_equal(first.durations, second.durations)
    assert np.array_equal(first.events, second.events)
    assert not np.array_equal(first.durations, other.durations)


def test_output_shapes_and_labels():
    data = generate_survival_data(120, seed=1)
    assert data.durations.shape == (120,)
    assert data.events.dtype == bool
    assert data.groups.shape == (120,)
    assert set(np.unique(data.groups)) == {"control", "treatment"}
    assert np.all(np.isfinite(data.durations))
    assert np.all(data.durations > 0)


def test_zero_censor_fraction_marks_every_observation_as_event():
    data = generate_survival_data(500, censor_fraction=0.0, seed=2)
    assert bool(np.all(data.events))
    assert data.censor_fraction == 0.0


def test_exponential_mean_is_recovered_with_shape_one():
    data = generate_survival_data(40000, shape=1.0, scale=2.0, seed=3)
    assert float(np.mean(data.durations)) == pytest.approx(2.0, rel=0.08)


def test_weibull_median_matches_theoretical_value():
    data = generate_survival_data(40000, shape=2.0, scale=3.0, seed=4)
    empirical = float(np.median(data.durations))
    theoretical = 3.0 * np.log(2.0) ** 0.5
    assert empirical == pytest.approx(theoretical, rel=0.03)


def test_population_median_field_matches_the_latent_family():
    data = generate_survival_data(10, shape=1.5, scale=4.0, seed=7)
    assert data.population_median == pytest.approx(
        4.0 * np.log(2.0) ** (1.0 / 1.5), rel=1e-12
    )


def test_censor_calibration_hits_the_requested_share():
    data = generate_survival_data(6000, censor_fraction=0.3, seed=5)
    realized = float(np.mean(~data.events))
    assert 0.26 < realized < 0.34


def test_group_scale_ratio_shifts_the_treatment_arm_upward():
    data = generate_survival_data(8000, scale=1.0, group_scale_ratio=3.0, seed=6)
    control = data.durations[data.groups == "control"]
    treatment = data.durations[data.groups == "treatment"]
    assert control.size > 0
    assert treatment.size > 0
    assert float(np.median(control)) * 2.0 < float(np.median(treatment))


def test_arm_sizes_track_the_requested_fraction():
    data = generate_survival_data(10000, arm_fraction=0.25, seed=8)
    share = float(np.mean(data.groups == "treatment"))
    assert abs(share - 0.25) < 0.03


def test_csv_round_trip_preserves_every_column(tmp_path):
    data = generate_survival_data(60, censor_fraction=0.2, seed=9)
    path = tmp_path / "sample.csv"
    scores = np.arange(60, dtype=float)
    save_csv(data, path, extra_columns={"score": scores})
    durations, events, groups, extras = load_csv(str(path))
    assert np.allclose(durations, data.durations)
    assert np.array_equal(events, data.events)
    assert np.array_equal(groups.astype(str), data.groups.astype(str))
    assert np.allclose(extras["score"], scores)


def test_ph_generator_reproduces_with_the_same_seed():
    first = generate_ph_data(80, [0.4, -0.2], seed=5)
    second = generate_ph_data(80, [0.4, -0.2], seed=5)
    other = generate_ph_data(80, [0.4, -0.2], seed=6)
    assert np.array_equal(first.durations, second.durations)
    assert np.array_equal(first.covariates, second.covariates)
    assert not np.array_equal(first.durations, other.durations)


def test_ph_generator_shapes_and_zero_censoring():
    data = generate_ph_data(150, [0.3, -0.1, 0.2], shape=1.5, scale=2.0, seed=2)
    assert data.covariates.shape == (150, 3)
    assert data.durations.shape == (150,)
    assert bool(np.all(data.events))
    assert data.censor_fraction == 0.0
    assert np.allclose(data.coefficients, [0.3, -0.1, 0.2])


def test_ph_censor_calibration_hits_the_requested_share():
    data = generate_ph_data(4000, [0.5, -0.4], censor_fraction=0.3, seed=5)
    realized = float(np.mean(~data.events))
    assert 0.26 < realized < 0.34


def test_ph_supplied_covariates_are_honored():
    rng = np.random.default_rng(8)
    X = rng.normal(size=(40, 2))
    data = generate_ph_data(40, [0.2, -0.3], covariates=X, seed=8)
    assert np.allclose(data.covariates, X)


def test_ph_invalid_parameters_raise():
    with pytest.raises(ValueError):
        generate_ph_data(0, [0.1])
    with pytest.raises(ValueError):
        generate_ph_data(10, [0.1], shape=0.0)
    with pytest.raises(ValueError):
        generate_ph_data(10, [np.nan])
    with pytest.raises(ValueError):
        generate_ph_data(10, [0.1], covariates=np.zeros((9, 1)))


def test_invalid_parameters_raise():
    with pytest.raises(ValueError):
        generate_survival_data(0)
    with pytest.raises(ValueError):
        generate_survival_data(10, shape=0.0)
    with pytest.raises(ValueError):
        generate_survival_data(10, scale=-1.0)
    with pytest.raises(ValueError):
        generate_survival_data(10, censor_fraction=1.5)
    with pytest.raises(ValueError):
        generate_survival_data(10, arm_fraction=0.0)


def test_competing_risks_reproduces_with_the_same_seed():
    first = generate_competing_risks_data(80, seed=5)
    second = generate_competing_risks_data(80, seed=5)
    other = generate_competing_risks_data(80, seed=6)
    assert np.array_equal(first.durations, second.durations)
    assert np.array_equal(first.event_types, second.event_types)
    assert not np.array_equal(first.durations, other.durations)


def test_competing_risks_shapes_and_cause_codes():
    data = generate_competing_risks_data(200, cause_rates=(0.4, 0.3, 0.2), seed=1)
    assert data.durations.shape == (200,)
    assert data.event_types.dtype == int
    assert set(np.unique(data.event_types)) <= {0, 1, 2, 3}
    assert {1, 2, 3} <= set(np.unique(data.event_types))
    assert bool(np.all(data.events))
    assert data.censor_fraction == 0.0
    assert set(np.unique(data.groups)) == {"control", "treatment"}


def test_competing_risks_censor_calibration_hits_the_requested_share():
    data = generate_competing_risks_data(
        6000, cause_rates=(0.4, 0.3), censor_fraction=0.3, seed=5
    )
    realized = float(np.mean(data.event_types == 0))
    assert 0.26 < realized < 0.34


def test_competing_risks_rate_ratio_shifts_cause1_incidence():
    data = generate_competing_risks_data(
        8000,
        cause_rates=(0.5, 0.3),
        group_rate_ratios=(0.2, 1.0),
        seed=6,
    )
    control = data.true_cif(2.0, cause=1, group="control")
    treated = data.true_cif(2.0, cause=1, group="treatment")
    assert control > treated
    control_share = float(np.mean(data.event_types[data.groups == "control"] == 1))
    treated_share = float(np.mean(data.event_types[data.groups == "treatment"] == 1))
    assert control_share > treated_share


def test_exponential_competing_cif_closed_form():
    value = exponential_competing_cif(1.0, (0.4, 0.2), cause=1)
    expected = 0.4 / 0.6 * (1.0 - np.exp(-0.6))
    assert value == pytest.approx(expected)
    vector = exponential_competing_cif([0.0, 1.0], (0.4, 0.2), cause=2)
    assert vector[0] == pytest.approx(0.0)
    assert vector[1] == pytest.approx(0.2 / 0.6 * (1.0 - np.exp(-0.6)))


def test_competing_risks_csv_round_trip_preserves_cause_codes(tmp_path):
    data = generate_competing_risks_data(60, censor_fraction=0.2, seed=9)
    path = tmp_path / "cr.csv"
    save_csv(data, path)
    durations, events, groups, extras, event_types = load_csv(str(path), with_event_types=True)
    assert np.allclose(durations, data.durations)
    assert np.array_equal(event_types, data.event_types)
    assert np.array_equal(events, data.events)
    assert np.array_equal(groups.astype(str), data.groups.astype(str))
    assert extras == {}


def test_competing_risks_invalid_parameters_raise():
    with pytest.raises(ValueError):
        generate_competing_risks_data(0)
    with pytest.raises(ValueError):
        generate_competing_risks_data(10, cause_rates=(0.4,))
    with pytest.raises(ValueError):
        generate_competing_risks_data(10, cause_rates=(0.4, -0.1))
    with pytest.raises(ValueError):
        generate_competing_risks_data(10, group_rate_ratios=(1.0, 2.0, 3.0))
    with pytest.raises(ValueError):
        generate_competing_risks_data(10, censor_fraction=1.0)