"""Tests for Cox residuals and the Grambsch-Therneau PH test."""

import numpy as np
import pytest

from survival_kit.cli import main
from survival_kit.cox import fit_cox_ph
from survival_kit.cox_diagnostics import (
    TIME_TRANSFORMS,
    deviance_residuals,
    martingale_residuals,
    proportional_hazards_test,
    schoenfeld_residuals,
    transform_event_times,
)
from survival_kit.synth import generate_ph_data


@pytest.fixture(scope="module")
def ph_sample():
    data = generate_ph_data(300, [0.7, -0.4], censor_fraction=0.3, seed=3)
    fit = fit_cox_ph(data.durations, data.events, data.covariates)
    return data, fit


def _non_ph_sample(n=600, seed=0, change=0.5):
    """Binary covariate with HR 3 before ``change`` and HR 1/3 afterwards."""
    rng = np.random.default_rng(seed)
    group = rng.integers(0, 2, size=n).astype(float)
    hr_early = np.where(group == 1, 3.0, 1.0)
    hr_late = np.where(group == 1, 1.0 / 3.0, 1.0)
    e = rng.exponential(size=n)
    # Base rate 1: H(t) = hr_early * min(t, c) + hr_late * max(t - c, 0).
    h_change = hr_early * change
    t = np.where(e < h_change, e / hr_early, change + (e - h_change) / hr_late)
    censor = rng.exponential(4.0, size=n)
    durations = np.minimum(t, censor)
    events = t <= censor
    return durations, events, group


# Reference values from lifelines 0.30.3 (CoxPHFitter + proportional_hazard_test)
# on ph_sample; there are no tied times, so Efron and Breslow coincide.
LIFELINES_STATS = {
    "rank": (0.39650461, 0.14387949),
    "identity": (0.26726856, 0.47908664),
    "log": (0.02342696, 0.19477214),
}


@pytest.mark.parametrize("transform", sorted(LIFELINES_STATS))
def test_per_variable_statistics_match_lifelines(ph_sample, transform):
    data, fit = ph_sample
    result = proportional_hazards_test(
        fit, data.durations, data.events, data.covariates, transform=transform
    )
    assert np.allclose(result.statistics, LIFELINES_STATS[transform], rtol=2e-4)


def test_martingale_residuals_match_lifelines_spot_values(ph_sample):
    data, fit = ph_sample
    m = martingale_residuals(fit, data.durations, data.events, data.covariates)
    assert m.shape == (300,)
    # Breslow martingale residuals sum to zero at the MLE.
    assert m.sum() == pytest.approx(0.0, abs=1e-8)
    assert np.all(m <= 1.0)
    # lifelines 0.30.3 compute_residuals(..., "martingale"/"deviance"), first five rows.
    assert np.allclose(m[:5], [0.17822889, -1.4957108, 0.4540211, -0.07075764, -0.02176445], atol=1e-4)
    d = deviance_residuals(fit, data.durations, data.events, data.covariates)
    assert np.allclose(d[:5], [0.19007624, -1.72957266, 0.54982515, -0.37618515, -0.20863581], atol=1e-4)


def test_schoenfeld_residuals_solve_the_score_equations(ph_sample):
    data, fit = ph_sample
    res = schoenfeld_residuals(fit, data.durations, data.events, data.covariates)
    assert res.residuals.shape == (fit.n_events, 2)
    assert np.allclose(res.residuals.sum(axis=0), 0.0, atol=1e-6)
    assert np.all(np.diff(res.times) >= 0)
    assert np.all(data.events[res.index])
    # Scaled residuals average to beta.
    assert np.allclose(res.scaled.mean(axis=0), fit.coefficients, atol=1e-6)
    assert res.feature_names == ("x0", "x1")


def test_schoenfeld_hand_computation_without_ties():
    durations = np.array([1.0, 2.0, 3.0, 4.0])
    events = np.array([True, True, False, True])
    x = np.array([1.0, 0.0, 1.0, 0.0])
    fit = fit_cox_ph(durations, events, x)
    b = fit.coefficients[0]
    w = np.exp(b * x)
    expected = [
        x[0] - (w @ x) / w.sum(),
        x[1] - (w[1:] @ x[1:]) / w[1:].sum(),
        x[3] - x[3],
    ]
    res = schoenfeld_residuals(fit, durations, events, x)
    assert np.allclose(res.residuals[:, 0], expected)
    assert list(res.index) == [0, 1, 3]


def test_tied_deaths_share_the_risk_set_mean():
    durations = np.array([1.0, 1.0, 2.0, 3.0, 3.0, 4.0])
    events = np.array([True, True, True, False, True, True])
    x = np.array([0.0, 1.0, 1.0, 0.0, 1.0, 0.0])
    fit = fit_cox_ph(durations, events, x)
    res = schoenfeld_residuals(fit, durations, events, x)
    first_two = res.residuals[:2, 0]
    assert first_two[1] - first_two[0] == pytest.approx(1.0)
    assert np.allclose(res.residuals.sum(axis=0), 0.0, atol=1e-6)


def test_deviance_residuals_are_signed_martingale_transform(ph_sample):
    data, fit = ph_sample
    m = martingale_residuals(fit, data.durations, data.events, data.covariates)
    d = deviance_residuals(fit, data.durations, data.events, data.covariates)
    assert np.all(np.sign(d) == np.sign(m))
    censored = ~data.events
    assert np.allclose(d[censored], -np.sqrt(-2.0 * m[censored]))
    # Deviance residuals are more symmetric than martingale residuals.
    from scipy.stats import skew

    assert abs(skew(d)) < abs(skew(m))


def test_detects_crossing_hazards():
    durations, events, group = _non_ph_sample()
    fit = fit_cox_ph(durations, events, group, feature_names=("group",))
    for transform in TIME_TRANSFORMS:
        result = proportional_hazards_test(fit, durations, events, group, transform=transform)
        assert result.p_values[0] < 1e-6
        assert result.correlations[0] < 0  # effect decreases over time
        assert result.violations() == ("group",)


def test_ph_data_rarely_rejected():
    rejections = 0
    for seed in range(20):
        data = generate_ph_data(200, [0.5, -0.5], censor_fraction=0.2, seed=seed)
        fit = fit_cox_ph(data.durations, data.events, data.covariates)
        result = proportional_hazards_test(fit, data.durations, data.events, data.covariates)
        rejections += result.global_p_value < 0.05
    assert rejections <= 4


def test_global_statistic_equals_per_variable_with_one_covariate():
    durations, events, group = _non_ph_sample(n=300, seed=4)
    fit = fit_cox_ph(durations, events, group)
    result = proportional_hazards_test(fit, durations, events, group, transform="rank")
    assert result.global_statistic == pytest.approx(result.statistics[0])
    assert result.global_df == 1
    rows = result.summary_rows()
    assert rows[-1][0] == "GLOBAL"
    assert rows[0][0] == "x0"


def test_km_transform_is_left_continuous():
    durations = np.array([1.0, 2.0, 3.0, 4.0])
    events = np.array([True, True, True, True])
    g = transform_event_times(durations, durations, events, "km")
    assert np.allclose(g, [0.0, 0.25, 0.5, 0.75])
    assert np.allclose(transform_event_times(durations, durations, events, "rank"), [1, 2, 3, 4])
    assert np.allclose(transform_event_times(durations, durations, events, "log"), np.log(durations))
    with pytest.raises(ValueError):
        transform_event_times(durations, durations, events, "sqrt")
    with pytest.raises(ValueError):
        transform_event_times(np.array([0.0, 1.0]), durations, events, "log")


def test_mismatched_data_rejected(ph_sample):
    data, fit = ph_sample
    with pytest.raises(ValueError):
        proportional_hazards_test(fit, data.durations[:-1], data.events[:-1], data.covariates[:-1])
    with pytest.raises(ValueError):
        proportional_hazards_test(fit, data.durations, data.events, data.covariates, transform="x")


def test_cli_cox_zph(tmp_path, capsys):
    durations, events, group = _non_ph_sample(n=400, seed=1)
    path = tmp_path / "nonph.csv"
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("duration,event,dose\n")
        for t, e, x in zip(durations, events, group):
            handle.write(f"{t},{int(e)},{x}\n")
    out_path = tmp_path / "zph.csv"
    assert main(
        [
            "cox-zph", "--data", str(path), "--covariate-cols", "dose",
            "--transform", "rank", "--out", str(out_path),
        ]
    ) == 0
    out = capsys.readouterr().out
    assert "GLOBAL" in out
    assert "violates PH" in out
    lines = out_path.read_text(encoding="utf-8").strip().splitlines()
    assert lines[0] == "covariate,rho,chisq,df,p_value"
    assert lines[1].startswith("dose,")
    assert lines[2].startswith("GLOBAL,")
