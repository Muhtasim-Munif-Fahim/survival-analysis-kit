"""Tests for the Fine-Gray subdistribution hazard model."""

import csv

import numpy as np
import pytest
from scipy.optimize import minimize_scalar

from survival_kit.cli import main
from survival_kit.competing_risks import _fine_gray_weights
from survival_kit.cox import breslow_partial_likelihood, fit_cox_ph
from survival_kit.fine_gray import (
    _weight_matrix,
    fine_gray_partial_likelihood,
    fit_fine_gray,
)
from survival_kit.kaplan_meier import fit_kaplan_meier
from survival_kit.synth import generate_competing_risks_data, generate_ph_data, save_csv
from survival_kit.utils import step_value


# Cause 2 at 1, censored at 2, cause 1 at 3 (x=0) and 4 (x=1).
CENSORED = np.array([1.0, 2.0, 3.0, 4.0])
CENSORED_CAUSES = np.array([2, 0, 1, 1])
CENSORED_X = np.array([0.0, 1.0, 0.0, 1.0])


def _closed_censored_loglik(beta):
    """Hand-derived weighted log partial likelihood for CENSORED."""
    beta = float(beta)
    return -np.log(5.0 / 3.0 + np.exp(beta)) + beta - np.log(2.0 / 3.0 + np.exp(beta))


def _simulate_fine_gray(n, beta, p=0.5, censor_rate=0.0, seed=0):
    """Draw cause-1 times from a proportional subdistribution hazard model.

    The baseline CIF is ``F0(t) = p (1 - exp(-t))``, and
    ``1 - F(t | x) = (1 - F0(t)) ** exp(x beta)`` with ``x`` in ``{0, 1}``.
    Subjects who do not fail of cause 1 fail of cause 2 at an exponential time.
    """
    rng = np.random.default_rng(seed)
    x = rng.integers(0, 2, size=n).astype(float)
    u = rng.random(n)
    shrink = np.exp(x * beta)
    f_inf = -np.expm1(shrink * np.log1p(-p))
    cause1 = u < f_inf
    one_minus = np.clip(1.0 - u, 1e-16, 1.0)
    base = np.exp(np.log(one_minus) / shrink)
    arg = (base - (1.0 - p)) / p
    times = np.full(n, np.nan)
    valid = cause1 & (arg > 0.0)
    times[valid] = -np.log(np.clip(arg[valid], 1e-300, 1.0))
    competing = ~cause1
    times[competing] = rng.exponential(1.0, size=int(np.count_nonzero(competing)))
    event_types = np.where(cause1, 1, 2).astype(int)
    if censor_rate > 0.0:
        censoring = rng.exponential(1.0 / censor_rate, size=n)
        censored = censoring < times
        event_types = np.where(censored, 0, event_types)
        times = np.minimum(times, censoring)
    return times, event_types, x


def test_ipcw_weights_match_gray_risk_set_and_the_hand_table():
    curve = fit_kaplan_meier(CENSORED, CENSORED_CAUSES == 0)
    times = np.array([3.0, 4.0])
    weights = _weight_matrix(CENSORED, CENSORED_CAUSES, 1, times, curve)
    assert np.allclose(weights[0], [2.0 / 3.0, 0.0, 1.0, 1.0])
    assert np.allclose(weights[1], [2.0 / 3.0, 0.0, 0.0, 1.0])
    for row, t in enumerate(times):
        assert np.allclose(
            weights[row],
            _fine_gray_weights(CENSORED, CENSORED_CAUSES, t, 1, curve),
        )


def test_censored_partial_likelihood_matches_hand_computation():
    loglik, score, information, times, increments = fine_gray_partial_likelihood(
        [0.0], CENSORED, CENSORED_CAUSES, CENSORED_X, cause=1
    )
    assert loglik == pytest.approx(-np.log(40.0 / 9.0))
    assert score[0] == pytest.approx(1.0 / 40.0)
    assert information[0, 0] == pytest.approx(759.0 / 1600.0)
    assert np.allclose(times, [3.0, 4.0])
    assert increments[0] == pytest.approx(3.0 / 8.0)
    assert increments[1] == pytest.approx(3.0 / 5.0)


def test_uncensored_partial_likelihood_keeps_competing_events_in_the_risk_set():
    durations = np.array([1.0, 2.0, 3.0])
    causes = np.array([1, 2, 1])
    x = np.array([0.0, 1.0, 0.0])
    loglik, score, information, times, increments = fine_gray_partial_likelihood(
        [0.0], durations, causes, x, cause=1
    )
    assert loglik == pytest.approx(-np.log(6.0))
    assert score[0] == pytest.approx(-5.0 / 6.0)
    assert information[0, 0] == pytest.approx(17.0 / 36.0)
    assert np.allclose(times, [1.0, 3.0])
    assert increments[0] == pytest.approx(1.0 / 3.0)
    assert increments[1] == pytest.approx(1.0 / 2.0)


def test_tied_cause_events_use_the_breslow_denominator():
    durations = np.array([1.0, 1.0, 2.0])
    causes = np.array([1, 1, 2])
    x = np.array([0.0, 1.0, 0.0])
    loglik, score, information, _, _ = fine_gray_partial_likelihood(
        [0.0], durations, causes, x, cause=1
    )
    assert loglik == pytest.approx(-2.0 * np.log(3.0))
    assert score[0] == pytest.approx(1.0 / 3.0)
    assert information[0, 0] == pytest.approx(4.0 / 9.0)


def test_null_score_matches_the_gray_score_accumulation():
    durations = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 4.0])
    event_types = np.array([1, 2, 0, 1, 2, 1])
    groups = np.array(["a", "a", "b", "b", "a", "b"])
    labels, index = np.unique(groups, return_inverse=True)
    design = np.eye(labels.size, dtype=float)[index][:, : labels.size - 1]
    _, score, _, _, _ = fine_gray_partial_likelihood(
        np.zeros(design.shape[1]), durations, event_types, design, cause=1
    )

    curve = fit_kaplan_meier(durations, event_types == 0)
    manual = np.zeros(design.shape[1])
    for t in np.unique(durations[event_types == 1]):
        at_event = (durations == t) & (event_types == 1)
        deaths = int(np.count_nonzero(at_event))
        weights = _fine_gray_weights(durations, event_types, float(t), 1, curve)
        total = float(weights.sum())
        xbar = (weights.reshape(-1, 1) * design).sum(axis=0) / total
        manual += design[at_event].sum(axis=0) - deaths * xbar
    assert np.allclose(score, manual)


def test_mle_matches_the_scalar_maximizer_on_the_censored_table():
    closed = minimize_scalar(
        lambda beta: -_closed_censored_loglik(beta), bounds=(-5.0, 5.0), method="bounded"
    )
    fit = fit_fine_gray(CENSORED, CENSORED_CAUSES, CENSORED_X, cause=1)
    assert fit.converged
    assert fit.cause == 1
    assert fit.n_events == 2
    assert fit.coefficients[0] == pytest.approx(closed.x, abs=1e-8)
    assert fit.log_partial_likelihood == pytest.approx(-closed.fun, rel=1e-8)
    _, score, _, _, _ = fine_gray_partial_likelihood(
        fit.coefficients, CENSORED, CENSORED_CAUSES, CENSORED_X, cause=1
    )
    assert score[0] == pytest.approx(0.0, abs=1e-8)


def test_score_and_information_match_central_differences():
    rng = np.random.default_rng(4)
    durations = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    event_types = np.array([1, 2, 0, 1, 2, 1, 0])
    X = rng.normal(size=(7, 2))
    beta = np.array([0.3, -0.4])
    _, score, information, _, _ = fine_gray_partial_likelihood(
        beta, durations, event_types, X, cause=1
    )
    eps = 1e-5
    numeric_score = np.empty(2)
    numeric_info = np.empty((2, 2))
    for j in range(2):
        bump = np.zeros(2)
        bump[j] = eps
        plus, score_plus, _, _, _ = fine_gray_partial_likelihood(
            beta + bump, durations, event_types, X, cause=1
        )
        minus, score_minus, _, _, _ = fine_gray_partial_likelihood(
            beta - bump, durations, event_types, X, cause=1
        )
        numeric_score[j] = (plus - minus) / (2.0 * eps)
        numeric_info[:, j] = (score_plus - score_minus) / (2.0 * eps)
    assert np.allclose(score, numeric_score, rtol=1e-4, atol=1e-5)
    assert np.allclose(-information, numeric_info, rtol=1e-4, atol=1e-4)


def test_single_cause_fit_matches_cox():
    data = generate_ph_data(350, [0.4, -0.25], censor_fraction=0.2, seed=5)
    event_types = data.events.astype(int)
    fine_gray = fit_fine_gray(data.durations, event_types, data.covariates, cause=1)
    cox = fit_cox_ph(data.durations, data.events, data.covariates)
    assert np.allclose(fine_gray.coefficients, cox.coefficients, atol=1e-8)
    assert np.allclose(fine_gray.std_err, cox.std_err, atol=1e-8)
    assert fine_gray.log_partial_likelihood == pytest.approx(cox.log_partial_likelihood)
    assert np.allclose(fine_gray.baseline_time, cox.baseline_time)
    assert np.allclose(fine_gray.baseline_cumhazard, cox.baseline_cumhazard, atol=1e-8)
    assert fine_gray.n_events == cox.n_events


def test_uncensored_competing_events_match_cox_with_an_extended_risk_set():
    rng = np.random.default_rng(9)
    x = rng.normal(size=180)
    latent_cause = rng.exponential(np.exp(-0.4 * x))
    latent_other = rng.exponential(1.2, size=x.size)
    cause1 = latent_cause <= latent_other
    durations = np.minimum(latent_cause, latent_other)
    event_types = np.where(cause1, 1, 2)
    fine_gray = fit_fine_gray(durations, event_types, x, cause=1)
    horizon = float(durations[cause1].max())
    modified = durations.copy()
    modified[~cause1] = np.maximum(modified[~cause1], horizon)
    cox = fit_cox_ph(modified, cause1, x)
    assert np.allclose(fine_gray.coefficients, cox.coefficients, atol=1e-7)
    assert np.allclose(fine_gray.baseline_cumhazard, cox.baseline_cumhazard, atol=1e-7)
    assert fine_gray.log_partial_likelihood == pytest.approx(cox.log_partial_likelihood)


def test_censoring_weights_differ_from_cause_specific_and_unit_weight_cox():
    x = CENSORED_X.reshape(-1, 1)
    fine_gray, _, _, _, _ = fine_gray_partial_likelihood(
        [0.0], CENSORED, CENSORED_CAUSES, x, cause=1
    )
    cause_specific, _, _, _, _ = breslow_partial_likelihood(
        [0.0], CENSORED, CENSORED_CAUSES == 1, x
    )
    horizon = float(CENSORED[CENSORED_CAUSES == 1].max())
    extended = CENSORED.copy()
    extended[CENSORED_CAUSES == 2] = horizon
    unit_weight, _, _, _, _ = breslow_partial_likelihood(
        [0.0], extended, CENSORED_CAUSES == 1, x
    )
    assert fine_gray != pytest.approx(cause_specific)
    assert fine_gray != pytest.approx(unit_weight)


def test_recovers_a_known_subdistribution_hazard_ratio():
    beta = 0.6
    durations, event_types, x = _simulate_fine_gray(5000, beta, p=0.45, seed=11)
    fit = fit_fine_gray(durations, event_types, x, cause=1)
    assert fit.coefficients[0] == pytest.approx(beta, abs=0.15)
    assert abs(fit.coefficients[0] - beta) < 3.0 * fit.std_err[0]
    assert fit.ci_lower[0] < fit.coefficients[0] < fit.ci_upper[0]


def test_recovers_the_coefficient_under_independent_censoring():
    beta = 0.55
    durations, event_types, x = _simulate_fine_gray(
        5000, beta, p=0.5, censor_rate=0.25, seed=13
    )
    assert 0.05 < np.mean(event_types == 0) < 0.45
    fit = fit_fine_gray(durations, event_types, x, cause=1)
    assert fit.coefficients[0] == pytest.approx(beta, abs=0.2)
    assert abs(fit.coefficients[0] - beta) < 3.5 * fit.std_err[0]


def test_null_covariate_is_not_significant():
    durations, event_types, x = _simulate_fine_gray(2500, beta=0.0, p=0.4, seed=17)
    fit = fit_fine_gray(durations, event_types, x, cause=1)
    assert fit.p_values[0] > 0.01
    assert fit.likelihood_ratio_p_value > 0.01


def test_null_baseline_incidence_tracks_the_known_cif():
    durations, event_types, x = _simulate_fine_gray(8000, beta=0.0, p=0.4, seed=19)
    _, _, _, times, increments = fine_gray_partial_likelihood(
        [0.0], durations, event_types, x, cause=1
    )
    query = np.array([0.25, 0.5, 1.0, 1.5])
    cumulative = step_value(times, np.cumsum(increments), query, outside=0.0)
    expected = 0.4 * (1.0 - np.exp(-query))
    assert np.allclose(1.0 - np.exp(-cumulative), expected, atol=0.03)


def test_predicted_incidence_is_a_unit_interval_step():
    fit = fit_fine_gray(CENSORED, CENSORED_CAUSES, CENSORED_X, cause=1)
    assert np.all(np.diff(fit.baseline_cumhazard) >= -1e-12)
    assert fit.cumulative_incidence_at(0.0, [0.0]) == pytest.approx(0.0)
    assert fit.baseline_at(fit.baseline_time[0] - 1e-9) == 0.0
    assert fit.baseline_at(fit.baseline_time[0]) == pytest.approx(fit.baseline_cumhazard[0])
    assert fit.coefficients[0] > 0.0
    early = fit.cumulative_incidence_at(fit.baseline_time, [0.0])
    late = fit.cumulative_incidence_at(fit.baseline_time, [2.0])
    assert np.all((early >= 0.0) & (early <= 1.0))
    assert np.all(np.diff(early) >= -1e-12)
    assert np.all(late >= early - 1e-12)
    assert fit.relative_subdistribution_hazard([0.0])[0] == pytest.approx(1.0)
    assert np.allclose(
        fit.cumulative_incidence_at(fit.baseline_time, [0.0]),
        1.0 - np.exp(-fit.baseline_cumhazard),
    )


def test_subdistribution_hazard_ratios_are_exp_coefficients():
    durations, event_types, x = _simulate_fine_gray(400, beta=0.4, seed=23)
    fit = fit_fine_gray(durations, event_types, x, cause=1, feature_names=("arm",))
    assert fit.feature_names == ("arm",)
    assert np.allclose(fit.subdistribution_hazard_ratios, np.exp(fit.coefficients))
    assert np.allclose(
        fit.subdistribution_hazard_ratio_ci_lower, np.exp(fit.ci_lower)
    )
    assert np.allclose(
        fit.subdistribution_hazard_ratio_ci_upper, np.exp(fit.ci_upper)
    )


def test_higher_confidence_level_never_narrows_the_interval():
    durations, event_types, x = _simulate_fine_gray(300, beta=0.3, seed=29)
    narrow = fit_fine_gray(durations, event_types, x, confidence_level=0.90)
    wide = fit_fine_gray(durations, event_types, x, confidence_level=0.99)
    assert np.all(wide.ci_upper - wide.ci_lower >= narrow.ci_upper - narrow.ci_lower - 1e-15)


def test_invalid_inputs_raise():
    with pytest.raises(ValueError, match="requested cause"):
        fit_fine_gray([1.0, 2.0], [2, 0], [0.0, 1.0], cause=1)
    with pytest.raises(ValueError, match="cause must be a positive integer"):
        fit_fine_gray([1.0, 2.0], [1, 2], [0.0, 1.0], cause=0)
    with pytest.raises(ValueError, match="vary"):
        fit_fine_gray([1.0, 2.0, 3.0], [1, 1, 2], [1.0, 1.0, 1.0], cause=1)
    with pytest.raises(ValueError, match="collinear"):
        X = np.column_stack([np.arange(6.0), 2.0 * np.arange(6.0)])
        fit_fine_gray(np.arange(1.0, 7.0), np.array([1, 2, 1, 2, 1, 1]), X, cause=1)
    with pytest.raises(ValueError, match="finite"):
        fit_fine_gray([1.0, 2.0], [1, 1], [0.0, np.nan], cause=1)
    with pytest.raises(ValueError, match="feature_names"):
        fit_fine_gray(
            [1.0, 2.0, 3.0], [1, 2, 1], [0.0, 1.0, 0.0], cause=1, feature_names=("a", "b")
        )
    with pytest.raises(ValueError, match="confidence_level"):
        fit_fine_gray([1.0, 2.0, 3.0], [1, 1, 2], [0.0, 1.0, 0.0], confidence_level=1.0)
    with pytest.raises(ValueError):
        fit_fine_gray([1.0, 2.0], [1], [0.0, 1.0], cause=1)
    with pytest.raises(ValueError, match="integer cause"):
        fine_gray_partial_likelihood([0.0], [1.0, 2.0], [1.5, 2], [0.0, 1.0], cause=1)
    fit = fit_fine_gray(CENSORED, CENSORED_CAUSES, CENSORED_X, cause=1)
    with pytest.raises(ValueError, match="single observation"):
        fit.cumulative_incidence_at([1.0, 2.0], [[0.0], [1.0]])


def test_finegray_command_writes_subdistribution_hazard_ratios(tmp_path, capsys):
    data = generate_competing_risks_data(
        700,
        cause_rates=(0.6, 0.25),
        group_rate_ratios=(0.35, 1.0),
        censor_fraction=0.15,
        seed=7,
    )
    data_path = tmp_path / "cr.csv"
    out_path = tmp_path / "finegray.csv"
    save_csv(data, data_path)
    assert (
        main(
            [
                "finegray",
                "--data", str(data_path),
                "--group-col", "group",
                "--cause", "1",
                "--out", str(out_path),
            ]
        )
        == 0
    )
    text = capsys.readouterr().out
    assert "cause 1:" in text
    assert "SHR=" in text
    assert "group[treatment]" in text
    with open(out_path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["covariate"] == "group[treatment]"
    assert float(rows[0]["subdistribution_hazard_ratio"]) < 1.0


def test_finegray_command_requires_a_design_matrix():
    with pytest.raises(SystemExit):
        main(["finegray", "--data", "missing.csv", "--out", "finegray.csv"])
