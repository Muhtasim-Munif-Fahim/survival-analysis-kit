"""Tests for the Turnbull NPMLE on interval-censored data."""

from __future__ import annotations

import csv

import numpy as np
import pytest

from survival_kit import fit_turnbull, interval_censor, turnbull_intervals
from survival_kit.cli import main
from survival_kit.kaplan_meier import fit_kaplan_meier
from survival_kit.synth import generate_survival_data


def test_innermost_intervals_of_textbook_example():
    left = [0, 0, 2, 3, 1, 4]
    right = [2, 3, np.inf, 5, 4, np.inf]
    lo, hi, closed, alpha = turnbull_intervals(left, right)
    assert lo.tolist() == [1.0, 2.0, 3.0, 4.0]
    assert hi.tolist() == [2.0, 3.0, 4.0, 5.0]
    assert not closed.any()
    # (0, 2] only contains (1, 2]; (4, inf] only contains (4, 5].
    assert alpha[0].tolist() == [True, False, False, False]
    assert alpha[5].tolist() == [False, False, False, True]
    assert alpha[4].tolist() == [True, True, True, False]


def test_npmle_of_textbook_example_reaches_the_known_optimum():
    result = fit_turnbull([0, 0, 2, 3, 1, 4], [2, 3, np.inf, 5, 4, np.inf])
    # Masses 1/2 on (1, 2] and (4, 5] give every observation likelihood 1/2.
    assert result.probabilities == pytest.approx([0.5, 0.0, 0.0, 0.5], abs=1e-8)
    assert result.log_likelihood == pytest.approx(6 * np.log(0.5), abs=1e-8)
    assert result.max_gradient == pytest.approx(1.0, abs=1e-6)
    assert result.converged
    assert (result.n_left_censored, result.n_interval_censored, result.n_right_censored) == (2, 2, 2)


def test_exact_and_right_censored_data_reproduce_kaplan_meier():
    data = generate_survival_data(120, censor_fraction=0.35, seed=3)
    durations = np.round(data.durations, 2) + 0.01  # create ties on purpose
    events = data.events
    left = durations.copy()
    right = np.where(events, durations, np.inf)
    result = fit_turnbull(left, right, tol=1e-12)
    curve = fit_kaplan_meier(durations, events)
    grid = np.linspace(0.0, durations.max() + 1.0, 200)
    np.testing.assert_allclose(result.survival_at(grid), curve.at(grid), atol=1e-8)
    # The lower envelope drops mass on the unbounded tail (c_max, inf] right
    # after the last follow-up, where KM is undefined; agree up to there.
    inside = grid <= durations.max()
    np.testing.assert_allclose(
        result.survival_at(grid[inside], envelope="lower"), curve.at(grid[inside]), atol=1e-8
    )
    assert result.n_exact == int(events.sum())
    tail = ~np.isfinite(result.right)
    assert result.left_closed[~tail].all()
    assert result.mass_beyond_last_finite() == pytest.approx(curve.survival[-1], abs=1e-8)


def test_tied_exact_and_censored_times_follow_km_convention():
    # Censoring at t=2 is treated as surviving past an event at t=2, as in KM.
    left = np.array([1.0, 2.0, 2.0, 3.0])
    right = np.array([1.0, 2.0, np.inf, 3.0])
    result = fit_turnbull(left, right, tol=1e-12)
    curve = fit_kaplan_meier(left, np.isfinite(right))
    np.testing.assert_allclose(result.survival_at([1.0, 2.0, 2.5, 3.0]), curve.at([1.0, 2.0, 2.5, 3.0]), atol=1e-9)


def test_interval_censor_from_shared_schedule():
    durations = np.array([0.3, 1.0, 1.7, 9.0])
    left, right = interval_censor(durations, [0.5, 1.0, 1.5, 2.0])
    assert left.tolist() == [0.0, 0.5, 1.5, 2.0]
    assert right.tolist() == [0.5, 1.0, 2.0, np.inf]


def test_interval_censor_with_censored_subjects_and_ragged_schedules():
    durations = np.array([1.2, 2.5, 0.4])
    events = np.array([1, 0, 1])
    schedules = [[1.0, 2.0], [0.7, 1.4, 2.1, 2.8], [0.5]]
    left, right = interval_censor(durations, schedules, events=events)
    assert left.tolist() == [1.0, 2.1, 0.0]
    assert right.tolist() == [2.0, np.inf, 0.5]


def test_npmle_recovers_weibull_survival_from_periodic_inspection():
    rng = np.random.default_rng(0)
    durations = rng.weibull(1.5, 1500) * 2.0
    grid = np.arange(0.5, 5.01, 0.5)
    left, right = interval_censor(durations, grid)
    result = fit_turnbull(left, right)
    truth = np.exp(-((grid[:-1] / 2.0) ** 1.5))
    np.testing.assert_allclose(result.survival_at(grid[:-1]), truth, atol=0.03)
    # With a shared grid the inspection times are the only identified points,
    # and both envelopes agree there.
    np.testing.assert_allclose(
        result.survival_at(grid[:-1]), result.survival_at(grid[:-1], envelope="lower"), atol=1e-12
    )
    assert result.median() == pytest.approx(2.0 * np.log(2.0) ** (1 / 1.5), abs=0.5)


def test_envelopes_bracket_survival_inside_innermost_intervals():
    result = fit_turnbull([0.0, 1.0, 2.0], [3.0, 4.0, 5.0])
    # Single innermost interval (2, 3] carries all mass.
    assert result.n_intervals == 1
    assert result.survival_at(2.5) == pytest.approx(1.0)
    assert result.survival_at(2.5, envelope="lower") == pytest.approx(0.0)
    assert result.survival_at(3.0) == pytest.approx(0.0)


def test_weights_act_as_frequencies():
    left = np.array([0.0, 1.0, 2.0, 1.0])
    right = np.array([1.0, 3.0, np.inf, 2.0])
    weights = np.array([2.0, 1.0, 3.0, 1.0])
    weighted = fit_turnbull(left, right, weights=weights, tol=1e-12)
    expanded = fit_turnbull(np.repeat(left, 2 * weights.astype(int)), np.repeat(right, 2 * weights.astype(int)), tol=1e-12)
    np.testing.assert_allclose(weighted.probabilities, expanded.probabilities, atol=1e-8)


def test_likelihood_beats_any_perturbation():
    rng = np.random.default_rng(5)
    durations = rng.exponential(1.0, 80)
    schedules = [np.cumsum(rng.uniform(0.2, 0.8, 6)) for _ in range(80)]
    left, right = interval_censor(durations, schedules)
    result = fit_turnbull(left, right, tol=1e-11)
    _, _, _, alpha = turnbull_intervals(left, right)
    a = alpha.astype(float)
    for _ in range(20):
        noise = rng.dirichlet(np.ones(result.n_intervals))
        other = 0.9 * result.probabilities + 0.1 * noise
        assert np.sum(np.log(a @ other)) <= result.log_likelihood + 1e-9


@pytest.mark.parametrize(
    "left,right,message",
    [
        ([1.0], [0.5], ">= its left"),
        ([-1.0], [1.0], "non-negative"),
        ([0.0], [0.0], "time 0"),
        ([1.0], [np.nan], "NaN"),
        ([1.0, 2.0], [3.0], "same length"),
    ],
)
def test_validation(left, right, message):
    with pytest.raises(ValueError, match=message):
        fit_turnbull(left, right)


def test_cli_writes_npmle_table(tmp_path, capsys):
    path = tmp_path / "ic.csv"
    rows = [
        ("0", "2", "a"), ("0", "3", "a"), ("2", "inf", "a"), ("3", "5", "a"),
        ("1", "4", "a"), ("4", "", "a"), ("1", "2", "b"), ("2", "3", "b"),
    ]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        handle.write("left,right,arm\n")
        for row in rows:
            handle.write(",".join(row) + "\n")
    out = tmp_path / "npmle.csv"
    assert main(["turnbull", "--data", str(path), "--group-col", "arm", "--out", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "a: n=6" in printed and "b: n=2" in printed
    with open(out, newline="", encoding="utf-8") as handle:
        table = list(csv.DictReader(handle))
    arm_a = [r for r in table if r["group"] == "a"]
    assert sum(float(r["probability"]) for r in arm_a) == pytest.approx(1.0)
    assert [float(r["probability"]) for r in arm_a] == pytest.approx([0.5, 0, 0, 0.5], abs=1e-8)
    assert float(arm_a[-1]["survival_upper"]) == pytest.approx(0.0)


def test_cli_reports_missing_column(tmp_path):
    path = tmp_path / "ic.csv"
    path.write_text("lo,hi\n0,1\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="missing column: left"):
        main(["turnbull", "--data", str(path), "--out", str(tmp_path / "o.csv")])
