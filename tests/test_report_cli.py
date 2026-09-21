"""Integration tests for the markdown renderer and command-line interface."""

import csv

import numpy as np
import pytest

from survival_kit.cli import main
from survival_kit.concordance import concordance_index
from survival_kit.cox import fit_cox_ph
from survival_kit.kaplan_meier import fit_kaplan_meier
from survival_kit.logrank import log_rank_test_groups
from survival_kit.report import CohortSummary, format_p_value, render_report
from survival_kit.synth import generate_ph_data, generate_survival_data, save_csv


def test_p_value_formatting_uses_a_floor():
    assert format_p_value(0.0) == "<0.001"
    assert format_p_value(1e-4) == "<0.001"
    assert format_p_value(0.0327) == "0.033"


def _summary(data, label):
    mask = data.groups == label
    curve = fit_kaplan_meier(data.durations[mask], data.events[mask])
    return CohortSummary(label, curve, int(mask.sum()), int(data.events[mask].sum()))


def test_report_renders_cohort_tables_and_log_rank_section():
    data = generate_survival_data(240, group_scale_ratio=2.5, seed=11)
    cohorts = [_summary(data, "control"), _summary(data, "treatment")]
    result = log_rank_test_groups(data.durations, data.events, data.groups)

    text = render_report(
        "Demo", cohorts, log_rank=result, log_rank_labels=["control", "treatment"]
    )
    assert text.startswith("# Demo")
    assert "## Overview" in text and "## Cohorts" in text
    assert "### control" in text and "### treatment" in text
    assert "## Log-rank test" in text
    assert "p-value: <0.001" in text
    assert "- Median survival:" in text
    rows = [
        line
        for line in text.splitlines()
        if line.startswith("| ") and "---" not in line
    ]
    assert len(rows) > 4
    assert len({row.count("|") for row in rows}) == 1


def test_concordance_section_is_optional_and_well_formed():
    data = generate_survival_data(120, seed=15)
    rng = np.random.default_rng(16)
    result = concordance_index(data.durations, data.events, rng.normal(size=120))
    single = CohortSummary("all", fit_kaplan_meier(data.durations, data.events), 120, int(data.events.sum()))

    plain = render_report("T", [single])
    assert "## Concordance" not in plain and "## Log-rank test" not in plain

    scored = render_report("T", [single], concordance=result)
    assert "Harrell's C:" in scored
    assert "Comparable pairs:" in scored


def test_horizon_grid_starts_at_zero_and_matches_quantiles():
    data = generate_survival_data(90, seed=17)
    from survival_kit.report import default_horizons

    horizons = default_horizons([_summary(data, "control")])
    assert horizons[0] == 0.0
    assert all(b > a for a, b in zip(horizons, horizons[1:]))


def test_generate_then_fit_round_trip(tmp_path):
    data_path = tmp_path / "data.csv"
    curve_path = tmp_path / "curves.csv"

    assert main(["generate", "--n", "80", "--seed", "1", "--out", str(data_path)]) == 0
    with open(data_path, newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    assert rows[0] == ["duration", "event", "group"]
    assert len(rows) == 81

    assert (
        main(["fit", "--data", str(data_path), "--out", str(curve_path), "--group-col", "group"])
        == 0
    )
    with open(curve_path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert {"control", "treatment"} <= {row["cohort"] for row in rows}
    for name in ("control", "treatment"):
        series = [
            float(row["survival"]) for row in rows if row["cohort"] == name
        ]
        assert series
        assert all(0.0 <= s <= 1.0 for s in series)
        assert all(b <= a + 1e-12 for a, b in zip(series, series[1:]))


def test_compare_reports_separation_between_arms(tmp_path, capsys):
    data_path = tmp_path / "separated.csv"
    main(["generate", "--n", "300", "--group-scale-ratio", "3", "--seed", "2", "--out", str(data_path)])
    assert main(["compare", "--data", str(data_path), "--group-col", "group"]) == 0
    out = capsys.readouterr().out
    assert "chi-square(1)" in out
    assert "p-value" in out
    assert "control: observed" in out


def test_report_command_writes_markdown_with_log_rank(tmp_path):
    data_path = tmp_path / "data.csv"
    report_path = tmp_path / "report.md"
    main(["generate", "--n", "150", "--seed", "4", "--out", str(data_path)])
    code = main(
        ["report", "--data", str(data_path), "--group-col", "group", "--out", str(report_path)]
    )
    assert code == 0
    text = report_path.read_text(encoding="utf-8")
    assert "# Survival Analysis Report" in text
    assert "## Log-rank test" in text


def test_report_command_includes_concordance_for_score_columns(tmp_path):
    data_path = tmp_path / "scored.csv"
    report_path = tmp_path / "scored.md"
    data = generate_survival_data(150, group_scale_ratio=4.0, seed=8)
    noise = np.random.default_rng(9).normal(scale=0.05, size=150)
    scores = np.where(data.groups == "control", 1.0, 0.0) + noise
    save_csv(data, data_path, extra_columns={"score": scores})

    code = main(
        [
            "report",
            "--data", str(data_path),
            "--group-col", "group",
            "--score-col", "score",
            "--title", "Scored demo",
            "--out", str(report_path),
        ]
    )
    assert code == 0
    text = report_path.read_text(encoding="utf-8")
    assert "## Concordance" in text
    c_index = float(text.split("Harrell's C: ")[1].splitlines()[0])
    assert c_index > 0.55


def test_unknown_command_and_missing_arguments_are_rejected():
    with pytest.raises(SystemExit) as excinfo:
        main(["explode"])
    assert excinfo.value.code == 2
    with pytest.raises(SystemExit):
        main(["generate"])


def test_cox_command_recovers_treatment_hazard_ratio(tmp_path, capsys):
    data_path = tmp_path / "arms.csv"
    out_path = tmp_path / "cox.csv"
    main(
        [
            "generate",
            "--n", "800",
            "--shape", "1.2",
            "--group-scale-ratio", "2.0",
            "--seed", "5",
            "--out", str(data_path),
        ]
    )
    assert main(["cox", "--data", str(data_path), "--group-col", "group", "--out", str(out_path)]) == 0
    out = capsys.readouterr().out
    assert "group[treatment]" in out
    assert "HR=" in out
    assert "log partial likelihood" in out
    with open(out_path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["covariate"] == "group[treatment]"
    hr = float(rows[0]["hazard_ratio"])
    assert hr == pytest.approx(2.0 ** (-1.2), rel=0.25)


def test_cox_command_requires_a_design_matrix():
    with pytest.raises(SystemExit):
        main(["cox", "--data", "missing.csv", "--out", "cox.csv"])


def test_report_includes_cox_section_for_grouped_data(tmp_path):
    data_path = tmp_path / "data.csv"
    report_path = tmp_path / "report.md"
    main(["generate", "--n", "150", "--seed", "4", "--out", str(data_path)])
    main(
        ["report", "--data", str(data_path), "--group-col", "group", "--out", str(report_path)]
    )
    text = report_path.read_text(encoding="utf-8")
    assert "## Cox proportional hazards" in text
    assert "group[treatment]" in text
    assert "Likelihood-ratio" in text


def test_report_cox_section_from_numeric_covariates(tmp_path):
    data_path = tmp_path / "ph.csv"
    report_path = tmp_path / "ph.md"
    ph = generate_ph_data(180, [0.7], seed=12)
    dummy = generate_survival_data(180, seed=12)
    dummy.durations = ph.durations
    dummy.events = ph.events
    save_csv(dummy, data_path, extra_columns={"x0": ph.covariates[:, 0]})
    main(
        [
            "report",
            "--data", str(data_path),
            "--covariate-cols", "x0",
            "--out", str(report_path),
        ]
    )
    text = report_path.read_text(encoding="utf-8")
    assert "## Cox proportional hazards" in text
    assert "| x0 |" in text


def test_render_report_optional_cox_section():
    data = generate_survival_data(120, group_scale_ratio=2.5, seed=11)
    cohorts = [_summary(data, "control"), _summary(data, "treatment")]
    treated = (data.groups == "treatment").astype(float)
    cox = fit_cox_ph(data.durations, data.events, treated, feature_names=("treated",))
    text = render_report("Demo", cohorts, cox=cox)
    assert "## Cox proportional hazards" in text
    assert "treated" in text
    assert "Log partial likelihood:" in text


def test_aft_command_recovers_treatment_time_ratio(tmp_path, capsys):
    data_path = tmp_path / "arms.csv"
    out_path = tmp_path / "aft.csv"
    main(
        [
            "generate",
            "--n", "800",
            "--shape", "1.2",
            "--group-scale-ratio", "2.0",
            "--seed", "5",
            "--out", str(data_path),
        ]
    )
    assert main(["aft", "--data", str(data_path), "--group-col", "group", "--out", str(out_path)]) == 0
    out = capsys.readouterr().out
    assert "group[treatment]" in out
    assert "TR=" in out
    assert "log-likelihood" in out
    assert "shape =" in out
    with open(out_path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    names = {row["parameter"] for row in rows}
    assert {"(Intercept)", "group[treatment]", "log(sigma)"} <= names
    treated = next(row for row in rows if row["parameter"] == "group[treatment]")
    tr = float(treated["time_ratio"])
    assert tr == pytest.approx(2.0, rel=0.25)


def test_aft_command_fits_intercept_only(tmp_path, capsys):
    data_path = tmp_path / "one.csv"
    out_path = tmp_path / "aft.csv"
    main(["generate", "--n", "120", "--seed", "3", "--out", str(data_path)])
    assert main(["aft", "--data", str(data_path), "--out", str(out_path)]) == 0
    out = capsys.readouterr().out
    assert "(Intercept)" in out
    assert "log(sigma)" in out
    with open(out_path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["parameter"] for row in rows] == ["(Intercept)", "log(sigma)"]


def test_report_includes_aft_section_for_grouped_data(tmp_path):
    data_path = tmp_path / "data.csv"
    report_path = tmp_path / "report.md"
    main(["generate", "--n", "150", "--seed", "4", "--out", str(data_path)])
    main(
        ["report", "--data", str(data_path), "--group-col", "group", "--out", str(report_path)]
    )
    text = report_path.read_text(encoding="utf-8")
    assert "## Weibull accelerated failure time" in text
    assert "group[treatment]" in text
    assert "Shape (1/sigma):" in text
    assert "time ratio" in text


def test_report_includes_intercept_only_aft_without_design(tmp_path):
    data_path = tmp_path / "data.csv"
    report_path = tmp_path / "report.md"
    main(["generate", "--n", "80", "--seed", "4", "--out", str(data_path)])
    main(["report", "--data", str(data_path), "--out", str(report_path)])
    text = report_path.read_text(encoding="utf-8")
    assert "## Weibull accelerated failure time" in text
    assert "(Intercept)" in text
    assert "## Cox proportional hazards" not in text


def test_render_report_optional_aft_section():
    from survival_kit.aft import fit_weibull_aft

    data = generate_survival_data(120, group_scale_ratio=2.5, seed=11)
    cohorts = [_summary(data, "control"), _summary(data, "treatment")]
    treated = (data.groups == "treatment").astype(float)
    aft = fit_weibull_aft(data.durations, data.events, treated, feature_names=("treated",))
    text = render_report("Demo", cohorts, aft=aft)
    assert "## Weibull accelerated failure time" in text
    assert "treated" in text
    assert "Log-likelihood:" in text
    assert "Shape (1/sigma):" in text


def test_rmst_command_reports_two_group_difference(tmp_path, capsys):
    data_path = tmp_path / "arms.csv"
    out_path = tmp_path / "rmst.csv"
    main(
        [
            "generate",
            "--n", "400",
            "--shape", "1.0",
            "--scale", "2.0",
            "--group-scale-ratio", "3.0",
            "--seed", "7",
            "--out", str(data_path),
        ]
    )
    assert (
        main(
            [
                "rmst",
                "--data", str(data_path),
                "--group-col", "group",
                "--tau", "3",
                "--out", str(out_path),
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "tau = 3" in out
    assert "control: RMST=" in out
    assert "treatment: RMST=" in out
    assert "difference (control - treatment)" in out
    assert "p-value" in out
    with open(out_path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    names = {row["cohort"] for row in rows}
    assert {"control", "treatment", "control-treatment"} <= names
    control = next(row for row in rows if row["cohort"] == "control")
    treatment = next(row for row in rows if row["cohort"] == "treatment")
    assert float(treatment["rmst"]) > float(control["rmst"])


def test_rmst_command_rejects_tau_past_follow_up(tmp_path):
    data_path = tmp_path / "tiny.csv"
    main(["generate", "--n", "40", "--seed", "3", "--out", str(data_path)])
    with pytest.raises(SystemExit, match="last follow-up"):
        main(["rmst", "--data", str(data_path), "--group-col", "group", "--tau", "1e9"])


def test_report_command_includes_rmst_section(tmp_path):
    data_path = tmp_path / "data.csv"
    report_path = tmp_path / "report.md"
    main(["generate", "--n", "150", "--seed", "4", "--out", str(data_path)])
    main(
        [
            "report",
            "--data", str(data_path),
            "--group-col", "group",
            "--tau", "2.5",
            "--out", str(report_path),
        ]
    )
    text = report_path.read_text(encoding="utf-8")
    assert "## Restricted mean survival time" in text
    assert "Truncation time (tau): 2.500" in text
    assert "Difference (control - treatment):" in text


def test_cif_command_writes_curves_and_gray_test(tmp_path, capsys):
    data_path = tmp_path / "cr.csv"
    out_path = tmp_path / "cif.csv"
    main(
        [
            "generate",
            "--n", "500",
            "--cause-rates", "0.5", "0.25",
            "--cause-rate-ratios", "0.3", "1.0",
            "--censor-fraction", "0.2",
            "--seed", "7",
            "--out", str(data_path),
        ]
    )
    with open(data_path, newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    assert rows[0] == ["duration", "event", "group"]
    causes = {int(row[1]) for row in rows[1:]}
    assert {1, 2} <= causes

    assert (
        main(
            [
                "cif",
                "--data", str(data_path),
                "--group-col", "group",
                "--out", str(out_path),
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "cause 1" in out
    assert "cause 2" in out
    assert "Gray cause 1" in out
    assert "p-value" in out
    with open(out_path, newline="", encoding="utf-8") as handle:
        fitted = list(csv.DictReader(handle))
    assert {"control", "treatment"} <= {row["cohort"] for row in fitted}
    assert {"1", "2"} <= {row["cause"] for row in fitted}
    for row in fitted:
        incidence = float(row["incidence"])
        assert 0.0 <= incidence <= 1.0


def test_cif_command_can_restrict_to_one_cause(tmp_path, capsys):
    data_path = tmp_path / "cr.csv"
    out_path = tmp_path / "cif.csv"
    main(
        [
            "generate",
            "--n", "120",
            "--cause-rates", "0.4", "0.3",
            "--seed", "3",
            "--out", str(data_path),
        ]
    )
    main(
        [
            "cif",
            "--data", str(data_path),
            "--cause", "2",
            "--out", str(out_path),
        ]
    )
    out = capsys.readouterr().out
    assert "cause 2" in out
    assert "cause 1:" not in out
    with open(out_path, newline="", encoding="utf-8") as handle:
        fitted = list(csv.DictReader(handle))
    assert fitted
    assert {row["cause"] for row in fitted} == {"2"}


def test_report_command_includes_cif_and_gray_for_competing_risks(tmp_path):
    data_path = tmp_path / "cr.csv"
    report_path = tmp_path / "cr.md"
    main(
        [
            "generate",
            "--n", "250",
            "--cause-rates", "0.45", "0.3",
            "--cause-rate-ratios", "0.4", "1.0",
            "--seed", "4",
            "--out", str(data_path),
        ]
    )
    main(
        [
            "report",
            "--data", str(data_path),
            "--group-col", "group",
            "--title", "Competing risks demo",
            "--out", str(report_path),
        ]
    )
    text = report_path.read_text(encoding="utf-8")
    assert "## Cumulative incidence" in text
    assert "### control, cause 1" in text
    assert "### treatment, cause 2" in text
    assert "### Gray's test (cause 1)" in text
    assert "Observed vs expected events:" in text


def test_report_omits_cif_section_for_single_cause_data(tmp_path):
    data_path = tmp_path / "data.csv"
    report_path = tmp_path / "report.md"
    main(["generate", "--n", "80", "--seed", "4", "--out", str(data_path)])
    main(
        ["report", "--data", str(data_path), "--group-col", "group", "--out", str(report_path)]
    )
    text = report_path.read_text(encoding="utf-8")
    assert "## Cumulative incidence" not in text


def test_render_report_optional_cif_section():
    from survival_kit.competing_risks import fit_all_cumulative_incidence, gray_test_groups
    from survival_kit.synth import generate_competing_risks_data

    data = generate_competing_risks_data(180, cause_rates=(0.4, 0.3), seed=11)
    cohorts = [_summary(data, "control"), _summary(data, "treatment")]
    cif_summaries = [
        (
            "control",
            fit_all_cumulative_incidence(
                data.durations[data.groups == "control"],
                data.event_types[data.groups == "control"],
            ),
        ),
        (
            "treatment",
            fit_all_cumulative_incidence(
                data.durations[data.groups == "treatment"],
                data.event_types[data.groups == "treatment"],
            ),
        ),
    ]
    gray = [
        gray_test_groups(data.durations, data.event_types, data.groups, cause=1),
        gray_test_groups(data.durations, data.event_types, data.groups, cause=2),
    ]
    text = render_report(
        "CR",
        cohorts,
        cif_summaries=cif_summaries,
        gray_tests=gray,
        gray_labels=["control", "treatment"],
    )
    assert "## Cumulative incidence" in text
    assert "Gray's test (cause 1)" in text
    assert "control, cause 1" in text
