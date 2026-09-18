"""Markdown rendering of survival analysis summaries."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np

from .utils import step_value


@dataclass
class CohortSummary:
    """One cohort's fitted curve plus the counts behind it."""

    name: str
    curve: object
    observations: int
    events: int

    @property
    def censored(self):
        return self.observations - self.events


def format_p_value(p):
    """Render a p-value with a small-significance floor."""
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


def default_horizons(cohorts, fractions=(0.25, 0.5, 0.75, 0.9)):
    """Evaluation grid: time zero plus pooled event-time quantiles."""
    pooled = [c.curve.time for c in cohorts if c.curve.time.size]
    if not pooled:
        return [0.0]
    combined = np.concatenate(pooled)
    return [0.0] + [float(np.quantile(combined, q)) for q in fractions]


def render_overview(cohorts):
    total_n = sum(c.observations for c in cohorts)
    total_events = sum(c.events for c in cohorts)
    total_censored = total_n - total_events
    safe_total = max(total_n, 1)
    labels = ", ".join(f"{c.name} (n={c.observations})" for c in cohorts)
    lines = [
        "## Overview",
        "",
        f"- Observations: {total_n}",
        f"- Events: {total_events} ({100.0 * total_events / safe_total:.1f}%)",
        f"- Censored: {total_censored} ({100.0 * total_censored / safe_total:.1f}%)",
        f"- Cohorts: {labels}",
    ]
    return "\n".join(lines)


def render_cohort_section(cohort, horizons, confidence_level=0.95):
    """Survival table at ``horizons`` plus the cohort's median survival."""
    curve = cohort.curve
    ci_label = f"{round(100 * confidence_level)}% CI"
    lines = [
        f"### {cohort.name}",
        "",
        f"| time | survival | {ci_label} | std. error |",
        "| ---: | ---: | :---: | ---: |",
    ]
    for t in horizons:
        s = step_value(curve.time, curve.survival, t, outside=1.0)
        lo = step_value(curve.time, curve.ci_lower, t, outside=1.0)
        hi = step_value(curve.time, curve.ci_upper, t, outside=1.0)
        se = step_value(curve.time, curve.std_err, t, outside=0.0)
        lines.append(f"| {t:.3f} | {s:.3f} | [{lo:.3f}, {hi:.3f}] | {se:.3f} |")

    median = curve.median()
    median_text = f"{median:.3f}" if median is not None else "not reached"
    lines.extend(["", f"- Median survival: {median_text}"])
    return "\n".join(lines)


def render_log_rank_section(result, labels):
    """Chi-square, p-value, and per-group observed/expected deaths."""
    pairs = ", ".join(
        f"{label} {observed:.1f}/{expected:.1f}"
        for label, observed, expected in zip(labels, result.observed_deaths, result.expected_deaths)
    )
    lines = [
        "## Log-rank test",
        "",
        f"- Chi-square({result.degrees_of_freedom}): {result.statistic:.2f}",
        f"- p-value: {format_p_value(result.p_value)}",
        f"- Observed vs expected deaths: {pairs}",
    ]
    return "\n".join(lines)


def render_concordance_section(result):
    """Discrimination of a risk score against observed event orderings."""
    return "\n".join(
        [
            "## Concordance",
            "",
            f"- Harrell's C: {result.c_index:.3f}",
            (
                f"- Comparable pairs: {result.comparable_pairs} "
                f"(concordant {result.concordant_pairs}, tied {result.tied_pairs})"
            ),
        ]
    )


def render_rmst_section(results, labels, difference=None, confidence_level=0.95):
    """Per-cohort RMST table and, when present, the two-group difference."""
    ci_label = f"{round(100 * confidence_level)}% CI"
    tau = results[0].tau
    lines = [
        "## Restricted mean survival time",
        "",
        f"- Truncation time (tau): {tau:.3f}",
        "",
        f"| cohort | RMST | {ci_label} | std. error | RMTL |",
        "| --- | ---: | :---: | ---: | ---: |",
    ]
    for label, result in zip(labels, results):
        lines.append(
            f"| {label} | {result.rmst:.3f} | [{result.ci_lower:.3f}, {result.ci_upper:.3f}] "
            f"| {result.std_err:.3f} | {result.rmtl:.3f} |"
        )
    if difference is not None and len(labels) >= 2:
        lines.extend(
            [
                "",
                (
                    f"- Difference ({labels[0]} - {labels[1]}): "
                    f"{difference.difference:.3f}"
                ),
                f"- Difference std. error: {difference.std_err:.3f}",
                f"- z: {difference.z_score:.2f}",
                f"- p-value: {format_p_value(difference.p_value)}",
            ]
        )
    return "\n".join(lines)


def render_cox_section(result, confidence_level=0.95):
    """Coefficient table and partial-likelihood summary for a Cox PH fit."""
    ci_label = f"{round(100 * confidence_level)}% CI (HR)"
    lines = [
        "## Cox proportional hazards",
        "",
        f"| covariate | coef | HR | {ci_label} | p |",
        "| --- | ---: | ---: | :---: | ---: |",
    ]
    for i, name in enumerate(result.feature_names):
        lo = result.hazard_ratio_ci_lower[i]
        hi = result.hazard_ratio_ci_upper[i]
        lines.append(
            f"| {name} | {result.coefficients[i]:.3f} | {result.hazard_ratios[i]:.3f} "
            f"| [{lo:.3f}, {hi:.3f}] | {format_p_value(result.p_values[i])} |"
        )
    lines.extend(
        [
            "",
            f"- Log partial likelihood: {result.log_partial_likelihood:.3f}",
            (
                f"- Likelihood-ratio chi-square({len(result.coefficients)}): "
                f"{result.likelihood_ratio_statistic:.2f}"
            ),
            f"- Likelihood-ratio p-value: {format_p_value(result.likelihood_ratio_p_value)}",
        ]
    )
    return "\n".join(lines)


def render_report(
    title,
    cohorts,
    log_rank=None,
    log_rank_labels=None,
    concordance=None,
    cox=None,
    rmst=None,
    rmst_difference=None,
    rmst_labels=None,
    generated_on=None,
):
    """Assemble the full markdown evaluation report."""
    day = generated_on or date.today()
    parts = [f"# {title}", "", f"Generated: {day.isoformat()}", ""]
    parts.append(render_overview(cohorts))
    parts.append("")
    parts.append("## Cohorts")
    parts.append("")
    horizons = default_horizons(cohorts)
    for cohort in cohorts:
        parts.append(render_cohort_section(cohort, horizons))
        parts.append("")
    if log_rank is not None:
        parts.append(render_log_rank_section(log_rank, log_rank_labels or []))
        parts.append("")
    if rmst:
        parts.append(
            render_rmst_section(
                rmst, rmst_labels or [c.name for c in cohorts], difference=rmst_difference
            )
        )
        parts.append("")
    if cox is not None:
        parts.append(render_cox_section(cox))
        parts.append("")
    if concordance is not None:
        parts.append(render_concordance_section(concordance))
        parts.append("")
    return "\n".join(parts).rstrip() + "\n"