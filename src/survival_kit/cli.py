"""Command-line interface wiring generate -> fit -> compare -> report."""

from __future__ import annotations

import argparse
import sys
from datetime import date

import numpy as np

from .concordance import concordance_index
from .cox import fit_cox_ph
from .kaplan_meier import fit_kaplan_meier
from .logrank import log_rank_test, log_rank_test_groups
from .report import CohortSummary, render_report
from .synth import generate_survival_data, load_csv, save_csv


def build_parser():
    parser = argparse.ArgumentParser(
        prog="survival-kit",
        description="Fit and compare right-censored survival data.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="draw a synthetic two-arm sample")
    gen.add_argument("--n", type=int, default=400)
    gen.add_argument("--shape", type=float, default=1.2)
    gen.add_argument("--scale", type=float, default=5.0)
    gen.add_argument("--censor-fraction", type=float, default=0.2)
    gen.add_argument("--group-scale-ratio", type=float, default=1.8)
    gen.add_argument("--arm-fraction", type=float, default=0.5)
    gen.add_argument("--seed", type=int, default=42)
    gen.add_argument("--out", required=True)

    fit = sub.add_parser("fit", help="fit Kaplan-Meier curves to a CSV")
    fit.add_argument("--data", required=True)
    fit.add_argument("--time-col", default="duration")
    fit.add_argument("--event-col", default="event")
    fit.add_argument("--group-col", default=None)
    fit.add_argument("--out", required=True)

    compare = sub.add_parser("compare", help="log-rank comparison between groups")
    compare.add_argument("--data", required=True)
    compare.add_argument("--time-col", default="duration")
    compare.add_argument("--event-col", default="event")
    compare.add_argument("--group-col", required=True)

    report = sub.add_parser("report", help="write a markdown evaluation report")
    report.add_argument("--data", required=True)
    report.add_argument("--time-col", default="duration")
    report.add_argument("--event-col", default="event")
    report.add_argument("--group-col", default=None)
    report.add_argument("--score-col", default=None)
    report.add_argument(
        "--covariate-cols",
        nargs="+",
        default=None,
        help="numeric columns to include in an optional Cox PH section",
    )
    report.add_argument("--title", default="Survival Analysis Report")
    report.add_argument("--out", required=True)

    cox = sub.add_parser("cox", help="fit a Cox proportional hazards model")
    cox.add_argument("--data", required=True)
    cox.add_argument("--time-col", default="duration")
    cox.add_argument("--event-col", default="event")
    cox.add_argument("--group-col", default=None)
    cox.add_argument("--covariate-cols", nargs="+", default=None)
    cox.add_argument("--out", required=True)
    return parser


def _load(args):
    return load_csv(args.data, time_col=args.time_col, event_col=args.event_col)


def _cohort_labels(args, groups):
    if args.group_col:
        return [str(label) for label in np.unique(groups)]
    return ["all"]


def _design_matrix(groups, extras, group_col=None, covariate_cols=None):
    """Dummy-encode groups (dropping the first level) and stack numeric columns."""
    columns = []
    names = []
    if group_col:
        labels = [str(label) for label in np.unique(groups)]
        if len(labels) >= 2:
            for label in labels[1:]:
                columns.append((groups == label).astype(float))
                names.append(f"{group_col}[{label}]")
    for col in covariate_cols or []:
        if col not in extras:
            raise SystemExit(f"unknown covariate column: {col}")
        columns.append(np.asarray(extras[col], dtype=float))
        names.append(col)
    if not columns:
        return None, None
    return np.column_stack(columns), names


def _cohorts(durations, events, groups, labels):
    summaries = []
    for label in labels:
        mask = groups == label if label != "all" else np.ones(durations.size, bool)
        curve = fit_kaplan_meier(durations[mask], events[mask])
        summaries.append(
            CohortSummary(
                name=label,
                curve=curve,
                observations=int(mask.sum()),
                events=int(events[mask].sum()),
            )
        )
    return summaries


def run_generate(args):
    data = generate_survival_data(
        args.n,
        shape=args.shape,
        scale=args.scale,
        censor_fraction=args.censor_fraction,
        group_scale_ratio=args.group_scale_ratio,
        arm_fraction=args.arm_fraction,
        seed=args.seed,
    )
    save_csv(data, args.out)
    print(f"wrote {args.n} observations to {args.out}")


def run_fit(args):
    durations, events, groups, _ = _load(args)
    labels = _cohort_labels(args, groups)
    for cohort in _cohorts(durations, events, groups, labels):
        median = cohort.curve.median()
        median_text = f"{median:.3f}" if median is not None else "not reached"
        print(
            f"{cohort.name}: n={cohort.observations}, events={cohort.events}, "
            f"median={median_text}"
        )

    with open(args.out, "w", newline="", encoding="utf-8") as handle:
        multi = len(labels) > 1 or labels != ["all"]
        header = "time,survival,std_err,ci_lower,ci_upper"
        if args.group_col:
            handle.write(header + ",cohort\n")
        else:
            handle.write(header + "\n")
        for cohort in _cohorts(durations, events, groups, labels):
            curve = cohort.curve
            for i in range(curve.time.size):
                row = (
                    f"{curve.time[i]:.6f},{curve.survival[i]:.6f},"
                    f"{curve.std_err[i]:.6f},{curve.ci_lower[i]:.6f},"
                    f"{curve.ci_upper[i]:.6f}"
                )
                if args.group_col:
                    row += f",{cohort.name}"
                handle.write(row + "\n")
    print(f"wrote fitted curves to {args.out}")


def run_compare(args):
    durations, events, groups, _ = _load(args)
    labels = [str(label) for label in np.unique(groups)]
    if len(labels) < 2:
        raise SystemExit("compare needs at least two distinct groups")
    if len(labels) == 2:
        first, second = (groups == label for label in labels)
        result = log_rank_test(durations[first], events[first], durations[second], events[second])
    else:
        result = log_rank_test_groups(durations, events, groups)
    print(f"groups: {len(labels)}")
    for label, observed, expected in zip(labels, result.observed_deaths, result.expected_deaths):
        print(f"{label}: observed {observed:g}, expected {expected:.2f}")
    print(f"chi-square({result.degrees_of_freedom}) = {result.statistic:.3f}")
    print(f"p-value = {result.p_value:.3g}")


def run_report(args):
    durations, events, groups, extras = _load(args)
    labels = _cohort_labels(args, groups)
    cohorts = _cohorts(durations, events, groups, labels)

    log_rank = None
    if len(labels) >= 2:
        if len(labels) == 2:
            first, second = (groups == label for label in labels)
            log_rank = log_rank_test(
                durations[first], events[first], durations[second], events[second]
            )
        else:
            log_rank = log_rank_test_groups(durations, events, groups)

    discrimination = None
    if args.score_col:
        scores = extras[args.score_col]
        discrimination = concordance_index(durations, events, scores)

    cox_fit = None
    design, names = _design_matrix(
        groups, extras, group_col=args.group_col, covariate_cols=args.covariate_cols
    )
    if design is not None:
        cox_fit = fit_cox_ph(durations, events, design, feature_names=names)

    text = render_report(
        title=args.title,
        cohorts=cohorts,
        log_rank=log_rank,
        log_rank_labels=labels if len(labels) >= 2 else None,
        concordance=discrimination,
        cox=cox_fit,
        generated_on=date.today(),
    )
    with open(args.out, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    print(f"wrote report to {args.out}")
    if discrimination is not None:
        print(f"Harrell's C for {args.score_col!r}: {discrimination.c_index:.3f}")
    if cox_fit is not None:
        print(
            f"Cox PH log partial likelihood: {cox_fit.log_partial_likelihood:.3f} "
            f"(LR p={cox_fit.likelihood_ratio_p_value:.3g})"
        )


def run_cox(args):
    if not args.group_col and not args.covariate_cols:
        raise SystemExit("cox needs --group-col and/or --covariate-cols")
    durations, events, groups, extras = _load(args)
    design, names = _design_matrix(
        groups, extras, group_col=args.group_col, covariate_cols=args.covariate_cols
    )
    if design is None:
        raise SystemExit("cox produced an empty design matrix")
    result = fit_cox_ph(durations, events, design, feature_names=names)

    print(f"n={result.n_observations}, events={result.n_events}, iterations={result.n_iterations}")
    print(f"log partial likelihood = {result.log_partial_likelihood:.6f}")
    print(
        f"likelihood-ratio chi-square({len(result.coefficients)}) = "
        f"{result.likelihood_ratio_statistic:.3f} (p={result.likelihood_ratio_p_value:.3g})"
    )
    for i, name in enumerate(result.feature_names):
        print(
            f"{name}: coef={result.coefficients[i]:.4f} se={result.std_err[i]:.4f} "
            f"HR={result.hazard_ratios[i]:.4f} "
            f"[{result.hazard_ratio_ci_lower[i]:.4f}, {result.hazard_ratio_ci_upper[i]:.4f}] "
            f"p={result.p_values[i]:.3g}"
        )

    with open(args.out, "w", newline="", encoding="utf-8") as handle:
        handle.write(
            "covariate,coefficient,std_err,z,p_value,hazard_ratio,hr_ci_lower,hr_ci_upper\n"
        )
        for i, name in enumerate(result.feature_names):
            handle.write(
                f"{name},{result.coefficients[i]:.10g},{result.std_err[i]:.10g},"
                f"{result.z_scores[i]:.10g},{result.p_values[i]:.10g},"
                f"{result.hazard_ratios[i]:.10g},{result.hazard_ratio_ci_lower[i]:.10g},"
                f"{result.hazard_ratio_ci_upper[i]:.10g}\n"
            )
    print(f"wrote Cox PH estimates to {args.out}")


COMMANDS = {
    "generate": run_generate,
    "fit": run_fit,
    "compare": run_compare,
    "report": run_report,
    "cox": run_cox,
}


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    COMMANDS[args.command](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())