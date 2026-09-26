"""Command-line interface wiring generate -> fit -> compare -> cox -> aft -> aalen -> finegray -> report."""

from __future__ import annotations

import argparse
import sys
from datetime import date

import numpy as np

from .aft import fit_weibull_aft
from .competing_risks import (
    fit_all_cumulative_incidence,
    fit_cumulative_incidence,
    gray_test_groups,
)
from .concordance import concordance_index
from .aalen import fit_aalen_additive
from .cox import fit_cox_ph
from .fine_gray import fit_fine_gray
from .kaplan_meier import fit_kaplan_meier
from .logrank import log_rank_test, log_rank_test_groups
from .report import CohortSummary, render_report
from .rmst import (
    default_truncation_time,
    restricted_mean_survival_time,
    rmst_difference_test,
)
from .synth import generate_competing_risks_data, generate_survival_data, load_csv, save_csv
from .utils import observed_causes


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
    gen.add_argument(
        "--cause-rates",
        nargs="+",
        type=float,
        default=None,
        help="exponential rates per competing cause; switches generate to competing risks",
    )
    gen.add_argument(
        "--cause-rate-ratios",
        nargs="+",
        type=float,
        default=None,
        help="treatment/control rate ratios per cause (default: 1 for every cause)",
    )
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
    report.add_argument(
        "--tau",
        type=float,
        default=None,
        help="truncation time for the RMST section (default: min last follow-up)",
    )
    report.add_argument("--out", required=True)

    rmst = sub.add_parser("rmst", help="restricted mean survival time up to tau")
    rmst.add_argument("--data", required=True)
    rmst.add_argument("--time-col", default="duration")
    rmst.add_argument("--event-col", default="event")
    rmst.add_argument("--group-col", default=None)
    rmst.add_argument(
        "--tau",
        type=float,
        default=None,
        help="truncation time (default: min last follow-up across cohorts)",
    )
    rmst.add_argument("--out", default=None, help="optional CSV of RMST estimates")

    cif = sub.add_parser("cif", help="Aalen-Johansen cumulative incidence for competing events")
    cif.add_argument("--data", required=True)
    cif.add_argument("--time-col", default="duration")
    cif.add_argument("--event-col", default="event")
    cif.add_argument("--group-col", default=None)
    cif.add_argument(
        "--cause",
        type=int,
        default=None,
        help="cause code to estimate (default: every observed cause)",
    )
    cif.add_argument("--out", default=None, help="optional CSV of CIF estimates")

    cox = sub.add_parser("cox", help="fit a Cox proportional hazards model")
    cox.add_argument("--data", required=True)
    cox.add_argument("--time-col", default="duration")
    cox.add_argument("--event-col", default="event")
    cox.add_argument("--group-col", default=None)
    cox.add_argument("--covariate-cols", nargs="+", default=None)
    cox.add_argument("--out", required=True)

    aalen = sub.add_parser(
        "aalen",
        help="fit Aalen's additive hazards model (cumulative coefficients)",
    )
    aalen.add_argument("--data", required=True)
    aalen.add_argument("--time-col", default="duration")
    aalen.add_argument("--event-col", default="event")
    aalen.add_argument("--group-col", default=None)
    aalen.add_argument("--covariate-cols", nargs="+", default=None)
    aalen.add_argument("--out", required=True)

    aft = sub.add_parser("aft", help="fit a Weibull accelerated failure time model")
    aft.add_argument("--data", required=True)
    aft.add_argument("--time-col", default="duration")
    aft.add_argument("--event-col", default="event")
    aft.add_argument("--group-col", default=None)
    aft.add_argument("--covariate-cols", nargs="+", default=None)
    aft.add_argument("--out", required=True)

    finegray = sub.add_parser(
        "finegray",
        help="fit a Fine-Gray subdistribution hazard model for one competing cause",
    )
    finegray.add_argument("--data", required=True)
    finegray.add_argument("--time-col", default="duration")
    finegray.add_argument("--event-col", default="event")
    finegray.add_argument("--group-col", default=None)
    finegray.add_argument("--covariate-cols", nargs="+", default=None)
    finegray.add_argument(
        "--cause",
        type=int,
        default=1,
        help="cause code whose subdistribution hazard is modeled (default: 1)",
    )
    finegray.add_argument("--out", required=True)
    return parser


def _load(args, with_event_types=False):
    return load_csv(
        args.data,
        time_col=args.time_col,
        event_col=args.event_col,
        with_event_types=with_event_types,
    )


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


def _cohort_mask(groups, label):
    return groups == label if label != "all" else np.ones(groups.size, bool)


def _shared_tau(durations, groups, labels, tau):
    slices = [durations[_cohort_mask(groups, label)] for label in labels]
    identifiable = default_truncation_time(*slices)
    if tau is None:
        return identifiable
    tau = float(tau)
    if not np.isfinite(tau) or tau <= 0.0:
        raise SystemExit("tau must be a finite positive number")
    if tau > identifiable:
        raise SystemExit(
            f"tau must not exceed the last follow-up time ({identifiable})"
        )
    return tau


def _rmst_fits(durations, events, groups, labels, tau):
    fits = []
    for label in labels:
        mask = _cohort_mask(groups, label)
        try:
            fits.append(
                restricted_mean_survival_time(durations[mask], events[mask], tau=tau)
            )
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
    return fits


def _rmst_difference(durations, events, groups, labels, tau):
    if len(labels) != 2:
        return None
    first, second = (_cohort_mask(groups, label) for label in labels)
    try:
        return rmst_difference_test(
            durations[first],
            events[first],
            durations[second],
            events[second],
            tau=tau,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


def run_generate(args):
    if args.cause_rate_ratios is not None and args.cause_rates is None:
        raise SystemExit("--cause-rate-ratios requires --cause-rates")
    if args.cause_rates is not None:
        data = generate_competing_risks_data(
            args.n,
            cause_rates=args.cause_rates,
            censor_fraction=args.censor_fraction,
            group_rate_ratios=args.cause_rate_ratios if args.cause_rate_ratios else 1.0,
            arm_fraction=args.arm_fraction,
            seed=args.seed,
        )
    else:
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
    durations, events, groups, extras, event_types = _load(args, with_event_types=True)
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
    aft_fit = None
    design, names = _design_matrix(
        groups, extras, group_col=args.group_col, covariate_cols=args.covariate_cols
    )
    if design is not None:
        cox_fit = fit_cox_ph(durations, events, design, feature_names=names)
        aft_fit = fit_weibull_aft(durations, events, design, feature_names=names)
    else:
        aft_fit = fit_weibull_aft(durations, events)

    tau = _shared_tau(durations, groups, labels, args.tau)
    rmst_fits = _rmst_fits(durations, events, groups, labels, tau)
    rmst_diff = _rmst_difference(durations, events, groups, labels, tau)

    cif_summaries = None
    gray_tests = None
    causes = observed_causes(event_types)
    if len(causes) >= 2:
        cif_summaries = []
        for label in labels:
            mask = _cohort_mask(groups, label)
            cif_summaries.append(
                (label, fit_all_cumulative_incidence(durations[mask], event_types[mask]))
            )
        if len(labels) >= 2:
            gray_tests = [
                gray_test_groups(durations, event_types, groups, cause=cause)
                for cause in causes
            ]

    text = render_report(
        title=args.title,
        cohorts=cohorts,
        log_rank=log_rank,
        log_rank_labels=labels if len(labels) >= 2 else None,
        concordance=discrimination,
        cox=cox_fit,
        aft=aft_fit,
        rmst=rmst_fits,
        rmst_difference=rmst_diff,
        rmst_labels=labels,
        cif_summaries=cif_summaries,
        gray_tests=gray_tests,
        gray_labels=labels if gray_tests else None,
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
    if aft_fit is not None:
        print(
            f"Weibull AFT log-likelihood: {aft_fit.log_likelihood:.3f} "
            f"(shape={aft_fit.shape:.3f}, LR p={aft_fit.likelihood_ratio_p_value:.3g})"
        )
    if cif_summaries:
        print(f"competing causes: {', '.join(str(cause) for cause in causes)}")


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


def run_aalen(args):
    if not args.group_col and not args.covariate_cols:
        raise SystemExit("aalen needs --group-col and/or --covariate-cols")
    durations, events, groups, extras = _load(args)
    design, names = _design_matrix(
        groups, extras, group_col=args.group_col, covariate_cols=args.covariate_cols
    )
    if design is None:
        raise SystemExit("aalen produced an empty design matrix")
    result = fit_aalen_additive(durations, events, design, feature_names=names)

    print(f"n={result.n_observations}, events={result.n_events}, times={result.time.size}")
    # Report the final cumulative coefficients (end of follow-up).
    final = result.cumulative_coefficients[-1]
    final_se = result.std_err[-1]
    for i, name in enumerate(result.feature_names):
        print(
            f"{name}: B(T)={final[i]:.4f} se={final_se[i]:.4f} "
            f"[{result.ci_lower[-1, i]:.4f}, {result.ci_upper[-1, i]:.4f}]"
        )

    with open(args.out, "w", newline="", encoding="utf-8") as handle:
        header = ["time"] + [f"B_{name}" for name in result.feature_names] + [
            f"se_{name}" for name in result.feature_names
        ]
        handle.write(",".join(header) + "\n")
        for k, t in enumerate(result.time):
            row = [f"{t:.10g}"]
            row.extend(f"{result.cumulative_coefficients[k, j]:.10g}" for j in range(final.size))
            row.extend(f"{result.std_err[k, j]:.10g}" for j in range(final.size))
            handle.write(",".join(row) + "\n")
    print(f"wrote Aalen cumulative coefficients to {args.out}")



def run_aft(args):
    durations, events, groups, extras = _load(args)
    design, names = _design_matrix(
        groups, extras, group_col=args.group_col, covariate_cols=args.covariate_cols
    )
    try:
        result = fit_weibull_aft(
            durations, events, design, feature_names=names if design is not None else None
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    print(f"n={result.n_observations}, events={result.n_events}, iterations={result.n_iterations}")
    print(f"log-likelihood = {result.log_likelihood:.6f}")
    print(
        f"likelihood-ratio chi-square({len(result.coefficients)}) = "
        f"{result.likelihood_ratio_statistic:.3f} (p={result.likelihood_ratio_p_value:.3g})"
    )
    print(f"shape = {result.shape:.4f} (sigma = {result.sigma:.4f})")
    print(
        f"(Intercept): coef={result.intercept:.4f} se={result.std_err_intercept:.4f} "
        f"scale={result.scale:.4f} "
        f"[{np.exp(result.ci_lower_intercept):.4f}, {np.exp(result.ci_upper_intercept):.4f}] "
        f"p={result.intercept_p_value:.3g}"
    )
    for i, name in enumerate(result.feature_names):
        print(
            f"{name}: coef={result.coefficients[i]:.4f} se={result.std_err[i]:.4f} "
            f"TR={result.acceleration_factors[i]:.4f} "
            f"[{result.acceleration_factor_ci_lower[i]:.4f}, "
            f"{result.acceleration_factor_ci_upper[i]:.4f}] "
            f"p={result.p_values[i]:.3g}"
        )
    print(
        f"log(sigma): coef={result.log_sigma:.4f} se={result.std_err_log_sigma:.4f} "
        f"p={result.log_sigma_p_value:.3g}"
    )

    with open(args.out, "w", newline="", encoding="utf-8") as handle:
        handle.write(
            "parameter,estimate,std_err,z,p_value,time_ratio,tr_ci_lower,tr_ci_upper\n"
        )
        handle.write(
            f"(Intercept),{result.intercept:.10g},{result.std_err_intercept:.10g},"
            f"{result.intercept_z:.10g},{result.intercept_p_value:.10g},"
            f"{result.scale:.10g},{np.exp(result.ci_lower_intercept):.10g},"
            f"{np.exp(result.ci_upper_intercept):.10g}\n"
        )
        for i, name in enumerate(result.feature_names):
            handle.write(
                f"{name},{result.coefficients[i]:.10g},{result.std_err[i]:.10g},"
                f"{result.z_scores[i]:.10g},{result.p_values[i]:.10g},"
                f"{result.acceleration_factors[i]:.10g},"
                f"{result.acceleration_factor_ci_lower[i]:.10g},"
                f"{result.acceleration_factor_ci_upper[i]:.10g}\n"
            )
        handle.write(
            f"log(sigma),{result.log_sigma:.10g},{result.std_err_log_sigma:.10g},"
            f"{result.log_sigma_z:.10g},{result.log_sigma_p_value:.10g},,,\n"
        )
    print(f"wrote Weibull AFT estimates to {args.out}")


def run_finegray(args):
    if not args.group_col and not args.covariate_cols:
        raise SystemExit("finegray needs --group-col and/or --covariate-cols")
    if args.cause < 1:
        raise SystemExit("cause must be a positive integer")
    durations, _events, groups, extras, event_types = _load(args, with_event_types=True)
    design, names = _design_matrix(
        groups, extras, group_col=args.group_col, covariate_cols=args.covariate_cols
    )
    if design is None:
        raise SystemExit("finegray produced an empty design matrix")
    try:
        result = fit_fine_gray(
            durations, event_types, design, cause=args.cause, feature_names=names
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    print(
        f"cause {result.cause}: n={result.n_observations}, events={result.n_events}, "
        f"iterations={result.n_iterations}"
    )
    print(f"log partial likelihood = {result.log_partial_likelihood:.6f}")
    print(
        f"likelihood-ratio chi-square({len(result.coefficients)}) = "
        f"{result.likelihood_ratio_statistic:.3f} (p={result.likelihood_ratio_p_value:.3g})"
    )
    for i, name in enumerate(result.feature_names):
        print(
            f"{name}: coef={result.coefficients[i]:.4f} se={result.std_err[i]:.4f} "
            f"SHR={result.subdistribution_hazard_ratios[i]:.4f} "
            f"[{result.subdistribution_hazard_ratio_ci_lower[i]:.4f}, "
            f"{result.subdistribution_hazard_ratio_ci_upper[i]:.4f}] "
            f"p={result.p_values[i]:.3g}"
        )

    with open(args.out, "w", newline="", encoding="utf-8") as handle:
        handle.write(
            "covariate,coefficient,std_err,z,p_value,"
            "subdistribution_hazard_ratio,shr_ci_lower,shr_ci_upper\n"
        )
        for i, name in enumerate(result.feature_names):
            handle.write(
                f"{name},{result.coefficients[i]:.10g},{result.std_err[i]:.10g},"
                f"{result.z_scores[i]:.10g},{result.p_values[i]:.10g},"
                f"{result.subdistribution_hazard_ratios[i]:.10g},"
                f"{result.subdistribution_hazard_ratio_ci_lower[i]:.10g},"
                f"{result.subdistribution_hazard_ratio_ci_upper[i]:.10g}\n"
            )
    print(f"wrote Fine-Gray estimates to {args.out}")


def _print_rmst_row(name, result):
    print(
        f"{name}: RMST={result.rmst:.3f} (SE={result.std_err:.3f}) "
        f"[{result.ci_lower:.3f}, {result.ci_upper:.3f}]  RMTL={result.rmtl:.3f}"
    )


def run_rmst(args):
    durations, events, groups, _ = _load(args)
    labels = _cohort_labels(args, groups)
    tau = _shared_tau(durations, groups, labels, args.tau)
    fits = _rmst_fits(durations, events, groups, labels, tau)
    difference = _rmst_difference(durations, events, groups, labels, tau)

    print(f"tau = {tau:.6g}")
    for label, result in zip(labels, fits):
        _print_rmst_row(label, result)
    if difference is not None:
        print(
            f"difference ({labels[0]} - {labels[1]}) = {difference.difference:.3f} "
            f"(SE={difference.std_err:.3f})"
        )
        print(f"z = {difference.z_score:.3f}")
        print(f"p-value = {difference.p_value:.3g}")

    if args.out:
        with open(args.out, "w", newline="", encoding="utf-8") as handle:
            handle.write(
                "cohort,tau,rmst,rmtl,std_err,ci_lower,ci_upper,n_observations,n_events\n"
            )
            for label, result in zip(labels, fits):
                handle.write(
                    f"{label},{result.tau:.10g},{result.rmst:.10g},{result.rmtl:.10g},"
                    f"{result.std_err:.10g},{result.ci_lower:.10g},{result.ci_upper:.10g},"
                    f"{result.n_observations},{result.n_events}\n"
                )
            if difference is not None:
                handle.write(
                    f"{labels[0]}-{labels[1]},{difference.tau:.10g},"
                    f"{difference.difference:.10g},,{difference.std_err:.10g},"
                    f"{difference.ci_lower:.10g},{difference.ci_upper:.10g},,\n"
                )
        print(f"wrote RMST estimates to {args.out}")


def _print_cif_row(name, curve):
    if curve.time.size == 0:
        print(f"{name}, cause {curve.cause}: events=0, CIF=0")
        return
    print(
        f"{name}, cause {curve.cause}: events={curve.n_events}, "
        f"CIF={curve.incidence[-1]:.3f} (SE={curve.std_err[-1]:.3f})"
    )


def run_cif(args):
    durations, _events, groups, _, event_types = _load(args, with_event_types=True)
    labels = _cohort_labels(args, groups)
    causes = observed_causes(event_types)
    if args.cause is not None:
        if args.cause < 1:
            raise SystemExit("cause must be a positive integer")
        causes = (args.cause,)
    if not causes:
        raise SystemExit("no competing events found in the event column")

    fits = []
    for label in labels:
        mask = _cohort_mask(groups, label)
        for cause in causes:
            try:
                curve = fit_cumulative_incidence(
                    durations[mask], event_types[mask], cause=cause
                )
            except ValueError as exc:
                raise SystemExit(str(exc)) from exc
            fits.append((label, curve))
            _print_cif_row(label, curve)

    if len(labels) >= 2:
        for cause in causes:
            try:
                result = gray_test_groups(durations, event_types, groups, cause=cause)
            except ValueError as exc:
                raise SystemExit(str(exc)) from exc
            print(
                f"Gray cause {cause}: chi-square({result.degrees_of_freedom}) = "
                f"{result.statistic:.3f}"
            )
            print(f"Gray cause {cause}: p-value = {result.p_value:.3g}")

    if args.out:
        with open(args.out, "w", newline="", encoding="utf-8") as handle:
            header = "time,cause,incidence,std_err,ci_lower,ci_upper"
            if args.group_col:
                handle.write(header + ",cohort\n")
            else:
                handle.write(header + "\n")
            for label, curve in fits:
                for i in range(curve.time.size):
                    row = (
                        f"{curve.time[i]:.6f},{curve.cause},"
                        f"{curve.incidence[i]:.6f},{curve.std_err[i]:.6f},"
                        f"{curve.ci_lower[i]:.6f},{curve.ci_upper[i]:.6f}"
                    )
                    if args.group_col:
                        row += f",{label}"
                    handle.write(row + "\n")
        print(f"wrote cumulative incidence curves to {args.out}")


COMMANDS = {
    "generate": run_generate,
    "fit": run_fit,
    "compare": run_compare,
    "report": run_report,
    "rmst": run_rmst,
    "cif": run_cif,
    "cox": run_cox,
    "aalen": run_aalen,
    "aft": run_aft,
    "finegray": run_finegray,
}


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    COMMANDS[args.command](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())