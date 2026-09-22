# survival-analysis-kit

A lightweight toolkit for survival analysis on right-censored data, built on
NumPy and SciPy.

## Features

- **Kaplan-Meier estimator** (`kaplan_meier`) with Greenwood variance and
  complementary log-log or linear confidence bands
- **Log-rank tests** (`logrank`) for two-sample and k-group comparisons using
  the hypergeometric covariance of observed minus expected deaths
- **Nelson-Aalen estimator** (`nelson_aalen`) for the cumulative hazard with
  Aalen or Greenwood pointwise variances
- **Harrell's concordance index** (`concordance`) with an explicit policy for
  tied times (never comparable) and tied scores (half credit)
- **Cox proportional hazards** (`cox`) maximizing the Breslow partial
  likelihood for right-censored data with covariates, returning coefficients,
  hazard ratios, Wald intervals, and a Breslow baseline cumulative hazard
- **Restricted mean survival time** (`rmst`) integrating a Kaplan-Meier curve
  up to a truncation time ``tau``, with Greenwood standard errors and a
  two-group RMST difference test
- **Weibull accelerated failure time** (`aft`) maximizing the right-censored
  extreme-value likelihood. Kaplan-Meier and log-rank already ship in this
  kit, so the parametric addition is a Weibull AFT fit (time ratios,
  shape/scale, Wald intervals, and a likelihood-ratio test) rather than a
  second nonparametric curve
- **Competing-risks cumulative incidence** (`competing_risks`) via the
  Aalen-Johansen / cause-specific CIF, Aalen (Coviello-Boggess) pointwise
  variances, and Gray's k-sample test of subdistribution equality
- **Fine-Gray subdistribution hazard** (`fine_gray`) for one competing cause.
  Nelson-Aalen cumulative hazard estimation, with optional Aalen or Greenwood
  variance, is already provided by `nelson_aalen`, so this addition is the
  proportional subdistribution model (IPCW weighted Breslow partial
  likelihood, subdistribution hazard ratios, and a model CIF) rather than
  another cumulative-hazard estimator
- **Synthetic data generator** (`synth`) drawing seeded Weibull/exponential
  latents in two arms, with independent exponential censoring calibrated to a
  target censored fraction, plus a Weibull PH covariate generator, an
  independent-exponential competing-risks generator, and CSV input/output
  helpers
- **Markdown reports** (`report`) and a small CLI (`cli`) wiring the pipeline
  together: `generate -> fit -> compare -> rmst -> cox -> aft -> cif -> finegray -> report`

## Installation

No release yet; run from a checkout:

    pip install -r requirements.txt

`pandas` is optional and not required by any current module.

## Quickstart (CLI)

Generate a two-arm sample where the treatment arm survives longer, fit
per-group Kaplan-Meier curves, test the difference, and write a report:

    python -m survival_kit.cli generate --n 400 --shape 1.2 --scale 5.0 \
        --censor-fraction 0.2 --group-scale-ratio 2.0 --seed 42 --out sample.csv
    python -m survival_kit.cli fit --data sample.csv --group-col group \
        --out curves.csv
    python -m survival_kit.cli compare --data sample.csv --group-col group
    python -m survival_kit.cli rmst --data sample.csv --group-col group --tau 8
    python -m survival_kit.cli cox --data sample.csv --group-col group --out cox.csv
    python -m survival_kit.cli aft --data sample.csv --group-col group --out aft.csv
    python -m survival_kit.cli report --data sample.csv --group-col group \
        --tau 8 --title "Two-arm demo" --out report.md

Competing events use integer codes in the `event` column (`0` = censored,
`1`, `2`, ... = causes). `--cause-rates` switches `generate` to independent
exponential causes; `cif` estimates Aalen-Johansen curves and, with two or
more groups, Gray's test. Reports include a cumulative-incidence section
whenever more than one cause is present:

    python -m survival_kit.cli generate --n 500 --cause-rates 0.45 0.30 \
        --cause-rate-ratios 0.4 1.0 --censor-fraction 0.2 --seed 42 --out cr.csv
    python -m survival_kit.cli cif --data cr.csv --group-col group --out cif.csv
    python -m survival_kit.cli finegray --data cr.csv --group-col group \
        --cause 1 --out finegray.csv
    python -m survival_kit.cli report --data cr.csv --group-col group \
        --title "Competing-risks demo" --out cr_report.md

If your table uses other column names, point `fit`/`compare`/`cox`/`aft`/`cif`/`finegray`/`report`
at them with `--time-col` and `--event-col`. Numeric covariates go to `cox`, `aft`,
`finegray`, and `report` via `--covariate-cols`. A numeric risk-score column can be added to
reports with `--score-col`; scores must be oriented so larger values predict
earlier events.

## Quickstart (Python)

```python
from survival_kit import (
    fit_cox_ph,
    fit_cumulative_incidence,
    fit_fine_gray,
    fit_kaplan_meier,
    fit_weibull_aft,
    generate_competing_risks_data,
    generate_ph_data,
    generate_survival_data,
    gray_test_groups,
    log_rank_test_groups,
    restricted_mean_survival_time,
    rmst_difference_test,
)

data = generate_survival_data(500, shape=1.0, scale=3.0, group_scale_ratio=2.0, seed=7)
control = data.groups == "control"
treated = data.groups == "treatment"
curve = fit_kaplan_meier(data.durations[control], data.events[control])
print(curve.median(), curve.at([1.0, 2.0, 4.0]), curve.restricted_mean(4.0))

test = log_rank_test_groups(data.durations, data.events, data.groups)
print(test.statistic, test.p_value)

rmst = restricted_mean_survival_time(data.durations[control], data.events[control], tau=4.0)
print(rmst.rmst, rmst.std_err)
diff = rmst_difference_test(
    data.durations[treated], data.events[treated],
    data.durations[control], data.events[control],
    tau=4.0,
)
print(diff.difference, diff.p_value)

ph = generate_ph_data(800, coefficients=[0.7, -0.4], shape=1.3, scale=4.0, seed=7)
cox = fit_cox_ph(ph.durations, ph.events, ph.covariates, feature_names=("x0", "x1"))
print(cox.coefficients, cox.hazard_ratios)

aft = fit_weibull_aft(ph.durations, ph.events, ph.covariates, feature_names=("x0", "x1"))
print(aft.coefficients, aft.acceleration_factors, aft.shape)
print(aft.survival_at([1.0, 2.0, 4.0], np.zeros(2)), aft.median(np.zeros(2)))

cr = generate_competing_risks_data(
    800, cause_rates=(0.45, 0.30), group_rate_ratios=(0.4, 1.0), seed=7
)
cif = fit_cumulative_incidence(cr.durations, cr.event_types, cause=1)
print(cif.at([1.0, 2.0]), cr.true_cif(1.0, cause=1))
print(gray_test_groups(cr.durations, cr.event_types, cr.groups, cause=1).p_value)

treatment = (cr.groups == "treatment").astype(float)
fg = fit_fine_gray(cr.durations, cr.event_types, treatment, cause=1)
print(fg.coefficients, fg.subdistribution_hazard_ratios)
print(fg.cumulative_incidence_at([1.0, 2.0], [0.0]))
```

See `examples/run_demo.py` for a complete end-to-end run that writes
`examples/output/demo_report.md`.

## Method notes

- Confidence bands default to the complementary log-log transform; points
  where the transform is undefined fall back to linear limits.
- The k-group log-rank statistic is the quadratic form over the first ``k-1``
  contrasts of the observed-minus-expected vector under the hypergeometric
  covariance.
- Concordance pairs sharing an observed time are skipped because their
  ordering is unidentifiable under right censoring.
- Cox PH uses the Breslow approximation for tied event times, estimates no
  intercept (absorbed into the baseline hazard), and reports ``exp(beta)`` as
  the per-unit hazard ratio. Group labels passed to the CLI are dummy-coded
  with the first level as the reference.
- Weibull AFT uses the log-linear model
  ``log T = mu + x @ beta + sigma * W`` with a Gumbel (minimum) residual
  ``W``. The Weibull shape is ``1 / sigma`` and the scale at covariate row
  ``x`` is ``exp(mu + x @ beta)``. Time ratios are ``exp(beta)``; because
  Weibull is also a PH family, the equivalent log hazard ratios are
  ``-beta / sigma``. Kaplan-Meier already supplies ``S(t)`` (and Greenwood
  intervals) and log-rank already supplies a two-/k-sample nonparametric
  test, so this module adds the parametric AFT companion rather than a
  second KM/log-rank implementation.
- RMST is the area under the Kaplan-Meier curve on ``[0, tau]``. The
  Greenwood plug-in variance is the sum of squared remaining areas after each
  event time, weighted by ``d_i / (n_i (n_i - d_i))``. The two-group test uses
  the sum of the arm-specific variances. ``tau`` must not exceed the last
  follow-up time (the earlier last follow-up, when two groups are compared);
  omitting it selects that identifiable maximum.
- Competing-risks CIF is the Aalen-Johansen estimator
  ``F_k(t) = sum_{t_j <= t} S(t_j-) d_{jk} / n_j``, where ``S`` is overall
  survival treating any event as a failure. This is the proper cause-specific
  cumulative incidence; ``1 - KM`` that censors other causes overestimates
  ``F_k``. Pointwise variances use the Aalen / Coviello-Boggess formula, which
  reduces to Greenwood's variance of ``1 - S(t)`` when only one cause is
  present. Gray's test is the Fine-Gray score test of group indicators at the
  null (IPCW weights keep competing events in the subdistribution risk set)
  and reduces to the log-rank test when there is a single cause.
- Nelson-Aalen cumulative hazard estimation is already in the kit
  (``fit_nelson_aalen``, with Aalen or Greenwood pointwise variance). The
  competing-risks regression added alongside it is the Fine-Gray
  proportional subdistribution hazard model, not a second cumulative-hazard
  estimator. For cause ``k``,
  ``lambda_k(t | x) = lambda_{k0}(t) exp(x @ beta)``, so the model CIF is
  ``F_k(t | x) = 1 - exp(-Lambda_{k0}(t) exp(x @ beta))``. The partial likelihood
  is the Breslow Cox likelihood on the subdistribution risk set. Subjects who
  fail of another cause stay in that risk set with IPCW weight
  ``G(t-) / G(T_i-)``, where ``G`` is the Kaplan-Meier estimator of the
  censoring distribution (the same weights Gray's test uses). Tied cause-``k``
  times use the Breslow denominator. ``exp(beta)`` is the per-unit
  subdistribution hazard ratio. Wald intervals invert the weighted
  partial-likelihood information and treat those censoring weights as fixed;
  they do not add the extra term for estimating ``G``. With no competing
  events the weights are one on the ordinary risk set and ``fit_fine_gray``
  matches ``fit_cox_ph``. Gray's chi-square uses the hypergeometric covariance
  of that score at the null, so it is close to the model score test but is
  not the same finite-sample statistic as the Wald or likelihood-ratio test.

## Testing

    python -m pytest tests -q

## License

MIT. See [LICENSE](LICENSE).
