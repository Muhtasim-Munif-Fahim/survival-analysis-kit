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
- **Synthetic data generator** (`synth`) drawing seeded Weibull/exponential
  latents in two arms, with independent exponential censoring calibrated to a
  target censored fraction, plus a Weibull PH covariate generator and CSV
  input/output helpers
- **Markdown reports** (`report`) and a small CLI (`cli`) wiring the pipeline
  together: `generate -> fit -> compare -> cox -> report`

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
    python -m survival_kit.cli cox --data sample.csv --group-col group --out cox.csv
    python -m survival_kit.cli report --data sample.csv --group-col group \
        --title "Two-arm demo" --out report.md

If your table uses other column names, point `fit`/`compare`/`cox`/`report` at
them with `--time-col` and `--event-col`. Numeric covariates go to `cox` and
`report` via `--covariate-cols`. A numeric risk-score column can be added to
reports with `--score-col`; scores must be oriented so larger values predict
earlier events.

## Quickstart (Python)

```python
from survival_kit import (
    fit_cox_ph,
    fit_kaplan_meier,
    generate_ph_data,
    generate_survival_data,
    log_rank_test_groups,
)

data = generate_survival_data(500, shape=1.0, scale=3.0, group_scale_ratio=2.0, seed=7)
curve = fit_kaplan_meier(data.durations[data.groups == "control"], data.events[data.groups == "control"])
print(curve.median(), curve.at([1.0, 2.0, 4.0]))

test = log_rank_test_groups(data.durations, data.events, data.groups)
print(test.statistic, test.p_value)

ph = generate_ph_data(800, coefficients=[0.7, -0.4], shape=1.3, scale=4.0, seed=7)
cox = fit_cox_ph(ph.durations, ph.events, ph.covariates, feature_names=("x0", "x1"))
print(cox.coefficients, cox.hazard_ratios)
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

## Testing

    python -m pytest tests -q

## License

MIT. See [LICENSE](LICENSE).