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
- **Synthetic data generator** (`synth`) drawing seeded Weibull/exponential
  latents in two arms, with independent exponential censoring calibrated to a
  target censored fraction, plus CSV input/output helpers
- **Markdown reports** (`report`) and a small CLI (`cli`) wiring the pipeline
  together: `generate -> fit -> compare -> report`

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
    python -m survival_kit.cli report --data sample.csv --group-col group \
        --title "Two-arm demo" --out report.md

If your table uses other column names, point `fit`/`compare`/`report` at them
with `--time-col` and `--event-col`. A numeric risk-score column can be added
to reports with `--score-col`; scores must be oriented so larger values
predict earlier events.

## Quickstart (Python)

```python
from survival_kit import (
    concordance_index,
    fit_kaplan_meier,
    generate_survival_data,
    log_rank_test_groups,
)

data = generate_survival_data(500, shape=1.0, scale=3.0, group_scale_ratio=2.0, seed=7)
curve = fit_kaplan_meier(data.durations[data.groups == "control"], data.events[data.groups == "control"])
print(curve.median(), curve.at([1.0, 2.0, 4.0]))

test = log_rank_test_groups(data.durations, data.events, data.groups)
print(test.statistic, test.p_value)
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

## Testing

    python -m pytest tests -q

## License

MIT. See [LICENSE](LICENSE).