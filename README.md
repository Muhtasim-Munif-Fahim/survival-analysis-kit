# survival-analysis-kit

A lightweight toolkit for survival analysis on right-censored data, built on
NumPy and SciPy.

## Scope

- Kaplan-Meier survival curve estimation with Greenwood variance and confidence bands
- Log-rank tests for two-sample and k-group comparisons
- Nelson-Aalen cumulative hazard estimation
- Harrell's concordance index for risk predictions under right censoring
- A seeded synthetic data generator (Weibull/exponential latents with independent censoring)
- Markdown evaluation reports and a small CLI tying the pieces together

## Status

Work in progress: estimators, tests, and tooling land incrementally.
The package currently ships its skeleton only.

## Development

From a checkout:

    pip install -r requirements.txt
    python -m pytest tests -q

`pandas` is optional and not required by any current module.

## License

MIT. See [LICENSE](LICENSE).