"""End-to-end demo: simulate two arms, fit curves, compare, and report.

Run from the repository root:

    python examples/run_demo.py

Artifacts land in ``examples/output/``.
"""

import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from survival_kit.cli import main  # noqa: E402
from survival_kit.concordance import concordance_index  # noqa: E402
from survival_kit.synth import (  # noqa: E402
    generate_competing_risks_data,
    generate_survival_data,
    save_csv,
)


def build_demo_dataset(output_dir):
    data = generate_survival_data(
        n=500,
        shape=1.2,
        scale=5.0,
        censor_fraction=0.25,
        group_scale_ratio=2.0,
        seed=2026,
    )
    # A naive prognostic score: control-arm membership predicts earlier events.
    rng = np.random.default_rng(2026)
    scores = np.where(data.groups == "control", 1.0, 0.0)
    scores = scores + rng.normal(scale=0.35, size=data.durations.shape[0])

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "demo_data.csv"
    save_csv(data, csv_path, extra_columns={"score": scores})
    return csv_path, scores, data


def run_demo():
    output_dir = Path(__file__).resolve().parent / "output"
    csv_path, scores, data = build_demo_dataset(output_dir)
    shared = ["--data", str(csv_path)]

    main(["fit", *shared, "--group-col", "group", "--out", str(output_dir / "demo_curves.csv")])
    main(["compare", *shared, "--group-col", "group"])
    main(["rmst", *shared, "--group-col", "group", "--out", str(output_dir / "demo_rmst.csv")])
    main(["cox", *shared, "--group-col", "group", "--out", str(output_dir / "demo_cox.csv")])
    main(
        [
            "report",
            *shared,
            "--group-col", "group",
            "--score-col", "score",
            "--title", "survival-analysis-kit demo",
            "--out", str(output_dir / "demo_report.md"),
        ]
    )

    result = concordance_index(data.durations, data.events, scores)
    print(f"c-index of the arm-membership score: {result.c_index:.3f}")

    cr = generate_competing_risks_data(
        n=500,
        cause_rates=(0.45, 0.30),
        group_rate_ratios=(0.4, 1.0),
        censor_fraction=0.2,
        seed=2026,
    )
    cr_csv = output_dir / "demo_cr.csv"
    save_csv(cr, cr_csv)
    main(
        [
            "cif",
            "--data", str(cr_csv),
            "--group-col", "group",
            "--out", str(output_dir / "demo_cif.csv"),
        ]
    )
    main(
        [
            "report",
            "--data", str(cr_csv),
            "--group-col", "group",
            "--title", "survival-analysis-kit competing-risks demo",
            "--out", str(output_dir / "demo_cr_report.md"),
        ]
    )
    print(f"artifacts written to {output_dir}")


if __name__ == "__main__":
    run_demo()