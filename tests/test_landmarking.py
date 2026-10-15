"""Tests for landmarking dynamic survival prediction."""

from __future__ import annotations

import numpy as np
import pytest

from survival_kit.landmarking import (
    landmark_dataset,
    landmark_kaplan_meier,
    landmark_survival,
)
from survival_kit.synth import generate_survival_data


def test_landmark_dataset_truncates():
    durations = np.array([1.0, 2.0, 5.0, 8.0])
    events = np.array([1, 0, 1, 1])
    rem, ev, _ = landmark_dataset(durations, events, landmark=2.0)
    assert rem.tolist() == pytest.approx([3.0, 6.0])
    assert ev.tolist() == [True, True]


def test_landmark_km_conditional_survival():
    data = generate_survival_data(n=200, seed=0)
    # API may return object or tuple - inspect
    if hasattr(data, "durations"):
        d, e = data.durations, data.events
    elif isinstance(data, tuple):
        d, e = data[0], data[1]
    else:
        d, e = data["durations"], data["events"]
    res = landmark_kaplan_meier(d, e, landmarks=[0.5, 1.0])
    assert len(res) == 2
    assert res[0].n_at_risk >= res[1].n_at_risk
    s = res[0].survival_at(1.0)
    assert 0.0 <= s <= 1.0


def test_landmark_survival_horizons():
    data = generate_survival_data(n=150, seed=1)
    if hasattr(data, "durations"):
        d, e = data.durations, data.events
    else:
        d, e = data[0], data[1]
    out = landmark_survival(d, e, landmark=0.5, horizons=[0.5, 1.0, 2.0])
    assert out["survival"].shape == (3,)
    # Survival should be non-increasing in horizon
    assert out["survival"][0] >= out["survival"][-1] - 1e-9


def test_landmark_no_subjects():
    with pytest.raises(ValueError, match="no subjects"):
        landmark_dataset([1.0, 1.5], [1, 1], landmark=10.0)
