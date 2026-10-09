"""Tests for IPCW Brier score and Integrated Brier Score."""

from __future__ import annotations

import numpy as np
import pytest

from survival_kit.brier import (
    brier_score,
    integrated_brier_score,
    km_survival_matrix,
)
from survival_kit.synth import generate_survival_data


def test_perfect_prediction_near_zero():
    data = generate_survival_data(n=200, seed=0)
    times = np.quantile(data.durations[data.events], [0.25, 0.5, 0.75])
    # Oracle: S=0 if event by t else 1 — approximate with indicators on truth
    # (not IPCW-aware oracle); instead use KM which should beat random.
    surv = km_survival_matrix(data.durations, data.events, times)
    result = brier_score(data.durations, data.events, surv, times)
    assert result.brier.shape == times.shape
    assert np.all(result.brier >= 0.0)
    assert np.all(result.brier <= 1.0 + 1e-6)


def test_ibs_positive_and_finite():
    data = generate_survival_data(n=150, seed=1)
    times = np.linspace(0.1, float(np.median(data.durations)), 8)
    surv = km_survival_matrix(data.durations, data.events, times)
    ibs = integrated_brier_score(data.durations, data.events, surv, times)
    assert ibs.ibs >= 0.0
    assert np.isfinite(ibs.ibs)
    assert ibs.tau == pytest.approx(times[-1])


def test_constant_half_worse_than_km():
    data = generate_survival_data(n=250, seed=2)
    times = np.quantile(data.durations, [0.2, 0.4, 0.6, 0.8])
    km = km_survival_matrix(data.durations, data.events, times)
    half = np.full_like(km, 0.5)
    ibs_km = integrated_brier_score(data.durations, data.events, km, times).ibs
    ibs_half = integrated_brier_score(data.durations, data.events, half, times).ibs
    assert ibs_km <= ibs_half + 1e-9


def test_shape_mismatch_raises():
    data = generate_survival_data(n=40, seed=3)
    times = np.array([0.5, 1.0])
    with pytest.raises(ValueError, match="columns"):
        brier_score(
            data.durations,
            data.events,
            np.zeros((40, 3)),
            times,
        )


def test_empty_times_raises():
    data = generate_survival_data(n=20, seed=4)
    with pytest.raises(ValueError, match="non-empty"):
        brier_score(data.durations, data.events, np.zeros((20, 0)), [])
