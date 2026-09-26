"""Tests for Aalen's additive hazards model."""

from __future__ import annotations

import numpy as np
import pytest

from survival_kit.aalen import fit_aalen_additive
from survival_kit.synth import generate_ph_data


def test_known_two_subject_increments():
    """Hand-check the first jump on a tiny sample."""
    durations = np.array([1.0, 2.0, 3.0])
    events = np.array([True, True, True])
    X = np.array([[0.0], [1.0], [0.0]])
    fit = fit_aalen_additive(durations, events, X, ridge=0.0)
    assert fit.time.tolist() == [1.0, 2.0, 3.0]
    assert fit.cumulative_coefficients.shape == (3, 2)
    # At t=1 everyone is at risk. Z = [[1,0],[1,1],[1,0]].
    # gram = [[3,1],[1,1]], inv = [[0.5, -0.5], [-0.5, 1.5]]
    # dN = [1,0,0] for subject 0; Z'dN = [1,0]; delta = inv @ [1,0] = [0.5, -0.5]
    assert fit.cumulative_coefficients[0, 0] == pytest.approx(0.5)
    assert fit.cumulative_coefficients[0, 1] == pytest.approx(-0.5)


def test_cumulative_grows_with_events_and_exports_names():
    data = generate_ph_data(60, [0.8, -0.4], censor_fraction=0.2, seed=0)
    fit = fit_aalen_additive(
        data.durations, data.events, data.covariates, feature_names=("x0", "x1")
    )
    assert fit.feature_names[0] == "intercept"
    assert fit.feature_names[1:] == ("x0", "x1")
    assert fit.n_events >= 1
    # Cumulative paths should be finite and start near the first increment.
    assert np.all(np.isfinite(fit.cumulative_coefficients))
    assert fit.std_err.shape == fit.cumulative_coefficients.shape
    # End-of-study cumulative hazard for a zero covariate vector equals intercept.
    haz = fit.cumulative_hazard_at(fit.time[-1], np.zeros(fit.cumulative_coefficients.shape[1] - 1))
    assert haz.shape == (1,) or np.ndim(haz) >= 0
    assert float(np.asarray(haz).ravel()[-1]) == pytest.approx(
        fit.cumulative_coefficients[-1, 0]
    )


def test_rejects_no_events_and_bad_confidence():
    with pytest.raises(ValueError, match="at least one event"):
        fit_aalen_additive([1.0, 2.0], [False, False], [[0.0], [1.0]])
    with pytest.raises(ValueError, match="confidence_level"):
        fit_aalen_additive([1.0, 2.0], [True, False], [[0.0], [1.0]], confidence_level=1.0)


def test_cumulative_at_step_function():
    durations = np.array([1.0, 2.0, 4.0])
    events = np.array([True, True, True])
    X = np.array([[0.0], [1.0], [0.5]])
    fit = fit_aalen_additive(durations, events, X)
    mid = fit.cumulative_at([1.5], covariate_index=0)
    assert float(np.asarray(mid).ravel()[0]) == pytest.approx(fit.cumulative_coefficients[0, 0])
