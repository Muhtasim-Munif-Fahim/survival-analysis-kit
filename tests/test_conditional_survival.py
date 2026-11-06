import numpy as np
import pytest

from survival_kit.conditional_survival import conditional_survival, mean_residual_life
from survival_kit.kaplan_meier import fit_kaplan_meier
from survival_kit.synth import generate_survival_data


def test_conditional_survival_at_zero_matches_km():
    data = generate_survival_data(n=400, seed=0)
    km = fit_kaplan_meier(data.durations, data.events)
    horizons = np.array([1.0, 2.0, 3.0])
    res = conditional_survival(0.0, horizons, curve=km)
    np.testing.assert_allclose(res.survival, km.at(horizons), rtol=1e-10)
    assert res.s_landmark == pytest.approx(1.0)


def test_conditional_survival_identity():
    rng_data = generate_survival_data(n=500, seed=1)
    km = fit_kaplan_meier(rng_data.durations, rng_data.events)
    t0, s = 1.5, 2.0
    res = conditional_survival(t0, s, curve=km)
    expected = float(km.at(t0 + s) / km.at(t0))
    assert float(res.survival[0]) == pytest.approx(expected)
    assert 0.0 <= float(res.ci_lower[0]) <= float(res.survival[0]) <= float(res.ci_upper[0]) <= 1.0


def test_mean_residual_life_at_zero_is_rmst():
    data = generate_survival_data(n=300, seed=2)
    km = fit_kaplan_meier(data.durations, data.events)
    tau = float(km.time[-1])
    mrl = mean_residual_life(0.0, curve=km, tau=tau)
    assert float(mrl.mrl[0]) == pytest.approx(km.restricted_mean(tau), rel=1e-10)


def test_mrl_decreases_for_exponential_like_data():
    # Light censoring exponential-ish: MRL should be roughly flat/decreasing.
    data = generate_survival_data(n=800, seed=3)
    km = fit_kaplan_meier(data.durations, data.events)
    tau = float(np.quantile(data.durations, 0.8))
    mrl = mean_residual_life([0.0, tau * 0.25, tau * 0.5], curve=km, tau=tau)
    assert np.all(np.isfinite(mrl.mrl))
    assert mrl.mrl[0] >= mrl.mrl[-1] - 0.5  # allow noise; should not explode upward


def test_rejects_bad_landmark():
    data = generate_survival_data(n=50, seed=4)
    with pytest.raises(ValueError):
        conditional_survival(-1.0, 1.0, durations=data.durations, events=data.events)
