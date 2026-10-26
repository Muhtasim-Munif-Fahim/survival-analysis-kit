"""Tests for the Ramlau-Hansen kernel-smoothed hazard."""

import csv

import numpy as np
import pytest
from scipy.integrate import quad

from survival_kit import (
    boundary_kernel,
    fit_nelson_aalen,
    fit_smoothed_hazard,
    select_bandwidth_cv,
)
from survival_kit.cli import main
from survival_kit.hazard_smoothing import KERNELS, kernel


def _weibull(n, shape, seed, censor_rate=0.0):
    rng = np.random.default_rng(seed)
    t = rng.weibull(shape, size=n)
    if censor_rate > 0:
        c = rng.exponential(1.0 / censor_rate, size=n)
        return np.minimum(t, c), t <= c
    return t, np.ones(n, dtype=bool)


@pytest.mark.parametrize("name", KERNELS)
def test_kernels_are_densities(name):
    assert quad(lambda x: float(kernel(x, name)), -1, 1)[0] == pytest.approx(1.0)
    assert quad(lambda x: x * float(kernel(x, name)), -1, 1)[0] == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("name", KERNELS)
@pytest.mark.parametrize("q", [0.0, 0.25, 0.6, 1.0])
def test_boundary_kernels_have_unit_mass_and_zero_first_moment(name, q):
    mass = quad(lambda x: float(boundary_kernel(x, q, name)), -1, q)[0]
    first = quad(lambda x: x * float(boundary_kernel(x, q, name)), -1, q)[0]
    assert mass == pytest.approx(1.0, abs=1e-10)
    assert first == pytest.approx(0.0, abs=1e-10)
    if q < 1:
        assert float(boundary_kernel(q + 0.01, q, name)) == 0.0


def test_hand_computed_value():
    d, e = [3, 5, 5, 7, 9], [1, 0, 1, 1, 0]
    # dH = 0.2, 0.25, 0.5 at 3, 5, 7. Uniform kernel b = 2 at t = 5:
    # (1/2) * 0.5 * (0.2 + 0.25 + 0.5) since |5 - t_j| / 2 <= 1 for all.
    fit = fit_smoothed_hazard(d, e, bandwidth=2.0, kernel_name="uniform", times=[5.0],
                              boundary_correction=False)
    assert fit.hazard[0] == pytest.approx(0.25 * 0.95)
    na = fit_nelson_aalen(d, e)
    var_inc = np.diff(np.r_[0.0, na.std_err**2])
    assert fit.std_err[0] ** 2 == pytest.approx(0.25 * 0.25 * var_inc.sum())


def test_recovers_constant_hazard_including_boundaries():
    d, e = _weibull(6000, 1.0, seed=0, censor_rate=0.3)
    grid = np.array([0.0, 0.1, 0.5, 1.0, 1.5])
    fit = fit_smoothed_hazard(d, e, bandwidth=0.4, times=grid)
    assert np.allclose(fit.hazard, 1.0, atol=0.15)
    raw = fit_smoothed_hazard(d, e, bandwidth=0.4, times=grid, boundary_correction=False)
    # Without correction the symmetric kernel loses about half its mass at 0.
    assert raw.hazard[0] == pytest.approx(0.5, abs=0.1)
    assert abs(fit.hazard[0] - 1.0) < abs(raw.hazard[0] - 1.0)
    assert np.all(fit.ci_lower <= fit.hazard) and np.all(fit.hazard <= fit.ci_upper)


def test_recovers_increasing_weibull_hazard():
    d, e = _weibull(8000, 2.0, seed=1)
    grid = np.linspace(0.2, 1.6, 8)
    fit = fit_smoothed_hazard(d, e, bandwidth=0.25, kernel_name="biweight", times=grid)
    assert np.allclose(fit.hazard, 2.0 * grid, rtol=0.15, atol=0.1)


def test_coverage_of_log_confidence_interval():
    hits = 0
    for seed in range(60):
        d, e = _weibull(400, 1.0, seed=100 + seed)
        fit = fit_smoothed_hazard(d, e, bandwidth=0.3, times=[0.6])
        hits += fit.ci_lower[0] <= 1.0 <= fit.ci_upper[0]
    assert hits / 60 >= 0.8


def test_default_grid_bandwidth_and_validation():
    d, e = _weibull(300, 1.5, seed=2)
    fit = fit_smoothed_hazard(d, e)
    assert fit.time.size == 101 and fit.time[0] == 0.0
    assert fit.time[-1] == pytest.approx(d[e].max())
    assert fit.bandwidth > 0 and np.all(fit.hazard >= 0)
    assert set(fit.to_dict()) >= {"time", "hazard", "bandwidth"}
    with pytest.raises(ValueError):
        fit_smoothed_hazard(d, e, bandwidth=-1)
    with pytest.raises(ValueError):
        fit_smoothed_hazard(d, e, bandwidth=d.max())
    with pytest.raises(ValueError):
        fit_smoothed_hazard(d, e, kernel_name="gauss")
    with pytest.raises(ValueError):
        fit_smoothed_hazard(d, e, times=[-1.0])
    with pytest.raises(ValueError):
        fit_smoothed_hazard([1.0, 2.0], [0, 0])
    with pytest.raises(ValueError):
        boundary_kernel(0.0, 1.5)


def test_cross_validation_prefers_moderate_bandwidth():
    d, e = _weibull(1500, 1.0, seed=3, censor_rate=0.2)
    cands = [0.02, 0.1, 0.3, 0.6]
    res = select_bandwidth_cv(d, e, bandwidths=cands)
    assert res.scores.shape == (4,)
    assert res.best_bandwidth in (0.1, 0.3, 0.6)
    assert res.best_bandwidth != 0.02
    auto = select_bandwidth_cv(d, e)
    assert auto.bandwidths.size == 20 and np.isfinite(auto.scores).any()
    with pytest.raises(ValueError):
        select_bandwidth_cv(d, e, bandwidths=[0.0])


def test_cli_hazard(tmp_path, capsys):
    rng = np.random.default_rng(4)
    path = tmp_path / "d.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["duration", "event", "group"])
        for i in range(400):
            w.writerow([rng.exponential(1.0 if i % 2 else 0.5), int(rng.random() < 0.8),
                        "a" if i % 2 else "b"])
    out = tmp_path / "hz.csv"
    assert main(["hazard", "--data", str(path), "--group-col", "group", "--bandwidth", "0.3",
                 "--points", "11", "--out", str(out)]) == 0
    text = capsys.readouterr().out
    assert "a: n=200" in text and "bandwidth=0.3" in text
    rows = list(csv.DictReader(open(out)))
    assert len(rows) == 22 and rows[0]["cohort"] in ("a", "b")
    assert main(["hazard", "--data", str(path), "--bandwidth", "cv", "--kernel", "biweight",
                 "--out", str(out)]) == 0
    assert "biweight" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        main(["hazard", "--data", str(path), "--bandwidth", "100", "--out", str(out)])
