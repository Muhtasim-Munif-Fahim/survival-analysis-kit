import numpy as np
import pytest
from scipy.stats import norm

from survival_kit import band_critical_value, fit_kaplan_meier, km_confidence_band


def _data(n=200, seed=0, rate=0.5, censor_rate=0.3):
    rng = np.random.default_rng(seed)
    t = rng.exponential(1 / rate, n)
    c = rng.exponential(1 / censor_rate, n)
    return np.minimum(t, c), (t <= c).astype(int)


def test_hall_wellner_full_range_is_kolmogorov_quantile():
    # sup over [0, 1) of |Brownian bridge| has the Kolmogorov distribution
    assert band_critical_value("hall-wellner", 0.0, 0.9999) == pytest.approx(1.3581, abs=0.02)
    assert band_critical_value("hall-wellner", 0.0, 0.9999, confidence_level=0.99) == pytest.approx(1.6276, abs=0.03)


def test_equal_precision_degenerate_interval_is_pointwise_z():
    c = band_critical_value("equal-precision", 0.4, 0.4 + 1e-9)
    assert c == pytest.approx(norm.ppf(0.975), abs=0.03)


def test_critical_values_grow_with_interval_and_are_reproducible():
    narrow = band_critical_value("equal-precision", 0.3, 0.6)
    wide = band_critical_value("equal-precision", 0.05, 0.95)
    assert norm.ppf(0.975) < narrow < wide
    assert band_critical_value("equal-precision", 0.3, 0.6) == narrow
    # grid refinement barely moves the corrected value
    fine = band_critical_value("equal-precision", 0.05, 0.95, n_grid=3000, n_simulations=5000)
    coarse = band_critical_value("equal-precision", 0.05, 0.95, n_grid=200, n_simulations=5000)
    assert fine == pytest.approx(coarse, abs=0.1)


@pytest.mark.parametrize("method", ["hall-wellner", "equal-precision"])
@pytest.mark.parametrize("transform", ["log-log", "linear"])
def test_band_contains_km_and_is_wider_than_pointwise(method, transform):
    d, e = _data()
    band = km_confidence_band(d, e, method=method, transform=transform, n_simulations=4000, n_grid=300)
    km = fit_kaplan_meier(d, e, ci_method=transform)
    assert np.all(band.lower <= band.survival + 1e-12) and np.all(band.survival <= band.upper + 1e-12)
    assert np.all((band.lower >= 0) & (band.upper <= 1))
    idx = np.searchsorted(km.time, band.time)
    np.testing.assert_allclose(band.survival, km.survival[idx])
    if method == "equal-precision":
        # same shape as the pointwise interval, scaled by a larger critical value
        assert np.all(band.upper - band.lower >= km.ci_upper[idx] - km.ci_lower[idx] - 1e-12)
    assert 0 < band.a_lower < band.a_upper < 1
    assert band.t_lower == band.time[0] and band.t_upper == band.time[-1]


@pytest.mark.parametrize("method", ["hall-wellner", "equal-precision"])
def test_simultaneous_coverage_near_nominal(method):
    rate = 0.5
    hits = []
    for rep in range(150):
        d, e = _data(n=250, seed=100 + rep, rate=rate)
        tl, tu = 0.3, 2.5
        band = km_confidence_band(
            d, e, method=method, t_lower=tl, t_upper=tu, n_simulations=2000, n_grid=200, seed=1
        )
        # check the true curve right after each event time and at the band ends
        grid = np.unique(np.r_[band.time, tl, tu])
        grid = grid[(grid >= tl) & (grid <= tu) & (grid >= band.time[0])]
        hits.append(band.contains(grid, np.exp(-rate * grid)))
    cover = np.mean(hits)
    assert 0.88 <= cover <= 1.0


def test_explicit_range_and_validation():
    d, e = _data()
    band = km_confidence_band(d, e, t_lower=0.5, t_upper=2.0, n_simulations=1000, n_grid=100)
    assert band.time.min() >= 0.5 and band.time.max() <= 2.0
    with pytest.raises(ValueError, match="inside"):
        band.contains([5.0], [0.1])
    with pytest.raises(ValueError, match="method"):
        km_confidence_band(d, e, method="nair")
    with pytest.raises(ValueError, match="transform"):
        km_confidence_band(d, e, transform="arcsine")
    with pytest.raises(ValueError, match="t_lower"):
        km_confidence_band(d, e, t_lower=3.0, t_upper=1.0)
    with pytest.raises(ValueError, match="a_lower > 0"):
        band_critical_value("equal-precision", 0.0, 0.5)
    with pytest.raises(ValueError, match="a_lower <= a_upper"):
        band_critical_value("hall-wellner", 0.6, 0.5)
    with pytest.raises(ValueError, match="two distinct"):
        km_confidence_band([1.0, 2.0], [1, 0])
