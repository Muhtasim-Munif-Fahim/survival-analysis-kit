import math

import numpy as np
import pytest

from survival_kit import (
    event_probability,
    hazard_from_median,
    log_rank_test,
    logrank_power,
    logrank_required_events,
    logrank_sample_size,
)


def test_schoenfeld_textbook_value():
    # HR 0.5, two-sided 5%, 80% power, 1:1 -> 65.3 -> 66 events.
    d = logrank_required_events(0.5, exact=True)
    assert d == pytest.approx(4 * (1.959964 + 0.841621) ** 2 / math.log(0.5) ** 2, rel=1e-6)
    assert logrank_required_events(0.5) == 66


def test_freedman_value_and_ordering():
    d = logrank_required_events(0.5, method="freedman", exact=True)
    assert d == pytest.approx((1.959964 + 0.841621) ** 2 * 1.5 ** 2 / 0.25, rel=1e-6)
    assert logrank_required_events(0.5, method="freedman") == 71
    # Freedman is more conservative than Schoenfeld for HR != 1.
    for hr in (0.6, 0.75, 1.5):
        assert logrank_required_events(hr, method="freedman") >= logrank_required_events(hr)


def test_symmetry_and_unbalanced_allocation():
    assert logrank_required_events(0.5, exact=True) == pytest.approx(
        logrank_required_events(2.0, exact=True)
    )
    # 2:1 allocation needs 9/8 of the balanced events under Schoenfeld.
    ratio = logrank_required_events(0.7, allocation=2 / 3, exact=True) / logrank_required_events(
        0.7, exact=True
    )
    assert ratio == pytest.approx(9 / 8)


@pytest.mark.parametrize("method", ["schoenfeld", "freedman"])
def test_power_inverts_required_events(method):
    d = logrank_required_events(0.7, power=0.9, method=method, exact=True)
    assert logrank_power(d, 0.7, method=method) == pytest.approx(0.9, abs=2e-3)
    assert logrank_power(10, 0.7) < logrank_power(100, 0.7) < logrank_power(1000, 0.7)


def test_one_sided_needs_fewer_events():
    assert logrank_required_events(0.7, sides=1) < logrank_required_events(0.7)


def test_event_probability_closed_forms():
    lam = hazard_from_median(12.0)
    assert lam == pytest.approx(math.log(2) / 12)
    assert event_probability(lam, follow_up=12.0) == pytest.approx(0.5)
    # Numerical integration over uniform entry times.
    a, f, eta = 24.0, 12.0, 0.01
    grid = np.linspace(0, a, 200001)
    tot = lam + eta
    integrand = lam / tot * (1 - np.exp(-tot * (a + f - grid)))
    numeric = np.trapezoid(integrand, grid) / a if hasattr(np, "trapezoid") else np.trapz(integrand, grid) / a
    assert event_probability(lam, accrual=a, follow_up=f, dropout_hazard=eta) == pytest.approx(
        numeric, rel=1e-6
    )
    # Drop-out lowers the event probability.
    assert event_probability(lam, accrual=a, follow_up=f, dropout_hazard=0.05) < event_probability(
        lam, accrual=a, follow_up=f
    )


def test_sample_size_end_to_end():
    r = logrank_sample_size(0.7, control_median=12.0, accrual=24.0, follow_up=12.0)
    assert r.events == logrank_required_events(0.7)
    assert r.event_probability_treatment < r.event_probability_control
    assert r.n_control == r.n_treatment
    assert r.n_total * r.event_probability >= r.events
    assert (r.n_total - 2) * r.event_probability < r.events
    r2 = logrank_sample_size(0.7, control_hazard=hazard_from_median(12.0), accrual=24.0, follow_up=12.0)
    assert r2 == r


def test_simulated_power_matches_design():
    rng = np.random.default_rng(7)
    hr, lam0, a, f = 0.6, hazard_from_median(10.0), 12.0, 18.0
    design = logrank_sample_size(hr, control_hazard=lam0, accrual=a, follow_up=f, power=0.8)
    n = design.n_control
    rejections = 0
    reps = 400
    for _ in range(reps):
        entry0, entry1 = rng.uniform(0, a, n), rng.uniform(0, a, n)
        t0, t1 = rng.exponential(1 / lam0, n), rng.exponential(1 / (hr * lam0), n)
        c0, c1 = a + f - entry0, a + f - entry1
        res = log_rank_test(np.minimum(t0, c0), (t0 <= c0).astype(int), np.minimum(t1, c1), (t1 <= c1).astype(int))
        rejections += res.p_value < 0.05
    assert abs(rejections / reps - 0.8) < 0.07


def test_validation():
    with pytest.raises(ValueError):
        logrank_required_events(1.0)
    with pytest.raises(ValueError):
        logrank_required_events(0.5, allocation=1.0)
    with pytest.raises(ValueError):
        logrank_required_events(0.5, method="lakatos")
    with pytest.raises(ValueError):
        logrank_power(0, 0.5)
    with pytest.raises(ValueError):
        event_probability(0.1, follow_up=0.0)
    with pytest.raises(ValueError):
        logrank_sample_size(0.5, follow_up=12.0)
    with pytest.raises(ValueError):
        logrank_sample_size(0.5, control_median=1.0, control_hazard=1.0, follow_up=12.0)
