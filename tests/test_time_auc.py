"""Tests for cumulative/dynamic time-dependent AUC."""

from __future__ import annotations

import numpy as np
import pytest

from survival_kit import TimeDependentAUCResult, cumulative_dynamic_auc, fit_kaplan_meier


def _brute_auc(durations, events, est, t, g_curve):
    num = den = 0.0
    n_ctrl = int(np.sum(durations > t))
    for i in range(len(durations)):
        if not (events[i] and durations[i] <= t):
            continue
        w = 1.0
        if g_curve is not None:
            w = 1.0 / g_curve.at(np.nextafter(durations[i], -np.inf))
        den += w * n_ctrl
        for j in range(len(durations)):
            if durations[j] > t:
                if est[i] > est[j]:
                    num += w
                elif est[i] == est[j]:
                    num += 0.5 * w
    return num / den


def _weibull_ph(n, beta, censor_rate, seed):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, len(beta)))
    eta = x @ np.asarray(beta)
    latent = (rng.exponential(size=n) / np.exp(eta)) ** (1 / 1.5)
    cens = rng.exponential(1 / censor_rate, size=n) if censor_rate > 0 else np.full(n, np.inf)
    durations = np.minimum(latent, cens)
    events = latent <= cens
    return durations, events, eta, latent


def test_no_censoring_equals_mann_whitney():
    rng = np.random.default_rng(0)
    d = rng.exponential(size=60)
    e = np.ones(60, dtype=bool)
    est = -d + rng.normal(scale=0.5, size=60)
    times = np.quantile(d, [0.25, 0.5, 0.75])
    res = cumulative_dynamic_auc(d, e, est, times)
    for k, t in enumerate(times):
        cases, ctrls = est[d <= t], est[d > t]
        mw = np.mean(cases[:, None] > ctrls[None, :]) + 0.5 * np.mean(cases[:, None] == ctrls[None, :])
        assert res.auc[k] == pytest.approx(mw)
    assert isinstance(res, TimeDependentAUCResult)
    assert res.ipcw


def test_matches_bruteforce_ipcw_with_ties():
    rng = np.random.default_rng(1)
    d = np.round(rng.exponential(size=80), 1)
    e = rng.random(80) < 0.6
    est = np.round(rng.normal(size=80) - 0.5 * d, 1)  # ties on purpose
    times = np.array([0.3, 0.7, 1.2])
    res = cumulative_dynamic_auc(d, e, est, times)
    g = fit_kaplan_meier(d, ~e)
    for k, t in enumerate(times):
        assert res.auc[k] == pytest.approx(_brute_auc(d, e, est, t, g))
        assert res.n_cases[k] == int(np.sum(e & (d <= t)))
        assert res.n_controls[k] == int(np.sum(d > t))


def test_hand_computed_small_example():
    d = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    e = np.array([True, False, True, True, False])
    est = np.array([0.9, 0.1, 0.4, 0.6, 0.2])
    res = cumulative_dynamic_auc(d, e, est, [3.5])
    # censoring KM: G drops only at t=2 (1 of 4 at risk) and t=5 -> G(1-)=1, G(3-)=0.75
    # cases by t=3.5: subj0 (w=1/G(1-)=1) and subj2 (w=1/G(3-)=1/0.75)
    # controls at t=3.5: subj3 (0.6), subj4 (0.2)
    w0, w2 = 1.0, 1 / 0.75
    num = w0 * 2 + w2 * 1  # 0.9 beats both; 0.4 beats only 0.2
    expected = num / ((w0 + w2) * 2)
    assert res.auc[0] == pytest.approx(expected)
    naive = cumulative_dynamic_auc(d, e, est, [3.5], ipcw=False)
    assert naive.auc[0] == pytest.approx((2 + 1) / (2 * 2))
    assert not naive.ipcw


def test_perfect_reverse_and_constant_scores():
    d, e, eta, _ = _weibull_ph(300, [1.0], 0.3, seed=2)
    times = np.quantile(d[e], [0.2, 0.5, 0.8])
    perfect = cumulative_dynamic_auc(d, e, -d, times)
    assert np.allclose(perfect.auc, 1.0)
    reverse = cumulative_dynamic_auc(d, e, d, times)
    assert np.allclose(reverse.auc, 0.0)
    const = cumulative_dynamic_auc(d, e, np.zeros_like(d), times)
    assert np.allclose(const.auc, 0.5)
    assert perfect.mean_auc == pytest.approx(1.0)


def test_ipcw_is_close_to_uncensored_oracle_under_heavy_censoring():
    d, e, eta, latent = _weibull_ph(4000, [1.2, -0.6], censor_rate=1.0, seed=3)
    assert 0.3 < 1 - e.mean() < 0.7
    times = np.quantile(d[e], [0.3, 0.6])
    res = cumulative_dynamic_auc(d, e, eta, times)
    oracle = cumulative_dynamic_auc(latent, np.ones_like(e), eta, times)
    assert np.max(np.abs(res.auc - oracle.auc)) < 0.02
    assert np.all(res.auc > 0.7)


def test_time_dependent_estimate_and_train_censoring():
    d, e, eta, _ = _weibull_ph(400, [1.0], 0.5, seed=4)
    times = np.quantile(d[e], [0.25, 0.5, 0.75])
    est2d = np.repeat(eta[:, None], 3, axis=1)
    a = cumulative_dynamic_auc(d, e, eta, times)
    b = cumulative_dynamic_auc(d, e, est2d, times)
    assert np.allclose(a.auc, b.auc)
    tr_d, tr_e, _, _ = _weibull_ph(600, [1.0], 0.5, seed=5)
    c = cumulative_dynamic_auc(d, e, eta, times, train_durations=tr_d, train_events=tr_e)
    assert np.allclose(c.auc, a.auc, atol=0.03)
    assert not np.allclose(c.auc, a.auc, atol=0.0)


def test_mean_auc_uses_km_drops():
    d, e, eta, _ = _weibull_ph(300, [1.0], 0.4, seed=6)
    times = np.quantile(d[e], [0.2, 0.4, 0.6, 0.8])
    res = cumulative_dynamic_auc(d, e, eta, times)
    km = fit_kaplan_meier(d, e).at(times)
    drop = -np.diff(np.r_[1.0, km])
    assert res.mean_auc == pytest.approx(np.dot(res.auc, drop) / (1 - km[-1]))
    single = cumulative_dynamic_auc(d, e, eta, times[:1])
    assert single.mean_auc == pytest.approx(single.auc[0])


def test_validation_errors():
    d = np.array([1.0, 2.0, 3.0])
    e = np.array([True, True, False])
    est = np.array([0.3, 0.2, 0.1])
    with pytest.raises(ValueError):
        cumulative_dynamic_auc(d, e, est, [])
    with pytest.raises(ValueError):
        cumulative_dynamic_auc(d, e, est, [2.0, 1.5])
    with pytest.raises(ValueError):
        cumulative_dynamic_auc(d, e, est[:2], [1.5])
    with pytest.raises(ValueError):
        cumulative_dynamic_auc(d, e, np.zeros((3, 2)), [1.5])
    with pytest.raises(ValueError, match="cases"):
        cumulative_dynamic_auc(d, e, est, [0.5])
    with pytest.raises(ValueError, match="controls"):
        cumulative_dynamic_auc(d, e, est, [3.0])
    with pytest.raises(ValueError):
        cumulative_dynamic_auc(d, e, est, [1.5], train_durations=d)
    with pytest.raises(ValueError):
        cumulative_dynamic_auc(d, e, np.array([0.1, np.nan, 0.2]), [1.5])
