"""Tests for D-calibration and 1-calibration of survival predictions."""

import json

import numpy as np
import pytest

from survival_kit import d_calibration, one_calibration, survival_at_observed
from survival_kit.cli import main


def _exp_data(seed, n=400, censor_scale=2.0, sigma=0.5):
    rng = np.random.default_rng(seed)
    lam = np.exp(rng.normal(scale=sigma, size=n))
    T = rng.exponential(1.0 / lam)
    C = rng.exponential(censor_scale, n)
    return lam, np.minimum(T, C), T <= C


# Reference values computed offline with SurvivalEVAL 0.9.0 on _exp_data(42, n=80):
#   d_calibration(S(T_i), events, 5)  and  one_calibration(1 - S(0.7), ..., 0.7, 4)
REF_D_STAT = 2.0940024456213764
REF_D_P = 0.7184743174599317
REF_D_HIST = [17.875426358209676, 15.02124139614239, 11.411586461969932, 17.084028275588626, 18.607717508089376]
REF_DN = (3.909800259099781, 0.27137024355501055, 3)
REF_DN_GREENWOOD = (4.019563687738021, 0.4033646671796231, 4)


class TestDCalibration:
    def test_matches_survivaleval_reference(self):
        lam, d, e = _exp_data(42, n=80)
        res = d_calibration(d, e, np.exp(-lam * d), n_bins=5)
        assert res.statistic == pytest.approx(REF_D_STAT, rel=1e-12)
        assert res.p_value == pytest.approx(REF_D_P, rel=1e-10)
        # SurvivalEVAL lists bins from S=1 down to 0; ours go up from 0
        assert np.allclose(res.counts, REF_D_HIST, rtol=1e-12)
        assert res.df == 4

    def test_hand_computed_censored_spreading(self):
        # events at S = 0.05 and 0.95; one censored subject with S(c) = 0.5
        res = d_calibration([1.0, 2.0, 3.0], [1, 1, 0], [0.05, 0.95, 0.5], n_bins=4)
        # censored mass: bin [0.5, 0.75) gets 0, bins [0, .25) and [.25, .5) get 0.25/0.5 each
        assert np.allclose(res.censored_mass, [0.5, 0.5, 0.0, 0.0])
        assert np.allclose(res.counts, [1.5, 0.5, 0.0, 1.0])
        expected = 3 / 4
        stat = sum((c - expected) ** 2 / expected for c in [1.5, 0.5, 0.0, 1.0])
        assert res.statistic == pytest.approx(stat)

    def test_edge_probabilities(self):
        res = d_calibration([1, 2, 3, 4], [1, 0, 0, 1], [1.0, 1.0, 0.0, 0.0], n_bins=2)
        # event at S=1 -> top bin; censored at S=1 -> spread evenly; censored at S=0 -> bottom
        assert np.allclose(res.counts, [1 + 0.5 + 1, 1 + 0.5])
        assert res.counts.sum() == pytest.approx(4)

    def test_mass_conserved(self):
        lam, d, e = _exp_data(3, n=500, censor_scale=0.7)
        res = d_calibration(d, e, np.exp(-lam * d), n_bins=10)
        assert res.counts.sum() == pytest.approx(500)
        assert res.proportions.sum() == pytest.approx(1.0)

    def test_true_model_is_calibrated_and_wrong_model_is_not(self):
        rejections = 0
        for seed in range(40):
            lam, d, e = _exp_data(seed)
            rejections += not d_calibration(d, e, np.exp(-lam * d)).calibrated
        assert rejections <= 6  # nominal 5% of 40 = 2
        lam, d, e = _exp_data(99)
        bad = d_calibration(d, e, np.exp(-3.0 * lam * d))  # hazards 3x too high
        assert bad.p_value < 1e-6 and bad.max_deviation > 0.05

    def test_validation(self):
        with pytest.raises(ValueError, match="n_bins"):
            d_calibration([1, 2], [1, 1], [0.2, 0.3], n_bins=1)
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            d_calibration([1, 2], [1, 1], [0.2, 1.3])
        with pytest.raises(ValueError, match="one value"):
            d_calibration([1, 2], [1, 1], [0.2])


class TestSurvivalAtObserved:
    def test_step_lookup(self):
        surv = np.array([[0.9, 0.5, 0.2], [0.8, 0.6, 0.4]])
        times = [1.0, 2.0, 3.0]
        out = survival_at_observed(surv, times, [0.5, 2.5])
        assert np.allclose(out, [1.0, 0.6])
        assert np.allclose(survival_at_observed(surv, times, [3.0, 10.0]), [0.2, 0.4])
        with pytest.raises(ValueError):
            survival_at_observed(surv, [1.0, 1.0, 2.0], [1.0, 2.0])


class TestOneCalibration:
    def test_matches_survivaleval_reference(self):
        lam, d, e = _exp_data(42, n=80)
        s_t = np.exp(-lam * 0.7)
        res = one_calibration(d, e, s_t, 0.7, n_groups=4)
        assert (res.statistic, res.p_value, res.df) == pytest.approx(REF_DN, rel=1e-10)
        gw = one_calibration(d, e, s_t, 0.7, n_groups=4, variance="greenwood")
        assert (gw.statistic, gw.p_value, gw.df) == pytest.approx(REF_DN_GREENWOOD, rel=1e-10)

    def test_uncensored_observed_is_empirical_rate(self):
        rng = np.random.default_rng(1)
        lam = np.exp(rng.normal(size=200))
        T = rng.exponential(1 / lam)
        res = one_calibration(T, np.ones(200), np.exp(-lam * 1.0), 1.0, n_groups=5)
        order = np.argsort(-(1 - np.exp(-lam)))
        for g, idx in enumerate(np.array_split(order, 5)):
            assert res.observed[g] == pytest.approx(np.mean(T[idx] <= 1.0))
        assert res.predicted[0] > res.predicted[-1]  # highest risk first
        assert len(res.table()) == 5 and res.table()[0]["n"] == 40

    def test_calibrated_vs_miscalibrated(self):
        lam, d, e = _exp_data(7, n=1500)
        good = one_calibration(d, e, np.exp(-lam * 1.0), 1.0)
        bad = one_calibration(d, e, np.exp(-0.4 * lam * 1.0), 1.0)
        assert good.p_value > 0.01
        assert bad.p_value < 1e-6 and not bad.calibrated
        gw = one_calibration(d, e, np.exp(-lam * 1.0), 1.0, variance="greenwood")
        assert gw.df == 10 and gw.p_value > 0.01

    def test_df_convention(self):
        lam, d, e = _exp_data(8, n=400)
        assert one_calibration(d, e, np.exp(-lam), 1.0, n_groups=15).df == 14
        assert one_calibration(d, e, np.exp(-lam), 1.0, n_groups=16).df == 14

    def test_validation(self):
        with pytest.raises(ValueError, match="horizon"):
            one_calibration([1, 2, 3], [1, 1, 1], [0.5, 0.5, 0.5], 0.0, n_groups=2)
        with pytest.raises(ValueError, match="variance"):
            one_calibration([1, 2, 3], [1, 1, 1], [0.5, 0.5, 0.5], 1.0, n_groups=2, variance="x")
        with pytest.raises(ValueError, match="n_groups"):
            one_calibration([1, 2], [1, 1], [0.5, 0.5], 1.0, n_groups=3)


def test_cli_calibration(tmp_path, capsys):
    lam, d, e = _exp_data(5, n=300)
    path = tmp_path / "preds.csv"
    with open(path, "w") as fh:
        fh.write("time,event,surv_obs,surv_1\n")
        for i in range(len(d)):
            fh.write(f"{d[i]},{int(e[i])},{np.exp(-lam[i] * d[i])},{np.exp(-lam[i])}\n")
    out = tmp_path / "cal.json"
    argv = ["calibration", "--data", str(path), "--surv-col", "surv_obs", "--horizon", "1.0",
            "--surv-at-horizon-col", "surv_1", "--out", str(out)]
    assert main(argv) == 0
    text = capsys.readouterr().out
    assert "D-calibration" in text and "1-calibration" in text
    res = json.loads(out.read_text())
    assert len(res["d_calibration"]["counts"]) == 10
    assert res["one_calibration"]["horizon"] == 1.0 and len(res["one_calibration"]["groups"]) == 10
    with pytest.raises(SystemExit):
        main(["calibration", "--data", str(path), "--surv-col", "nope", "--out", str(out)])
