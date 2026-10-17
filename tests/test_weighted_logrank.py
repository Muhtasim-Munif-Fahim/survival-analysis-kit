"""Tests for weighted (Fleming-Harrington, Gehan, Tarone-Ware, Peto) log-rank tests."""

import numpy as np
import pytest

from survival_kit import weighted_log_rank_test_groups
from survival_kit.cli import main
from survival_kit.logrank import log_rank_test_groups


def _three_groups():
    rng = np.random.default_rng(0)
    n = 150
    g = rng.integers(0, 3, n)
    t = np.round(rng.exponential(1 + 0.4 * g), 1) + 0.1
    c = rng.exponential(3, n)
    e = t <= c
    d = np.minimum(t, c).round(1) + 0.1
    return d, e, g


# Reference statistics from lifelines 0.30 ``multivariate_logrank_test`` on the
# same data (ties included), computed when this module was written.
REFERENCE = [
    ("log-rank", 0.0, 0.0, 3.406495),
    ("wilcoxon", 0.0, 0.0, 5.381109),
    ("tarone-ware", 0.0, 0.0, 5.250668),
    ("peto-peto", 0.0, 0.0, 5.361818),
    ("fleming-harrington", 1.0, 0.0, 5.457074),
    ("fleming-harrington", 0.0, 1.0, 0.314375),
    ("fleming-harrington", 1.0, 1.0, 2.837574),
]


@pytest.mark.parametrize("weighting, rho, gamma, expected", REFERENCE)
def test_matches_reference_implementation(weighting, rho, gamma, expected):
    d, e, g = _three_groups()
    res = weighted_log_rank_test_groups(d, e, g, weighting=weighting, rho=rho, gamma=gamma)
    assert res.statistic == pytest.approx(expected, abs=1e-5)
    assert res.degrees_of_freedom == 2
    assert res.weighting == weighting


def test_fh_zero_zero_equals_log_rank():
    d, e, g = _three_groups()
    fh = weighted_log_rank_test_groups(d, e, g, weighting="fleming-harrington")
    lr = log_rank_test_groups(d, e, g)
    assert fh.statistic == pytest.approx(lr.statistic)
    assert fh.p_value == pytest.approx(lr.p_value)
    assert np.allclose(fh.observed_deaths, lr.observed_deaths)
    assert np.allclose(fh.weighted_score, fh.observed_deaths - fh.expected_deaths)


def test_aliases_and_validation():
    d, e, g = _three_groups()
    a = weighted_log_rank_test_groups(d, e, g, weighting="gehan")
    b = weighted_log_rank_test_groups(d, e, g, weighting="wilcoxon")
    assert a.statistic == b.statistic
    with pytest.raises(ValueError, match="weighting"):
        weighted_log_rank_test_groups(d, e, g, weighting="nope")
    with pytest.raises(ValueError, match="non-negative"):
        weighted_log_rank_test_groups(d, e, g, rho=-1)
    with pytest.raises(ValueError, match="two distinct"):
        weighted_log_rank_test_groups(d, e, np.zeros_like(g))
    with pytest.raises(ValueError, match="strata"):
        weighted_log_rank_test_groups(d, e, g, strata=[0, 1])


def test_late_weights_detect_delayed_separation():
    # curves identical early, separate only after t = 2 (delayed treatment effect)
    rng = np.random.default_rng(3)
    n = 400
    g = np.repeat([0, 1], n // 2)
    t = rng.exponential(2.0, n)
    late = (g == 1) & (t > 2.0)
    t[late] = 2.0 + rng.exponential(6.0, late.sum())
    e = np.ones(n, dtype=bool)
    early = weighted_log_rank_test_groups(t, e, g, rho=1, gamma=0)
    late_w = weighted_log_rank_test_groups(t, e, g, rho=0, gamma=1)
    assert late_w.statistic > early.statistic
    assert late_w.p_value < 0.001


def test_single_stratum_matches_unstratified_and_strata_sum():
    d, e, g = _three_groups()
    plain = weighted_log_rank_test_groups(d, e, g, weighting="tarone-ware")
    one = weighted_log_rank_test_groups(d, e, g, weighting="tarone-ware", strata=np.ones_like(g))
    assert one.statistic == pytest.approx(plain.statistic)
    strata = (np.arange(d.size) % 2).astype(int)
    strat = weighted_log_rank_test_groups(d, e, g, weighting="log-rank", strata=strata)
    parts = [
        weighted_log_rank_test_groups(d[strata == s], e[strata == s], g[strata == s], weighting="log-rank")
        for s in (0, 1)
    ]
    assert np.allclose(strat.weighted_score, parts[0].weighted_score + parts[1].weighted_score)
    assert strat.n_strata == 2


def test_stratification_removes_confounded_difference():
    # group membership tracks a stratum with very different baseline hazards
    rng = np.random.default_rng(7)
    n = 600
    stratum = rng.integers(0, 2, n)
    g = np.where(rng.random(n) < np.where(stratum == 1, 0.8, 0.2), 1, 0)
    t = rng.exponential(np.where(stratum == 1, 0.5, 3.0))
    e = np.ones(n, dtype=bool)
    naive = weighted_log_rank_test_groups(t, e, g, weighting="log-rank")
    adjusted = weighted_log_rank_test_groups(t, e, g, weighting="log-rank", strata=stratum)
    assert naive.p_value < 1e-6
    assert adjusted.p_value > 0.01


def test_cli_weighted_and_stratified(tmp_path, capsys):
    path = tmp_path / "surv.csv"
    d, e, g = _three_groups()
    strata = np.arange(d.size) % 2  # numeric codes (extra CSV columns must be numeric)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("duration,event,group,site\n")
        for row in zip(d, e.astype(int), g, strata):
            fh.write(",".join(str(v) for v in row) + "\n")
    main(["compare", "--data", str(path), "--group-col", "group", "--weighting", "wilcoxon"])
    out = capsys.readouterr().out
    assert "test: wilcoxon" in out and "chi-square(2) = 5.381" in out
    main(["compare", "--data", str(path), "--group-col", "group",
          "--weighting", "fleming-harrington", "--rho", "1", "--strata-col", "site"])
    out = capsys.readouterr().out
    assert "stratified by site (2 strata)" in out
    with pytest.raises(SystemExit):
        main(["compare", "--data", str(path), "--group-col", "group", "--strata-col", "missing"])
