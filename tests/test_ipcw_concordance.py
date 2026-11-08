import numpy as np
import pytest

from survival_kit import concordance_index, gonen_heller_cpe, uno_concordance_index


def _ph_data(n, seed, censor_rate=None, nonlinear=False):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    eta = x if not nonlinear else np.abs(x) * 1.5 - 0.5 * x
    t = rng.exponential(np.exp(-eta))
    if censor_rate is None:
        return x, eta, t, t.copy(), np.ones(n, dtype=bool)
    c = rng.exponential(1.0 / censor_rate, n)
    return x, eta, t, np.minimum(t, c), t <= c


def _oracle_truncated_c(t, score, tau):
    i, j = np.where((t[:, None] < t[None, :]) & (t[:, None] < tau))
    diff = score[i] - score[j]
    return float(np.mean((diff > 0) + 0.5 * (diff == 0)))


def test_equals_harrell_without_censoring():
    _, eta, _, d, e = _ph_data(300, 0)
    score = np.round(eta, 1)  # create score ties
    uno = uno_concordance_index(d, e, score, tau=d.max() + 1)
    harrell = concordance_index(d, e, score)
    assert uno.c_index == pytest.approx(harrell.c_index)
    assert uno.comparable_pairs == harrell.comparable_pairs


def test_hand_computed_weights():
    d = np.array([1.0, 2.0, 3.0, 4.0])
    e = np.array([True, False, True, True])
    s = np.array([4.0, 3.0, 1.0, 2.0])
    # G(t-) from KM of censoring: G(1-)=1, G(3-)=1-1/3=2/3 (censored at 2 with 3 at risk)
    # pairs: i=0 (w=1): j=1,2,3 all concordant -> 3 ; i=2 (w=9/4): j=3, s2<s3 discordant
    res = uno_concordance_index(d, e, s, tau=10.0)
    assert res.c_index == pytest.approx(3.0 / (3.0 + 2.25))
    assert res.comparable_pairs == 4


@pytest.mark.parametrize("nonlinear", [False, True])
def test_uno_recovers_censoring_free_estimand(nonlinear):
    x, _, t, d, e = _ph_data(2500, 1, censor_rate=1.0, nonlinear=nonlinear)
    score = x  # deliberately misspecified when nonlinear
    tau = float(np.quantile(d, 0.8))
    oracle = _oracle_truncated_c(t, score, tau)
    uno = uno_concordance_index(d, e, score, tau=tau)
    assert uno.c_index == pytest.approx(oracle, abs=0.015)


def test_external_censoring_sample_and_bootstrap():
    _, eta, _, d, e = _ph_data(300, 2, censor_rate=0.7)
    _, _, _, d_tr, e_tr = _ph_data(600, 3, censor_rate=0.7)
    tau = float(np.quantile(d, 0.75))
    res = uno_concordance_index(d, e, eta, tau=tau, train_durations=d_tr, train_events=e_tr,
                                n_bootstrap=50, seed=0)
    assert 0.6 < res.c_index < 0.85
    assert res.std_err is not None and 0 < res.std_err < 0.1
    assert res.ci_lower < res.c_index < res.ci_upper


def test_gonen_heller_matches_uncensored_harrell():
    _, eta, _, d, e = _ph_data(1500, 4)
    cpe = gonen_heller_cpe(eta)
    assert cpe == pytest.approx(concordance_index(d, e, eta).c_index, abs=0.015)
    # chunking does not change the result
    assert gonen_heller_cpe(eta, chunk_size=7) == pytest.approx(cpe)
    assert gonen_heller_cpe(np.zeros(5)) == pytest.approx(0.5)
    assert gonen_heller_cpe(-eta) == pytest.approx(cpe)  # orientation-free by construction


def test_validation():
    with pytest.raises(ValueError):
        uno_concordance_index([1, 2], [0, 0], [1, 2])
    with pytest.raises(ValueError):
        uno_concordance_index([1, 2], [1, 1], [1.0])
    with pytest.raises(ValueError):
        uno_concordance_index([1, 2], [1, 1], [1, 2], train_durations=[1, 2])
    with pytest.raises(ValueError):
        gonen_heller_cpe([1.0])
