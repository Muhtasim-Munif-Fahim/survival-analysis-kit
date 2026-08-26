"""Tests for Harrell's concordance index."""

import numpy as np
import pytest

from survival_kit.concordance import concordance_index


def test_perfect_ranking_gives_one():
    result = concordance_index([1, 2, 3, 4, 5], [True] * 5, [5, 4, 3, 2, 1])
    assert result.c_index == 1.0
    assert result.comparable_pairs == 10
    assert result.concordant_pairs == 10
    assert result.tied_pairs == 0
    assert result.skipped_pairs == 0


def test_reversed_ranking_gives_zero():
    result = concordance_index([1, 2, 3, 4, 5], [True] * 5, [1, 2, 3, 4, 5])
    assert result.c_index == 0.0
    assert result.concordant_pairs == 0
    assert result.tied_pairs == 0


def test_constant_scores_earn_half_credit_on_every_pair():
    result = concordance_index([1, 2, 3], [True, True, True], [7.0, 7.0, 7.0])
    assert result.comparable_pairs == 3
    assert result.tied_pairs == 3
    assert result.c_index == 0.5


def test_hand_computed_mixed_case_with_censoring():
    result = concordance_index(
        [1, 2, 3, 4],
        [True, True, False, True],
        [0.9, 0.5, 0.8, 0.1],
    )
    assert result.comparable_pairs == 5
    assert result.concordant_pairs == 4
    assert result.tied_pairs == 0
    assert result.c_index == pytest.approx(0.8)


def test_tied_observed_times_are_never_comparable():
    result = concordance_index([2, 2, 3], [True, True, True], [0.1, 0.9, 0.5])
    assert result.skipped_pairs == 1
    assert result.comparable_pairs == 2
    assert result.c_index == 0.5


def test_censored_subject_can_serve_only_as_the_later_pair_member():
    result = concordance_index([5, 3, 4], [False, True, True], [9.0, 2.0, 1.0])
    assert result.comparable_pairs == 3
    assert result.concordant_pairs == 1
    assert result.skipped_pairs == 0
    assert result.c_index == pytest.approx(1.0 / 3.0)


def test_all_censored_sample_has_no_comparable_pairs():
    with pytest.raises(ValueError):
        concordance_index([1, 2], [False, False], [0.1, 0.2])


def test_noise_scores_hover_around_one_half():
    rng = np.random.default_rng(19)
    durations = rng.exponential(1.0, 250)
    result = concordance_index(durations, np.ones(250, bool), rng.normal(size=250))
    assert 0.42 < result.c_index < 0.58
    assert result.skipped_pairs == 0


def test_noisy_but_informative_scores_rank_almost_perfectly():
    rng = np.random.default_rng(29)
    durations = rng.exponential(1.0, 300)
    scores = -durations + rng.normal(scale=0.1, size=300)
    result = concordance_index(durations, np.ones(300, bool), scores)
    assert result.c_index > 0.90


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        concordance_index([1, 2], [True], [0.1, 0.2])
    with pytest.raises(ValueError):
        concordance_index([1, 2], [True, True], [0.1, np.nan])
    with pytest.raises(ValueError):
        concordance_index([-1, 2], [True, False], [0.1, 0.2])