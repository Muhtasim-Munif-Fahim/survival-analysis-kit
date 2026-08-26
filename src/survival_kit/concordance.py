"""Harrell's concordance index for right-censored risk predictions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .utils import validate_durations_events


@dataclass
class ConcordanceResult:
    """Pair accounting behind a concordance index estimate."""

    c_index: float
    comparable_pairs: int
    concordant_pairs: int
    tied_pairs: int
    skipped_pairs: int


def concordance_index(durations, events, risk_scores):
    """Estimate Harrell's C for risk scores under right censoring.

    ``risk_scores`` must be oriented so larger values predict earlier events.
    A usable pair needs distinct observed times where the shorter time is an
    event; pairs sharing an observed time are skipped because their true
    ordering is unidentifiable under censoring. Score ties earn half credit.
    """
    durations, events = validate_durations_events(durations, events)
    scores = np.asarray(risk_scores, dtype=float)
    if scores.ndim != 1 or scores.shape[0] != durations.shape[0]:
        raise ValueError("risk_scores must be one-dimensional and match the durations")
    if not np.all(np.isfinite(scores)):
        raise ValueError("risk_scores must contain only finite values")

    n = durations.size
    comparable = 0
    concordant = 0
    tied = 0
    skipped = 0
    for i in range(n):
        t_i = durations[i]
        e_i = events[i]
        s_i = scores[i]
        for j in range(i + 1, n):
            t_j = durations[j]
            if t_i == t_j:
                skipped += 1
                continue
            earlier, later = (i, j) if t_i < t_j else (j, i)
            if not events[earlier]:
                skipped += 1
                continue
            comparable += 1
            if scores[earlier] > scores[later]:
                concordant += 1
            elif scores[earlier] == scores[later]:
                tied += 1

    if comparable == 0:
        raise ValueError("no comparable pairs: c-index is undefined")
    return ConcordanceResult(
        c_index=(concordant + 0.5 * tied) / comparable,
        comparable_pairs=comparable,
        concordant_pairs=concordant,
        tied_pairs=tied,
        skipped_pairs=skipped,
    )