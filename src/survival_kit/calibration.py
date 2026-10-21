"""Distribution and single-horizon calibration of survival predictions.

* **D-calibration** (Haider, Hoehn, Davis & Greiner, JMLR 2020) asks whether
  the predicted survival curves are calibrated *as distributions*. If the
  model is right, the predicted survival probability evaluated at each
  subject's true event time, ``S_i(T_i)``, is Uniform(0, 1). The ``[0, 1]``
  interval is split into ``B`` equal bins and the bin counts are compared
  with ``n / B`` by Pearson's chi-square (``B - 1`` df). A subject censored
  at ``c`` with ``s = S_i(c)`` only tells us ``S_i(T_i) <= s``; under the
  null that value is Uniform(0, s), so its unit of mass is spread
  over ``[0, s]``: the bin containing ``s`` gets ``(s - lower) / s`` and
  each bin entirely below ``s`` gets ``(1 / B) / s``.

* **1-calibration** at a horizon ``t`` (Hosmer-Lemeshow for survival,
  D'Agostino & Nam 2003): subjects are sorted by predicted risk at ``t`` and
  split into ``G`` groups. Within each group the observed event probability
  ``1 - KM_g(t)`` (Kaplan-Meier, so censoring is handled) is compared with
  the mean predicted event probability ``pbar_g``:
  ``DN = sum_g (obs_g - pbar_g)^2 / var_g`` with binomial
  (``G - 1`` df) or Greenwood (``G`` df, Demler et al. 2015) variance.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import chi2

from .kaplan_meier import fit_kaplan_meier
from .utils import step_value, validate_durations_events


@dataclass
class DCalibrationResult:
    """Histogram of ``S_i(T_i)`` against Uniform(0, 1)."""

    bin_edges: np.ndarray
    counts: np.ndarray
    proportions: np.ndarray
    statistic: float
    df: int
    p_value: float
    n_observations: int
    n_events: int
    censored_mass: np.ndarray

    @property
    def calibrated(self) -> bool:
        """``True`` when uniformity is not rejected at the 5% level."""
        return bool(self.p_value >= 0.05)

    @property
    def max_deviation(self) -> float:
        """Largest absolute gap between a bin proportion and ``1 / B``."""
        return float(np.max(np.abs(self.proportions - 1.0 / self.counts.size)))


@dataclass
class OneCalibrationResult:
    """Grouped predicted vs Kaplan-Meier observed event probability at ``t``."""

    horizon: float
    n_groups: int
    group_sizes: np.ndarray
    predicted: np.ndarray
    observed: np.ndarray
    statistic: float
    df: int
    p_value: float
    n_observations: int
    variance: np.ndarray

    @property
    def calibrated(self) -> bool:
        return bool(self.p_value >= 0.05)

    def table(self) -> list[dict[str, float]]:
        """One row per risk group (highest predicted risk first)."""
        return [
            {
                "group": g + 1,
                "n": int(self.group_sizes[g]),
                "predicted_event_prob": float(self.predicted[g]),
                "observed_event_prob": float(self.observed[g]),
            }
            for g in range(self.n_groups)
        ]


def survival_at_observed(surv_probs, times, durations):
    """Evaluate each subject's predicted curve at its own observed time.

    ``surv_probs`` has shape ``(n_samples, n_times)`` with ``S_i(times[j])``;
    the curve is treated as a right-continuous step function that equals 1
    before ``times[0]``. Returns ``S_i(durations[i])``.
    """
    surv = np.asarray(surv_probs, dtype=float)
    times = np.asarray(times, dtype=float).ravel()
    durations = np.asarray(durations, dtype=float).ravel()
    if surv.ndim != 2 or surv.shape != (durations.size, times.size):
        raise ValueError("surv_probs must have shape (n_samples, n_times)")
    if times.size == 0 or np.any(np.diff(times) <= 0):
        raise ValueError("times must be strictly increasing")
    idx = np.searchsorted(times, durations, side="right") - 1
    out = np.ones(durations.size)
    ok = idx >= 0
    out[ok] = surv[np.nonzero(ok)[0], idx[ok]]
    return out


def _check_probs(values, n, name):
    values = np.asarray(values, dtype=float).ravel()
    if values.size != n:
        raise ValueError(f"{name} must have one value per observation")
    if not np.all(np.isfinite(values)) or np.any(values < -1e-12) or np.any(values > 1 + 1e-12):
        raise ValueError(f"{name} must lie in [0, 1]")
    return np.clip(values, 0.0, 1.0)


def d_calibration(durations, events, surv_at_observed_time, n_bins: int = 10) -> DCalibrationResult:
    """D-calibration test (Haider et al., 2020) for right-censored data.

    Parameters
    ----------
    durations, events:
        Observed times and event indicators.
    surv_at_observed_time:
        Predicted ``S_i(durations[i])`` per subject; see
        :func:`survival_at_observed` to obtain it from a survival matrix.
    n_bins:
        Number of equal-width bins on ``[0, 1]`` (default 10).
    """
    durations, events = validate_durations_events(durations, events)
    if not isinstance(n_bins, (int, np.integer)) or isinstance(n_bins, bool) or n_bins < 2:
        raise ValueError("n_bins must be an integer >= 2")
    n = durations.size
    s = _check_probs(surv_at_observed_time, n, "surv_at_observed_time")
    B = int(n_bins)
    edges = np.linspace(0.0, 1.0, B + 1)
    width = 1.0 / B
    # bin index: [k/B, (k+1)/B), with s == 1 in the top bin
    idx = np.minimum((s * B).astype(int), B - 1)

    counts = np.zeros(B)
    np.add.at(counts, idx[events], 1.0)

    censored_mass = np.zeros(B)
    cs = s[~events]
    cidx = idx[~events]
    zero = cs <= 0.0
    # S(c) = 0 means the event must already have happened: all mass at 0
    censored_mass[0] += float(zero.sum())
    cs, cidx = cs[~zero], cidx[~zero]
    if cs.size:
        np.add.at(censored_mass, cidx, (cs - edges[cidx]) / cs)
        below = np.arange(B)[None, :] < cidx[:, None]
        censored_mass += (below * (width / cs[:, None])).sum(axis=0)
    counts = counts + censored_mass

    expected = n / B
    statistic = float(np.sum((counts - expected) ** 2) / expected)
    df = B - 1
    return DCalibrationResult(
        bin_edges=edges,
        counts=counts,
        proportions=counts / n,
        statistic=statistic,
        df=df,
        p_value=float(chi2.sf(statistic, df)),
        n_observations=n,
        n_events=int(events.sum()),
        censored_mass=censored_mass,
    )


def one_calibration(
    durations,
    events,
    surv_at_horizon,
    horizon: float,
    n_groups: int = 10,
    variance: str = "binomial",
) -> OneCalibrationResult:
    """D'Agostino-Nam (Hosmer-Lemeshow-style) 1-calibration at horizon ``t``.

    ``surv_at_horizon`` is the predicted ``S_i(t)`` per subject. Subjects are
    sorted by predicted event probability ``1 - S_i(t)`` (highest risk first)
    and split into ``n_groups`` near-equal groups; the observed event
    probability per group is ``1 - KM_g(t)``.

    ``variance``:

    * ``"binomial"`` (original D'Agostino & Nam, 2003) - group variance
      ``pbar_g (1 - pbar_g) / n_g``; ``G - 1`` df (``G - 2`` when ``G > 15``).
    * ``"greenwood"`` (Demler, Paynter & Cook, 2015) - Greenwood variance of
      ``KM_g(t)``; ``G`` df. Better behaved under heavy censoring.

    A group whose variance is 0 contributes 0 if observed == predicted and
    ``inf`` otherwise.
    """
    durations, events = validate_durations_events(durations, events)
    horizon = float(horizon)
    if not np.isfinite(horizon) or horizon <= 0:
        raise ValueError("horizon must be a positive finite number")
    if not isinstance(n_groups, (int, np.integer)) or isinstance(n_groups, bool) or n_groups < 2:
        raise ValueError("n_groups must be an integer >= 2")
    if variance not in ("binomial", "greenwood"):
        raise ValueError("variance must be 'binomial' or 'greenwood'")
    n = durations.size
    if n < n_groups:
        raise ValueError("need at least n_groups observations")
    s = _check_probs(surv_at_horizon, n, "surv_at_horizon")
    risk = 1.0 - s
    groups = np.array_split(np.argsort(-risk), int(n_groups))
    sizes = np.array([g.size for g in groups])
    predicted = np.array([risk[g].mean() for g in groups])
    observed = np.empty(len(groups))
    var = np.empty(len(groups))
    for k, g in enumerate(groups):
        curve = fit_kaplan_meier(durations[g], events[g])
        observed[k] = 1.0 - float(curve.at(horizon))
        if variance == "greenwood":
            se = step_value(curve.time, curve.std_err, horizon, outside=0.0)
            var[k] = float(se) ** 2
        else:
            var[k] = predicted[k] * (1.0 - predicted[k]) / g.size
    num = (observed - predicted) ** 2
    contrib = np.divide(num, var, out=np.where(num == 0, 0.0, np.inf), where=var != 0)
    statistic = float(contrib.sum())
    G = len(groups)
    df = G if variance == "greenwood" else G - (2 if G > 15 else 1)
    return OneCalibrationResult(
        horizon=horizon,
        n_groups=G,
        group_sizes=sizes,
        predicted=predicted,
        observed=observed,
        statistic=statistic,
        df=df,
        p_value=float(chi2.sf(statistic, df)),
        n_observations=n,
        variance=var,
    )
