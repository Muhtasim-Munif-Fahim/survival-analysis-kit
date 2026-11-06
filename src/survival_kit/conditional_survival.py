"""Conditional survival and mean residual life from a Kaplan-Meier curve.

Clinical questions are often conditional: given survival to a landmark
``t0``, what is the chance of lasting another ``s`` years, and what is the
remaining life expectancy? Both follow from the Kaplan-Meier curve without
refitting:

* conditional survival ``P(T > t0 + s | T > t0) = S(t0 + s) / S(t0)``
* mean residual life ``m(t0) = (1 / S(t0)) * integral_{t0}^{tau} S(u) du``
  (restricted to ``tau`` when the curve does not reach zero)

Greenwood variance is delta-method propagated for the conditional survival
point estimate. MRL uses the restricted-mean identity already available on
:class:`~survival_kit.kaplan_meier.SurvivalCurve`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Union

import numpy as np
from scipy.stats import norm

from .kaplan_meier import SurvivalCurve, fit_kaplan_meier
from .utils import step_value

__all__ = [
    "ConditionalSurvivalResult",
    "MeanResidualLifeResult",
    "conditional_survival",
    "mean_residual_life",
]


Number = Union[float, Sequence[float], np.ndarray]


@dataclass
class ConditionalSurvivalResult:
    """Conditional survival ``P(T > landmark + horizon | T > landmark)``."""

    landmark: float
    horizon: np.ndarray
    survival: np.ndarray
    se: np.ndarray
    ci_lower: np.ndarray
    ci_upper: np.ndarray
    s_landmark: float
    confidence_level: float

    def at(self, horizons: Number) -> np.ndarray:
        h = np.atleast_1d(np.asarray(horizons, dtype=float))
        return np.interp(h, self.horizon, self.survival, left=np.nan, right=np.nan)


@dataclass
class MeanResidualLifeResult:
    """Restricted mean residual life at one or more landmarks."""

    landmark: np.ndarray
    mrl: np.ndarray
    tau: float
    s_landmark: np.ndarray


def _as_curve(durations=None, events=None, curve: Optional[SurvivalCurve] = None) -> SurvivalCurve:
    if curve is not None:
        return curve
    if durations is None or events is None:
        raise ValueError("pass either a SurvivalCurve or durations+events")
    return fit_kaplan_meier(durations, events)


def conditional_survival(
    landmark: float,
    horizons: Number,
    *,
    durations=None,
    events=None,
    curve: Optional[SurvivalCurve] = None,
    confidence_level: float = 0.95,
) -> ConditionalSurvivalResult:
    """Estimate ``S(landmark + h) / S(landmark)`` for each horizon ``h``.

    Standard errors use the delta method on Greenwood variances of the two
    Kaplan-Meier points (assuming the usual martingale independence of the
    increments). Horizons must be non-negative.
    """
    if landmark < 0:
        raise ValueError("landmark must be non-negative")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie strictly between 0 and 1")
    km = _as_curve(durations, events, curve)
    h = np.atleast_1d(np.asarray(horizons, dtype=float)).astype(float)
    if np.any(h < 0):
        raise ValueError("horizons must be non-negative")

    s0 = float(km.at(landmark))
    if s0 <= 0 or not np.isfinite(s0):
        raise ValueError("S(landmark) is zero or undefined; cannot condition")

    targets = landmark + h
    s1 = np.asarray(km.at(targets), dtype=float)
    cond = s1 / s0

    # Approximate SE via Greenwood SE on the KM curve (same step rule as S(t)).
    se0 = float(np.asarray(step_value(km.time, km.std_err, landmark, outside=0.0), dtype=float))
    se1 = np.asarray(step_value(km.time, km.std_err, targets, outside=0.0), dtype=float)
    # delta method for ratio: Var(s1/s0) ≈ (s1/s0)^2 * ((se1/s1)^2 + (se0/s0)^2)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.zeros_like(cond)
        ok = (s1 > 0) & np.isfinite(s1) & np.isfinite(se1)
        rel[ok] = (se1[ok] / s1[ok]) ** 2 + (se0 / s0) ** 2
        se = np.abs(cond) * np.sqrt(rel)
        se[~ok] = np.nan

    z = float(norm.ppf(0.5 + confidence_level / 2.0))
    lo = np.clip(cond - z * se, 0.0, 1.0)
    hi = np.clip(cond + z * se, 0.0, 1.0)
    return ConditionalSurvivalResult(
        landmark=float(landmark),
        horizon=h,
        survival=cond,
        se=se,
        ci_lower=lo,
        ci_upper=hi,
        s_landmark=s0,
        confidence_level=float(confidence_level),
    )


def mean_residual_life(
    landmarks: Number,
    *,
    durations=None,
    events=None,
    curve: Optional[SurvivalCurve] = None,
    tau: Optional[float] = None,
) -> MeanResidualLifeResult:
    """Restricted mean residual life ``m(t) = RMST(tau) residual past t``.

    With Kaplan-Meier ``S``,
    ``m(t) = (integral_t^tau S(u) du) / S(t)`` when ``S(t) > 0``.
    ``tau`` defaults to the last observed time on the curve.
    """
    km = _as_curve(durations, events, curve)
    t0 = np.atleast_1d(np.asarray(landmarks, dtype=float)).astype(float)
    if np.any(t0 < 0):
        raise ValueError("landmarks must be non-negative")
    if tau is None:
        tau = float(km.time[-1])
    tau = float(tau)
    if tau <= 0:
        raise ValueError("tau must be positive")

    # RMST from 0 to tau and from 0 to each landmark via the curve helper.
    rmst_tau = float(km.restricted_mean(tau))
    mrl = np.empty(t0.shape, dtype=float)
    s_land = np.empty(t0.shape, dtype=float)
    for i, t in enumerate(t0):
        if t >= tau:
            mrl[i] = 0.0
            s_land[i] = float(km.at(t))
            continue
        s = float(km.at(t))
        s_land[i] = s
        if s <= 0 or not np.isfinite(s):
            mrl[i] = np.nan
            continue
        rmst_t = float(km.restricted_mean(t))
        mrl[i] = (rmst_tau - rmst_t) / s
    return MeanResidualLifeResult(landmark=t0, mrl=mrl, tau=tau, s_landmark=s_land)
