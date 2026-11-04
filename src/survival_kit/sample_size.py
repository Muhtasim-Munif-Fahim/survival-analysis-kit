"""Sample size and power for the two-arm log-rank test.

Design-stage counterpart to :mod:`survival_kit.logrank`. Log-rank power is
driven by the **number of events**, not the number of subjects, so the
calculation has two steps:

1. Required events ``D`` for hazard ratio ``HR`` at level ``alpha`` with power
   ``1 - beta`` and allocation fraction ``p`` (share randomised to the
   treatment arm):

   * Schoenfeld (1981): ``D = (z_{1-a/2} + z_{1-b})^2 / (p (1-p) log(HR)^2)``
   * Freedman (1982):   ``D = (z_{1-a/2} + z_{1-b})^2 (1 + k HR)^2 / (k (1 - HR)^2)``
     with allocation ratio ``k = p / (1 - p)``.

2. Probability that a subject's event is observed by the end of the study,
   assuming exponential event times (hazard ``lam``), exponential drop-out
   (hazard ``eta``), uniform accrual over ``[0, accrual]`` and an extra
   ``follow_up`` after accrual closes:

   ``P = lam/(lam+eta) * (1 - (exp(-(lam+eta) f) - exp(-(lam+eta)(a+f))) / ((lam+eta) a))``

   (``1 - exp(-(lam+eta) f)`` when ``accrual == 0``). The total sample size is
   ``N = D / (p P_trt + (1-p) P_ctl)``.

``alpha`` is two-sided throughout (pass ``sides=1`` for one-sided).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from scipy.stats import norm

__all__ = [
    "LogRankSampleSize",
    "hazard_from_median",
    "logrank_required_events",
    "logrank_power",
    "event_probability",
    "logrank_sample_size",
]


def _z_alpha(alpha: float, sides: int) -> float:
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    if sides not in (1, 2):
        raise ValueError("sides must be 1 or 2")
    return float(norm.ppf(1.0 - alpha / sides))


def _check_hr(hr: float) -> float:
    hr = float(hr)
    if not hr > 0.0 or hr == 1.0 or not math.isfinite(hr):
        raise ValueError("hazard_ratio must be positive, finite and != 1")
    return hr


def _check_alloc(p: float) -> float:
    p = float(p)
    if not 0.0 < p < 1.0:
        raise ValueError("allocation must be in (0, 1)")
    return p


def hazard_from_median(median: float) -> float:
    """Exponential hazard with the given median survival time: ``ln 2 / median``."""
    if not median > 0:
        raise ValueError("median must be positive")
    return math.log(2.0) / float(median)


def logrank_required_events(
    hazard_ratio: float,
    *,
    alpha: float = 0.05,
    power: float = 0.8,
    allocation: float = 0.5,
    method: str = "schoenfeld",
    sides: int = 2,
    exact: bool = False,
):
    """Number of events needed (rounded up unless ``exact=True``)."""
    hr = _check_hr(hazard_ratio)
    p = _check_alloc(allocation)
    if not 0.0 < power < 1.0:
        raise ValueError("power must be in (0, 1)")
    z = _z_alpha(alpha, sides) + float(norm.ppf(power))
    if method == "schoenfeld":
        d = z * z / (p * (1.0 - p) * math.log(hr) ** 2)
    elif method == "freedman":
        k = p / (1.0 - p)
        d = z * z * (1.0 + k * hr) ** 2 / (k * (1.0 - hr) ** 2)
    else:
        raise ValueError("method must be 'schoenfeld' or 'freedman'")
    return d if exact else int(math.ceil(d - 1e-9))


def logrank_power(
    events: float,
    hazard_ratio: float,
    *,
    alpha: float = 0.05,
    allocation: float = 0.5,
    method: str = "schoenfeld",
    sides: int = 2,
) -> float:
    """Approximate power of the log-rank test after ``events`` events."""
    if not events > 0:
        raise ValueError("events must be positive")
    hr = _check_hr(hazard_ratio)
    p = _check_alloc(allocation)
    za = _z_alpha(alpha, sides)
    if method == "schoenfeld":
        drift = math.sqrt(events * p * (1.0 - p)) * abs(math.log(hr))
    elif method == "freedman":
        k = p / (1.0 - p)
        drift = math.sqrt(events * k) * abs(1.0 - hr) / (1.0 + k * hr)
    else:
        raise ValueError("method must be 'schoenfeld' or 'freedman'")
    pw = float(norm.cdf(drift - za))
    if sides == 2:
        pw += float(norm.cdf(-drift - za))
    return pw


def event_probability(
    hazard: float,
    *,
    accrual: float = 0.0,
    follow_up: float,
    dropout_hazard: float = 0.0,
) -> float:
    """P(event observed) under exponential events/drop-out and uniform accrual."""
    if not hazard > 0:
        raise ValueError("hazard must be positive")
    if accrual < 0 or follow_up < 0 or dropout_hazard < 0:
        raise ValueError("accrual, follow_up and dropout_hazard must be non-negative")
    if accrual == 0 and follow_up == 0:
        raise ValueError("accrual + follow_up must be positive")
    total = hazard + dropout_hazard
    share = hazard / total
    if accrual == 0:
        return share * (1.0 - math.exp(-total * follow_up))
    tail = (math.exp(-total * follow_up) - math.exp(-total * (accrual + follow_up))) / (
        total * accrual
    )
    return share * (1.0 - tail)


@dataclass(frozen=True)
class LogRankSampleSize:
    method: str
    hazard_ratio: float
    alpha: float
    power: float
    allocation: float
    events: int
    event_probability_control: float
    event_probability_treatment: float
    event_probability: float
    n_total: int
    n_control: int
    n_treatment: int


def logrank_sample_size(
    hazard_ratio: float,
    *,
    control_hazard: float | None = None,
    control_median: float | None = None,
    accrual: float = 0.0,
    follow_up: float,
    dropout_hazard: float = 0.0,
    alpha: float = 0.05,
    power: float = 0.8,
    allocation: float = 0.5,
    method: str = "schoenfeld",
    sides: int = 2,
) -> LogRankSampleSize:
    """Total subjects so that the expected number of events reaches ``D``.

    Give the control-arm hazard either directly (``control_hazard``) or via
    its median survival (``control_median``). The treatment hazard is
    ``hazard_ratio * control_hazard``. Arm sizes are rounded up separately,
    so ``n_total`` may exceed the unrounded total by one or two.
    """
    if (control_hazard is None) == (control_median is None):
        raise ValueError("give exactly one of control_hazard or control_median")
    lam0 = hazard_from_median(control_median) if control_hazard is None else float(control_hazard)
    if not lam0 > 0:
        raise ValueError("control_hazard must be positive")
    hr = _check_hr(hazard_ratio)
    p = _check_alloc(allocation)
    d = logrank_required_events(
        hr, alpha=alpha, power=power, allocation=p, method=method, sides=sides
    )
    kw = dict(accrual=accrual, follow_up=follow_up, dropout_hazard=dropout_hazard)
    p0 = event_probability(lam0, **kw)
    p1 = event_probability(hr * lam0, **kw)
    pbar = p * p1 + (1.0 - p) * p0
    n = d / pbar
    n_trt = int(math.ceil(p * n - 1e-9))
    n_ctl = int(math.ceil((1.0 - p) * n - 1e-9))
    return LogRankSampleSize(
        method=method,
        hazard_ratio=hr,
        alpha=float(alpha),
        power=float(power),
        allocation=p,
        events=d,
        event_probability_control=p0,
        event_probability_treatment=p1,
        event_probability=pbar,
        n_total=n_trt + n_ctl,
        n_control=n_ctl,
        n_treatment=n_trt,
    )
