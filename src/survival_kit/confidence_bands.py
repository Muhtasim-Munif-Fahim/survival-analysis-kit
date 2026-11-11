"""Simultaneous confidence bands for the Kaplan-Meier survival curve.

Pointwise Greenwood intervals cover ``S(t)`` at one fixed time. A
*simultaneous* band covers the whole curve on ``[t_L, t_U]`` with the
stated probability, which is what a plotted band implicitly claims. Two
classical bands are provided (Klein & Moeschberger 2003, section 4.4):

* **Hall-Wellner** (1980): half-width proportional to ``1 + n sigma^2(t)``
  where ``sigma^2(t) = sum d_i / (Y_i (Y_i - d_i))`` is the Greenwood sum.
  Its critical value ``h`` solves
  ``P(sup_{a_L <= x <= a_U} |W0(x)| <= h) = 1 - alpha`` for a Brownian
  bridge ``W0``.
* **Equal precision** (Nair 1984): half-width proportional to the
  pointwise standard error ``sigma(t)``; the critical value ``e`` solves
  ``P(sup |W0(x)| / sqrt(x (1 - x)) <= e) = 1 - alpha``.

In both cases ``a = n sigma^2(t) / (1 + n sigma^2(t))`` evaluated at
``t_L`` and ``t_U``. Bands come in the ``"linear"`` form or the
``"log-log"`` transformed form (better small-sample coverage, always inside
``[0, 1]``).

Critical values are obtained by simulating the limiting process exactly at
grid points (the Brownian bridge for Hall-Wellner; for equal precision the
standardized bridge, which is a unit Ornstein-Uhlenbeck process in
``tau = log(x / (1 - x)) / 2``) and correcting the discretely monitored
maximum for the continuous supremum with the Broadie-Glasserman-Kou shift
``0.5826 * sqrt(local variance per step)``. With the default fixed seed the
values are reproducible; for ``a_L -> 0, a_U -> 1`` the Hall-Wellner value
reproduces the Kolmogorov quantile (1.358 at 95%).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .utils import validate_durations_events

_BGK = 0.5826  # -zeta(1/2) / sqrt(2 pi)
_METHODS = ("equal-precision", "hall-wellner")
_TRANSFORMS = ("log-log", "linear")


@dataclass
class ConfidenceBand:
    """A simultaneous confidence band on the KM curve over ``[t_lower, t_upper]``."""

    time: np.ndarray
    survival: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    critical_value: float
    a_lower: float
    a_upper: float
    t_lower: float
    t_upper: float
    method: str
    transform: str
    confidence_level: float

    def contains(self, times, survival) -> bool:
        """Whether a curve ``survival(times)`` stays inside the band.

        ``times`` must lie in ``[t_lower, t_upper]``; the band is a
        right-continuous step function over the event times.
        """
        times = np.asarray(times, dtype=float)
        survival = np.asarray(survival, dtype=float)
        if np.any(times < self.t_lower) or np.any(times > self.t_upper):
            raise ValueError("times must lie inside [t_lower, t_upper]")
        idx = np.searchsorted(self.time, times, side="right") - 1
        return bool(np.all((survival >= self.lower[idx] - 1e-12) & (survival <= self.upper[idx] + 1e-12)))


def _sup_brownian_bridge(a_l, a_u, n_sims, n_grid, rng, chunk=2000):
    x = np.linspace(a_l, a_u, n_grid + 1)
    dx = x[1] - x[0] if n_grid > 0 else 0.0
    out = np.empty(n_sims)
    for start in range(0, n_sims, chunk):
        m = min(chunk, n_sims - start)
        w = rng.normal(size=m) * np.sqrt(x[0] * (1.0 - x[0]))
        best = np.abs(w)
        for i in range(1, x.size):
            x0, x1 = x[i - 1], x[i]
            if x1 >= 1.0:
                w = np.zeros(m)
            else:
                mean = w * (1.0 - x1) / (1.0 - x0)
                var = (x1 - x0) * (1.0 - x1) / (1.0 - x0)
                w = mean + rng.normal(size=m) * np.sqrt(max(var, 0.0))
            np.maximum(best, np.abs(w), out=best)
        out[start:start + m] = best
    return out + _BGK * np.sqrt(dx) * (a_u > a_l)


def _sup_standardized_bridge(a_l, a_u, n_sims, n_grid, rng, chunk=2000):
    tau = 0.5 * np.log(np.array([a_l, a_u]) / (1.0 - np.array([a_l, a_u])))
    grid = np.linspace(tau[0], tau[1], n_grid + 1)
    dt = grid[1] - grid[0] if n_grid > 0 else 0.0
    rho = np.exp(-dt)
    sd = np.sqrt(1.0 - rho * rho)
    out = np.empty(n_sims)
    for start in range(0, n_sims, chunk):
        m = min(chunk, n_sims - start)
        u = rng.normal(size=m)
        best = np.abs(u)
        for _ in range(n_grid):
            u = rho * u + sd * rng.normal(size=m)
            np.maximum(best, np.abs(u), out=best)
        out[start:start + m] = best
    return out + _BGK * np.sqrt(2.0 * dt) * (a_u > a_l)


def band_critical_value(
    method,
    a_lower,
    a_upper,
    confidence_level=0.95,
    n_simulations=20000,
    n_grid=1000,
    seed=0,
):
    """Critical value of a Hall-Wellner or equal-precision band.

    ``0 <= a_lower <= a_upper < 1`` (``a_lower > 0`` for equal precision,
    whose standardized process is unbounded at 0).
    """
    if method not in _METHODS:
        raise ValueError("method must be one of %s" % (_METHODS,))
    a_l, a_u = float(a_lower), float(a_upper)
    if not 0.0 <= a_l <= a_u < 1.0:
        raise ValueError("need 0 <= a_lower <= a_upper < 1")
    if method == "equal-precision" and a_l <= 0.0:
        raise ValueError("equal-precision bands need a_lower > 0")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie strictly between 0 and 1")
    if int(n_simulations) < 100 or int(n_grid) < 1:
        raise ValueError("n_simulations must be >= 100 and n_grid >= 1")
    rng = np.random.default_rng(seed)
    if method == "hall-wellner":
        sups = _sup_brownian_bridge(a_l, a_u, int(n_simulations), int(n_grid), rng)
    else:
        sups = _sup_standardized_bridge(a_l, a_u, int(n_simulations), int(n_grid), rng)
    return float(np.quantile(sups, confidence_level))


def km_confidence_band(
    durations,
    events,
    method="equal-precision",
    transform="log-log",
    confidence_level=0.95,
    t_lower=None,
    t_upper=None,
    n_simulations=20000,
    n_grid=1000,
    seed=0,
):
    """Simultaneous Hall-Wellner or equal-precision band for the KM curve.

    ``t_lower`` / ``t_upper`` default to the first event time and the last
    event time at which the curve is still positive with a finite Greenwood
    variance. The band is reported at the event times inside that range.
    """
    if method not in _METHODS:
        raise ValueError("method must be one of %s" % (_METHODS,))
    if transform not in _TRANSFORMS:
        raise ValueError("transform must be one of %s" % (_TRANSFORMS,))
    durations, events = validate_durations_events(durations, events)
    n = durations.size
    times = np.unique(durations[events])
    if times.size < 2:
        raise ValueError("need at least two distinct event times")
    at_risk = np.array([np.count_nonzero(durations >= t) for t in times], dtype=float)
    deaths = np.array([np.count_nonzero((durations == t) & events) for t in times], dtype=float)
    survival = np.cumprod(1.0 - deaths / at_risk)
    denom = at_risk * (at_risk - deaths)
    sigma2 = np.cumsum(np.where(denom > 0, deaths / np.where(denom > 0, denom, 1.0), np.inf))

    usable = (survival > 0.0) & np.isfinite(sigma2) & (sigma2 > 0.0)
    if not np.any(usable):
        raise ValueError("no event time with a usable variance estimate")
    t_lo = float(times[usable][0]) if t_lower is None else float(t_lower)
    t_hi = float(times[usable][-1]) if t_upper is None else float(t_upper)
    if t_lo > t_hi:
        raise ValueError("t_lower must not exceed t_upper")
    keep = usable & (times >= t_lo) & (times <= t_hi)
    if not np.any(keep):
        raise ValueError("no usable event times inside [t_lower, t_upper]")

    def _sigma2_at(t):
        i = np.searchsorted(times, t, side="right") - 1
        return float(sigma2[i]) if i >= 0 else 0.0

    s2_lo, s2_hi = _sigma2_at(t_lo), _sigma2_at(t_hi)
    if not (np.isfinite(s2_lo) and np.isfinite(s2_hi)):
        raise ValueError("variance is infinite inside [t_lower, t_upper]")
    a_l = n * s2_lo / (1.0 + n * s2_lo)
    a_u = n * s2_hi / (1.0 + n * s2_hi)
    if method == "equal-precision" and a_l <= 0.0:
        raise ValueError("equal-precision bands need t_lower at or after the first event time")
    c = band_critical_value(method, a_l, a_u, confidence_level, n_simulations, n_grid, seed)

    t, s, v = times[keep], survival[keep], sigma2[keep]
    if method == "hall-wellner":
        scale = (1.0 + n * v) / np.sqrt(n)
    else:
        scale = np.sqrt(v)
    if transform == "linear":
        lower = np.clip(s - c * scale * s, 0.0, 1.0)
        upper = np.clip(s + c * scale * s, 0.0, 1.0)
    else:
        lower = s.copy()
        upper = s.copy()
        inner = s < 1.0
        theta = np.exp(c * scale[inner] / np.log(s[inner]))
        lower[inner] = s[inner] ** (1.0 / theta)
        upper[inner] = s[inner] ** theta
    return ConfidenceBand(
        time=t,
        survival=s,
        lower=lower,
        upper=upper,
        critical_value=c,
        a_lower=float(a_l),
        a_upper=float(a_u),
        t_lower=t_lo,
        t_upper=t_hi,
        method=method,
        transform=transform,
        confidence_level=float(confidence_level),
    )
