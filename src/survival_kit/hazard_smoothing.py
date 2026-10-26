"""Kernel-smoothed hazard rate estimation (Ramlau-Hansen, 1983).

The Nelson-Aalen estimator ``H(t) = sum_{t_j <= t} d_j / n_j`` is a step
function, so its increments ``dH_j = d_j / n_j`` cannot be plotted as a
hazard *rate*. Ramlau-Hansen (1983) smooths them with a kernel ``K`` and
bandwidth ``b``:

    h(t) = (1 / b) * sum_j K((t - t_j) / b) * dH_j

with pointwise variance (Aalen form)

    Var h(t) = (1 / b^2) * sum_j K((t - t_j) / b)^2 * d_j / n_j^2.

Kernels (Klein & Moeschberger 2003, section 6.2):

* ``"uniform"``      ``K(x) = 1/2``
* ``"epanechnikov"`` ``K(x) = 3/4 (1 - x^2)``
* ``"biweight"``     ``K(x) = 15/16 (1 - x^2)^2``

all on ``[-1, 1]``. Near the boundaries ``t < b`` and ``t > t_D - b``
(``t_D`` the largest event time) a symmetric kernel loses mass and is biased.
With ``boundary_correction=True`` the Gasser-Müller / Müller-Wang (1994)
boundary kernels ``K_q(x) = K(x) (alpha(q) + beta(q) x)`` (for the uniform
kernel ``alpha(q) + beta(q) x``) are used, with
``q = t / b`` on the left and ``q = (t_D - t) / b`` on the right
(``K_q(-x)``). They integrate to one and have zero first moment on the
truncated support.

Confidence intervals are log-transformed so they stay positive:
``h(t) exp(+- z SE / h(t))``.

``select_bandwidth_cv`` picks ``b`` by minimising the Ramlau-Hansen
least-squares cross-validation score (K&M eq. 6.2.7):

    g(b) = int h_b(t)^2 dt - (2 / b) sum_{i != j} K((t_i - t_j) / b) dH_i dH_j.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from scipy.stats import norm

from .utils import validate_durations_events

_trapezoid = getattr(np, "trapezoid", None) or np.trapz

KERNELS = ("uniform", "epanechnikov", "biweight")

__all__ = [
    "KERNELS",
    "SmoothedHazard",
    "BandwidthCVResult",
    "boundary_kernel",
    "kernel",
    "fit_smoothed_hazard",
    "select_bandwidth_cv",
]


def kernel(x, name: str = "epanechnikov") -> np.ndarray:
    """Symmetric kernel on [-1, 1]."""
    x = np.asarray(x, dtype=float)
    inside = np.abs(x) <= 1.0
    if name == "uniform":
        k = np.full_like(x, 0.5)
    elif name == "epanechnikov":
        k = 0.75 * (1.0 - x**2)
    elif name == "biweight":
        k = (15.0 / 16.0) * (1.0 - x**2) ** 2
    else:
        raise ValueError(f"kernel must be one of {KERNELS}")
    return np.where(inside, k, 0.0)


def _boundary_coefficients(q: float, name: str):
    q = float(q)
    if name == "uniform":
        alpha = 4.0 * (1.0 + q**3) / (1.0 + q) ** 4
        beta = 6.0 * (1.0 - q) / (1.0 + q) ** 3
    elif name == "epanechnikov":
        den = (1.0 + q) ** 4 * (19.0 - 18.0 * q + 3.0 * q**2)
        alpha = 64.0 * (2.0 - 4.0 * q + 6.0 * q**2 - 3.0 * q**3) / den
        beta = 240.0 * (1.0 - q) ** 2 / den
    elif name == "biweight":
        den = (1.0 + q) ** 5 * (81.0 - 168.0 * q + 126.0 * q**2 - 40.0 * q**3 + 5.0 * q**4)
        alpha = 64.0 * (8.0 - 24.0 * q + 48.0 * q**2 - 45.0 * q**3 + 15.0 * q**4) / den
        beta = 1120.0 * (1.0 - q) ** 3 / den
    else:
        raise ValueError(f"kernel must be one of {KERNELS}")
    return alpha, beta


def boundary_kernel(x, q: float, name: str = "epanechnikov") -> np.ndarray:
    """Left-boundary kernel ``K_q(x) = K(x)(alpha + beta x)`` on ``[-1, q]``.

    ``q in [0, 1]``; ``q = 1`` recovers the symmetric kernel.
    """
    if not 0.0 <= q <= 1.0:
        raise ValueError("q must lie in [0, 1]")
    x = np.asarray(x, dtype=float)
    alpha, beta = _boundary_coefficients(q, name)
    # K&M state the uniform boundary kernel as alpha + beta x itself; the
    # Epanechnikov and biweight versions multiply the symmetric kernel.
    base = np.where(np.abs(x) <= 1.0, 1.0, 0.0) if name == "uniform" else kernel(x, name)
    k = base * (alpha + beta * x)
    return np.where(x <= q, k, 0.0)


@dataclass
class SmoothedHazard:
    """Kernel-smoothed hazard rate on an evaluation grid."""

    time: np.ndarray
    hazard: np.ndarray
    std_err: np.ndarray
    ci_lower: np.ndarray
    ci_upper: np.ndarray
    bandwidth: float
    kernel: str
    boundary_correction: bool
    confidence_level: float

    def to_dict(self) -> dict:
        return {
            "bandwidth": self.bandwidth,
            "kernel": self.kernel,
            "boundary_correction": self.boundary_correction,
            "confidence_level": self.confidence_level,
            "time": self.time.tolist(),
            "hazard": self.hazard.tolist(),
            "std_err": self.std_err.tolist(),
            "ci_lower": self.ci_lower.tolist(),
            "ci_upper": self.ci_upper.tolist(),
        }


@dataclass
class BandwidthCVResult:
    """Cross-validation scores over candidate bandwidths."""

    bandwidths: np.ndarray
    scores: np.ndarray
    best_bandwidth: float


def _increments(durations, events):
    event_times = np.unique(durations[events])
    if event_times.size == 0:
        raise ValueError("at least one event is required to estimate a hazard")
    at_risk = np.array([np.count_nonzero(durations >= t) for t in event_times], dtype=float)
    deaths = np.array(
        [np.count_nonzero((durations == t) & events) for t in event_times], dtype=float
    )
    return event_times, deaths / at_risk, deaths / at_risk**2


def _weights(t: float, event_times, b, name, boundary_correction, t_max):
    x = (t - event_times) / b
    if boundary_correction:
        if t < b:
            return boundary_kernel(x, t / b, name)
        if t > t_max - b:
            return boundary_kernel(-x, (t_max - t) / b, name)
    return kernel(x, name)


def _default_bandwidth(event_times) -> float:
    span = float(event_times.max() - event_times.min())
    if span <= 0:
        span = float(event_times.max()) or 1.0
    return span / 4.0


def fit_smoothed_hazard(
    durations,
    events,
    bandwidth: Optional[float] = None,
    kernel_name: str = "epanechnikov",
    times: Optional[Sequence[float]] = None,
    n_points: int = 101,
    boundary_correction: bool = True,
    confidence_level: float = 0.95,
) -> SmoothedHazard:
    """Ramlau-Hansen kernel-smoothed hazard rate with pointwise log-CIs.

    ``times`` defaults to ``n_points`` evenly spaced points on
    ``[0, t_D]``. Without boundary correction the grid runs over
    ``[b, t_D - b]`` when that interval is non-empty.
    """
    durations, events = validate_durations_events(durations, events)
    if kernel_name not in KERNELS:
        raise ValueError(f"kernel must be one of {KERNELS}")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie strictly between 0 and 1")
    event_times, dH, var_inc = _increments(durations, events)
    t_max = float(event_times.max())
    b = _default_bandwidth(event_times) if bandwidth is None else float(bandwidth)
    if not np.isfinite(b) or b <= 0:
        raise ValueError("bandwidth must be a finite positive number")
    if boundary_correction and 2.0 * b > t_max:
        raise ValueError(
            "bandwidth must be at most half the largest event time when boundary_correction=True"
        )
    if times is None:
        if n_points < 2:
            raise ValueError("n_points must be at least 2")
        lo, hi = (0.0, t_max)
        if not boundary_correction and t_max - b > b:
            lo, hi = b, t_max - b
        grid = np.linspace(lo, hi, n_points)
    else:
        grid = np.asarray(times, dtype=float).ravel()
        if grid.size == 0 or not np.all(np.isfinite(grid)):
            raise ValueError("times must be a non-empty array of finite values")
        if boundary_correction and (np.any(grid < 0) or np.any(grid > t_max)):
            raise ValueError("with boundary_correction, times must lie in [0, largest event time]")

    hazard = np.empty(grid.size)
    var = np.empty(grid.size)
    for i, t in enumerate(grid):
        w = _weights(t, event_times, b, kernel_name, boundary_correction, t_max)
        hazard[i] = np.dot(w, dH) / b
        var[i] = np.dot(w**2, var_inc) / b**2
    # Boundary kernels can go slightly negative; a hazard cannot.
    hazard = np.maximum(hazard, 0.0)
    se = np.sqrt(var)
    z = float(norm.ppf(0.5 + confidence_level / 2.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        factor = np.where(hazard > 0, np.exp(z * se / np.where(hazard > 0, hazard, 1.0)), np.nan)
    lower = np.where(hazard > 0, hazard / factor, 0.0)
    upper = np.where(hazard > 0, hazard * factor, z * se)
    return SmoothedHazard(
        time=grid,
        hazard=hazard,
        std_err=se,
        ci_lower=lower,
        ci_upper=upper,
        bandwidth=b,
        kernel=kernel_name,
        boundary_correction=boundary_correction,
        confidence_level=confidence_level,
    )


def select_bandwidth_cv(
    durations,
    events,
    bandwidths: Optional[Sequence[float]] = None,
    kernel_name: str = "epanechnikov",
    boundary_correction: bool = True,
    n_grid: int = 201,
) -> BandwidthCVResult:
    """Least-squares cross-validation choice of the bandwidth (K&M eq. 6.2.7)."""
    durations, events = validate_durations_events(durations, events)
    if kernel_name not in KERNELS:
        raise ValueError(f"kernel must be one of {KERNELS}")
    event_times, dH, _ = _increments(durations, events)
    t_max = float(event_times.max())
    if bandwidths is None:
        hi = t_max / 2.0 if boundary_correction else t_max
        bandwidths = np.linspace(hi / 20.0, hi, 20)
    bandwidths = np.asarray(bandwidths, dtype=float).ravel()
    if bandwidths.size == 0 or np.any(~np.isfinite(bandwidths)) or np.any(bandwidths <= 0):
        raise ValueError("bandwidths must be finite and positive")
    diff = event_times[:, None] - event_times[None, :]
    off = ~np.eye(event_times.size, dtype=bool)
    pair = np.outer(dH, dH)
    scores = np.empty(bandwidths.size)
    for k, b in enumerate(bandwidths):
        if boundary_correction and 2.0 * b > t_max:
            scores[k] = np.inf
            continue
        fit = fit_smoothed_hazard(
            durations,
            events,
            bandwidth=b,
            kernel_name=kernel_name,
            times=np.linspace(0.0, t_max, n_grid),
            boundary_correction=boundary_correction,
        )
        integral = float(_trapezoid(fit.hazard**2, fit.time))
        cross = float(np.sum(kernel(diff / b, kernel_name)[off] * pair[off]))
        scores[k] = integral - 2.0 * cross / b
    if not np.any(np.isfinite(scores)):
        raise ValueError("no candidate bandwidth is admissible")
    best = float(bandwidths[int(np.argmin(scores))])
    return BandwidthCVResult(bandwidths=bandwidths, scores=scores, best_bandwidth=best)
