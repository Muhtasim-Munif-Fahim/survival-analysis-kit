"""Seeded synthetic survival data with Weibull/exponential latents."""

from __future__ import annotations

import csv
from dataclasses import dataclass

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq

_CONTROL = "control"
_TREATMENT = "treatment"

_CSV_COLUMNS = ("duration", "event", "group")


@dataclass
class SurvivalData:
    """Synthetic right-censored survival sample."""

    durations: np.ndarray
    events: np.ndarray
    groups: np.ndarray
    population_median: float

    @property
    def censor_fraction(self):
        return float(np.mean(~self.events))


def latent_quantile(p, shape, scale):
    """Inverse CDF of ``Weibull(shape, scale)``, the latent event quantiles."""
    return scale * (-np.log(p)) ** (1.0 / shape)


def _expected_censored_share(rate, shape, scale, arm_fraction, ratio):
    """P(censoring time < latent event time) under independent draws.

    Written as ``1 - E[exp(-rate * T)]`` over the two-arm Weibull mixture so
    the quadrature integrand always decays quickly.
    """

    def laplace_transform(arm_scale):
        def integrand(t):
            standardized = t / arm_scale
            return (
                (shape / arm_scale)
                * standardized ** (shape - 1.0)
                * np.exp(-(standardized ** shape))
                * np.exp(-rate * t)
            )

        value, _ = quad(integrand, 0.0, np.inf, limit=200)
        return value

    mixture = (1.0 - arm_fraction) * laplace_transform(
        scale
    ) + arm_fraction * laplace_transform(scale * ratio)
    return 1.0 - mixture


def _calibrate_censor_rate(target, shape, scale, arm_fraction, ratio):
    """Exponential censoring rate whose expected censored share hits ``target``."""
    if target <= 0.0:
        return np.inf

    def excess(rate):
        return (
            _expected_censored_share(rate, shape, scale, arm_fraction, ratio)
            - target
        )

    low, high = 1e-6, 1e4
    if excess(low) > 0.0 or excess(high) < 0.0:
        raise ValueError("censor_fraction is unattainable for these latents")
    return float(brentq(excess, low, high, xtol=1e-10))


def generate_survival_data(
    n,
    *,
    shape=1.0,
    scale=1.0,
    censor_fraction=0.0,
    group_scale_ratio=1.0,
    arm_fraction=0.5,
    seed=0,
):
    """Draw ``n`` right-censored survival times from Weibull latents.

    Latent event times follow ``Weibull(shape, scale)`` in the control arm and
    ``Weibull(shape, scale * group_scale_ratio)`` in the treatment arm, so
    ``shape=1`` reduces both to exponentials. Censoring times are exponential
    and independent of the latents; the rate is calibrated numerically so the
    expected censored share matches ``censor_fraction``.
    """
    if not isinstance(n, (int, np.integer)) or isinstance(n, bool) or n < 1:
        raise ValueError("n must be a positive integer")
    if shape <= 0 or scale <= 0 or group_scale_ratio <= 0:
        raise ValueError("shape, scale, and group_scale_ratio must be positive")
    if not 0.0 <= censor_fraction < 1.0:
        raise ValueError("censor_fraction must lie in [0, 1)")
    if not 0.0 < arm_fraction < 1.0:
        raise ValueError("arm_fraction must lie strictly between 0 and 1")

    rng = np.random.default_rng(seed)
    treated = rng.random(n) < arm_fraction
    groups = np.where(treated, _TREATMENT, _CONTROL)
    arm_scale = np.where(treated, scale * group_scale_ratio, scale)

    uniforms = np.clip(rng.random(n), 1e-300, None)
    latent = arm_scale * (-np.log(uniforms)) ** (1.0 / shape)

    rate = _calibrate_censor_rate(
        censor_fraction, shape, scale, float(arm_fraction), float(group_scale_ratio)
    )
    if np.isfinite(rate):
        censoring = rng.exponential(1.0 / rate, n)
    else:
        censoring = np.full(n, np.inf)

    return SurvivalData(
        durations=np.minimum(latent, censoring),
        events=(latent <= censoring),
        groups=groups,
        population_median=float(latent_quantile(0.5, shape, scale)),
    )


def save_csv(data, path, extra_columns=None):
    """Write durations, events, and groups (plus named extras) to ``path``."""
    extras = {
        name: np.asarray(values, dtype=float)
        for name, values in (extra_columns or {}).items()
    }
    for name, values in extras.items():
        if values.shape[0] != data.durations.shape[0]:
            raise ValueError(f"extra column {name!r} length mismatch")
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(list(_CSV_COLUMNS) + sorted(extras))
        for i in range(data.durations.shape[0]):
            row = [
                repr(float(data.durations[i])),
                int(bool(data.events[i])),
                str(data.groups[i]),
            ]
            row.extend(repr(float(extras[name][i])) for name in sorted(extras))
            writer.writerow(row)


def load_csv(path, time_col="duration", event_col="event"):
    """Read a CSV written by :func:`save_csv`, honoring column aliases.

    Returns ``(durations, events, groups, extra_columns)`` where additional
    numeric columns are collected in ``extra_columns`` by name.
    """
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        missing = {time_col, event_col} - set(fieldnames)
        if missing:
            raise ValueError(f"missing required columns: {sorted(missing)}")
        extra_names = [name for name in fieldnames if name not in _CSV_COLUMNS]
        durations, events, groups = [], [], []
        extras = {name: [] for name in extra_names}
        for row in reader:
            durations.append(float(row[time_col]))
            events.append(bool(int(row[event_col])))
            groups.append(row.get("group", ""))
            for name in extra_names:
                extras[name].append(float(row[name]))

    if not durations:
        raise ValueError(f"{path} contains no observations")
    return (
        np.array(durations, dtype=float),
        np.array(events, dtype=bool),
        np.array(groups),
        {name: np.array(values, dtype=float) for name, values in extras.items()},
    )