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


@dataclass
class PHSurvivalData:
    """Right-censored sample from a Weibull proportional-hazards model."""

    durations: np.ndarray
    events: np.ndarray
    covariates: np.ndarray
    coefficients: np.ndarray
    population_median: float

    @property
    def censor_fraction(self):
        return float(np.mean(~self.events))


@dataclass
class CompetingRisksData:
    """Right-censored sample with integer competing-event codes.

    ``event_types`` uses ``0`` for censoring and ``1, 2, ...`` for mutually
    exclusive causes. Control-arm latents are independent exponentials with
    rates ``cause_rates``; the treatment arm multiplies those rates by
    ``group_rate_ratios``.
    """

    durations: np.ndarray
    event_types: np.ndarray
    groups: np.ndarray
    cause_rates: np.ndarray
    group_rate_ratios: np.ndarray

    @property
    def events(self):
        return self.event_types > 0

    @property
    def censor_fraction(self):
        return float(np.mean(self.event_types == 0))

    def true_cif(self, time, cause=1, group="control"):
        """Closed-form exponential competing-risks CIF for ``group``."""
        rates = np.array(self.cause_rates, dtype=float, copy=True)
        if group == _TREATMENT:
            rates = rates * self.group_rate_ratios
        elif group != _CONTROL:
            raise ValueError("group must be 'control' or 'treatment'")
        return exponential_competing_cif(time, rates, cause=cause)


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


def generate_ph_data(
    n,
    coefficients,
    *,
    covariates=None,
    shape=1.0,
    scale=1.0,
    censor_fraction=0.0,
    seed=0,
):
    """Draw right-censored times from a Weibull model with proportional hazards.

    Latent event times satisfy
    ``S(t | x) = exp(-(t / scale)**shape * exp(x @ beta))``, so ``coefficients``
    are the true log hazard ratios. Covariates default to independent standard
    normals. Independent exponential censoring is calibrated on the realized
    latents so the expected censored share matches ``censor_fraction``.
    """
    if not isinstance(n, (int, np.integer)) or isinstance(n, bool) or n < 1:
        raise ValueError("n must be a positive integer")
    if shape <= 0 or scale <= 0:
        raise ValueError("shape and scale must be positive")
    if not 0.0 <= censor_fraction < 1.0:
        raise ValueError("censor_fraction must lie in [0, 1)")

    beta = np.asarray(coefficients, dtype=float).reshape(-1)
    if beta.size < 1 or not np.all(np.isfinite(beta)):
        raise ValueError("coefficients must be a finite non-empty vector")

    rng = np.random.default_rng(seed)
    if covariates is None:
        X = rng.standard_normal((n, beta.size))
    else:
        X = np.asarray(covariates, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.shape != (n, beta.size):
            raise ValueError("covariates must have shape (n, p) matching coefficients")
        if not np.all(np.isfinite(X)):
            raise ValueError("covariates must contain only finite values")

    uniforms = np.clip(rng.random(n), 1e-300, None)
    linear_pred = X @ beta
    latent = scale * ((-np.log(uniforms)) / np.exp(linear_pred)) ** (1.0 / shape)

    if censor_fraction <= 0.0:
        censoring = np.full(n, np.inf)
    else:
        def excess(rate):
            return float(np.mean(1.0 - np.exp(-rate * latent))) - censor_fraction

        rate = float(brentq(excess, 1e-12, 1e6, xtol=1e-10))
        censoring = rng.exponential(1.0 / rate, n)

    return PHSurvivalData(
        durations=np.minimum(latent, censoring),
        events=(latent <= censoring),
        covariates=X,
        coefficients=np.array(beta, copy=True),
        population_median=float(latent_quantile(0.5, shape, scale)),
    )


def exponential_competing_cif(time, cause_rates, cause=1):
    """Closed-form CIF for independent exponential causes.

    ``F_k(t) = rate_k / rate_total * (1 - exp(-rate_total * t))``.
    """
    rates = np.asarray(cause_rates, dtype=float).reshape(-1)
    if rates.size < 1 or np.any(rates <= 0) or not np.all(np.isfinite(rates)):
        raise ValueError("cause_rates must be a finite vector of positive rates")
    cause = int(cause)
    if cause < 1 or cause > rates.size:
        raise ValueError("cause must index a provided rate")
    total = float(rates.sum())
    time = np.asarray(time, dtype=float)
    value = (rates[cause - 1] / total) * (1.0 - np.exp(-total * time))
    return float(value) if value.ndim == 0 else value


def _cr_expected_censored_share(censor_rate, total_control, total_treatment, arm_fraction):
    """P(C < min_k T_k) under independent exponential censoring and causes."""
    share_control = censor_rate / (censor_rate + total_control)
    share_treatment = censor_rate / (censor_rate + total_treatment)
    return (1.0 - arm_fraction) * share_control + arm_fraction * share_treatment


def _calibrate_cr_censor_rate(target, total_control, total_treatment, arm_fraction):
    """Exponential censoring rate matching ``target`` in a two-arm CR mixture."""
    if target <= 0.0:
        return np.inf

    def excess(rate):
        return (
            _cr_expected_censored_share(
                rate, total_control, total_treatment, arm_fraction
            )
            - target
        )

    low, high = 1e-12, 1e6
    if excess(low) > 0.0 or excess(high) < 0.0:
        raise ValueError("censor_fraction is unattainable for these latents")
    return float(brentq(excess, low, high, xtol=1e-10))


def generate_competing_risks_data(
    n,
    *,
    cause_rates=(0.4, 0.3),
    censor_fraction=0.0,
    group_rate_ratios=1.0,
    arm_fraction=0.5,
    seed=0,
):
    """Draw right-censored competing-risks times from independent exponentials.

    Each subject has a latent exponential time per cause; the observed time is
    the minimum of those latents and an independent exponential censoring time.
    The event type is the argmin cause, or ``0`` if censoring wins. Treatment
    multiplies control-arm ``cause_rates`` by ``group_rate_ratios`` (a scalar
    or a per-cause vector). The censoring rate is calibrated so the expected
    censored share matches ``censor_fraction``.
    """
    if not isinstance(n, (int, np.integer)) or isinstance(n, bool) or n < 1:
        raise ValueError("n must be a positive integer")
    if not 0.0 <= censor_fraction < 1.0:
        raise ValueError("censor_fraction must lie in [0, 1)")
    if not 0.0 < arm_fraction < 1.0:
        raise ValueError("arm_fraction must lie strictly between 0 and 1")

    rates = np.asarray(cause_rates, dtype=float).reshape(-1)
    if rates.size < 2 or np.any(rates <= 0) or not np.all(np.isfinite(rates)):
        raise ValueError("cause_rates must contain at least two positive finite rates")

    ratios = np.asarray(group_rate_ratios, dtype=float).reshape(-1)
    if ratios.size == 1:
        ratios = np.full(rates.size, float(ratios[0]))
    if ratios.shape != rates.shape or np.any(ratios <= 0) or not np.all(np.isfinite(ratios)):
        raise ValueError("group_rate_ratios must be a positive scalar or match cause_rates")

    rng = np.random.default_rng(seed)
    treated = rng.random(n) < arm_fraction
    groups = np.where(treated, _TREATMENT, _CONTROL)
    subject_rates = np.where(treated.reshape(-1, 1), rates * ratios, rates)

    latents = rng.exponential(1.0 / subject_rates)
    latent_time = latents.min(axis=1)
    cause = latents.argmin(axis=1) + 1

    total_control = float(rates.sum())
    total_treatment = float((rates * ratios).sum())
    censor_rate = _calibrate_cr_censor_rate(
        censor_fraction, total_control, total_treatment, float(arm_fraction)
    )
    if np.isfinite(censor_rate):
        censoring = rng.exponential(1.0 / censor_rate, n)
    else:
        censoring = np.full(n, np.inf)

    censored = censoring < latent_time
    event_types = np.where(censored, 0, cause).astype(int)
    return CompetingRisksData(
        durations=np.minimum(latent_time, censoring),
        event_types=event_types,
        groups=groups,
        cause_rates=np.array(rates, copy=True),
        group_rate_ratios=np.array(ratios, copy=True),
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
        event_codes = getattr(data, "event_types", None)
        for i in range(data.durations.shape[0]):
            event_value = (
                int(event_codes[i])
                if event_codes is not None
                else int(bool(data.events[i]))
            )
            row = [
                repr(float(data.durations[i])),
                event_value,
                str(data.groups[i]),
            ]
            row.extend(repr(float(extras[name][i])) for name in sorted(extras))
            writer.writerow(row)


def load_csv(path, time_col="duration", event_col="event", with_event_types=False):
    """Read a CSV written by :func:`save_csv`, honoring column aliases.

    Returns ``(durations, events, groups, extra_columns)`` where additional
    numeric columns are collected in ``extra_columns`` by name. Event columns
    may hold integer cause codes; ``events`` is the boolean mask of any
    failure. Pass ``with_event_types=True`` to also receive the integer codes.
    """
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        missing = {time_col, event_col} - set(fieldnames)
        if missing:
            raise ValueError(f"missing required columns: {sorted(missing)}")
        extra_names = [name for name in fieldnames if name not in _CSV_COLUMNS]
        durations, event_types, groups = [], [], []
        extras = {name: [] for name in extra_names}
        for row in reader:
            durations.append(float(row[time_col]))
            event_types.append(int(row[event_col]))
            groups.append(row.get("group", ""))
            for name in extra_names:
                extras[name].append(float(row[name]))

    if not durations:
        raise ValueError(f"{path} contains no observations")
    event_types = np.array(event_types, dtype=int)
    result = (
        np.array(durations, dtype=float),
        event_types != 0,
        np.array(groups),
        {name: np.array(values, dtype=float) for name, values in extras.items()},
    )
    if with_event_types:
        return result + (event_types,)
    return result