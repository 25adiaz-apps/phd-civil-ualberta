"""Engine excitation: sum of engine orders with a slowly varying rpm.

Four-stroke engine firing frequency: f_fire = rpm * N_cyl / 120, i.e. order o_fire = N_cyl / 2
(second order for four cylinders; Burdzik). Force on the sprung mass::

    F_e(t)   = sum_k A_k sin(Phi_k(t))
    Phi_k(t) = 2 pi o_k int_0^t rpm(tau) / 60 dtau + phi_k

The phase integrates the instantaneous frequency (never 2 pi f(t) t). rpm(t) is a smooth stationary
Gaussian process with a Gaussian correlation function exp(-(dt / tau)^2), synthesised as a sum of
cosines, then soft-limited to +/- 4 standard deviations with tanh (keeps it smooth and positive).
The absolute amplitude A_k is set afterwards from a target body acceleration RMS (the system is
linear), so here the force is built with relative amplitudes.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["firing_order", "firing_frequency", "EngineOrders", "smooth_gaussian_process", "engine_force"]


def firing_order(n_cylinders: int) -> float:
    """Firing order of a four-stroke engine: N_cyl / 2."""
    return n_cylinders / 2.0


def firing_frequency(rpm, n_cylinders: int):
    """f_fire = rpm N_cyl / 120 [Hz]."""
    return np.asarray(rpm) * n_cylinders / 120.0


@dataclass(frozen=True)
class EngineOrders:
    orders: tuple[float, ...]
    relative_amplitudes: tuple[float, ...]

    @classmethod
    def from_config(cls, n_cylinders: int, firing: float, double_firing: float, half_order: float) -> "EngineOrders":
        of = firing_order(n_cylinders)
        pairs = [(of, firing), (2 * of, double_firing), (0.5, half_order)]
        pairs = [(o, a) for o, a in pairs if a > 0]
        return cls(tuple(o for o, _ in pairs), tuple(float(a) for _, a in pairs))


def smooth_gaussian_process(rng: np.random.Generator, t: np.ndarray, correlation_time: float) -> np.ndarray:
    """Zero-mean, unit-variance smooth Gaussian process with covariance exp(-(dt/tau)^2).

    One-sided PSD: S(f) = 2 sqrt(pi) tau exp(-(pi f tau)^2). Harmonic synthesis with a frequency
    step 1/(4 T) up to 3/tau, amplitudes normalised so that the variance is exactly 1.
    """
    t = np.asarray(t, float)
    duration = max(t[-1] - t[0], correlation_time)
    df = 1.0 / (4.0 * duration)
    f = np.arange(1, int(np.ceil(3.0 / correlation_time / df)) + 1) * df
    s = 2.0 * np.sqrt(np.pi) * correlation_time * np.exp(-((np.pi * f * correlation_time) ** 2))
    a = np.sqrt(2.0 * s * df)
    a /= np.sqrt(0.5 * np.sum(a**2))
    phases = rng.uniform(0.0, 2.0 * np.pi, f.size)
    tt = t - t[0]
    out = np.zeros_like(tt)
    for s0 in range(0, f.size, 256):
        sl = slice(s0, s0 + 256)
        out += np.cos(2 * np.pi * np.outer(tt, f[sl]) + phases[sl]) @ a[sl]
    return out


def engine_force(rng: np.random.Generator, t: np.ndarray, orders: EngineOrders, rpm_mean: float,
                 rpm_relative_std: float, correlation_time: float) -> tuple[np.ndarray, np.ndarray]:
    """Unit-scale engine force (relative amplitudes) and rpm history on the time grid ``t``.

    Returns ``(force, rpm)``. Uses its own generator: rpm process first, then one phase per order.
    """
    t = np.asarray(t, float)
    if rpm_relative_std > 0:
        g = smooth_gaussian_process(rng, t, correlation_time)
        g = 4.0 * np.tanh(g / 4.0)
    else:
        g = np.zeros_like(t)
    rpm = rpm_mean * (1.0 + rpm_relative_std * g)
    # revolutions(t) = int rpm/60 dt by the trapezoidal rule (rpm is smooth: error ~ dt^2).
    rev = np.concatenate(([0.0], np.cumsum(0.5 * (rpm[1:] + rpm[:-1]) * np.diff(t)))) / 60.0
    phases = rng.uniform(0.0, 2.0 * np.pi, len(orders.orders))
    force = np.zeros_like(t)
    for o, a, ph in zip(orders.orders, orders.relative_amplitudes, phases):
        force += a * np.sin(2.0 * np.pi * o * rev + ph)
    return force, rpm
