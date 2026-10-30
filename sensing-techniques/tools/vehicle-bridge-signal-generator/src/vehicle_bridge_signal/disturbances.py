"""Other disturbances: wheel eccentricity and ambient bridge vibration.

Wheel eccentricity (unbalance / radial run-out)
    A harmonic contact displacement e(t) = A sin(2 pi f_w t + psi) at the wheel rotation frequency
    f_w = v / (pi D). Velocity and acceleration are analytical. Each axle has its own phase.

Ambient vibration (background traffic or wind)
    Independent white-noise modal forces, one per bridge mode. For a mode with modal mass M,
    damping zeta and circular frequency omega driven by discrete white noise of standard deviation
    sigma_f at rate fs (one-sided PSD 2 sigma_f^2 / fs), the stationary displacement variance is::

        sigma_q^2 = sigma_f^2 / (4 zeta omega^3 M^2 fs)

    The level is given as a_rms = omega^2 sigma_q (narrow-band modal acceleration at the antinode).
    The initial state is drawn from the stationary distribution (q ~ N(0, sigma_q^2),
    q' ~ N(0, omega^2 sigma_q^2), independent), so the bridge is already vibrating when the vehicle
    enters and no long warm-up is needed for lightly damped bridges.
"""
from __future__ import annotations

import numpy as np

from .bridge import Bridge

__all__ = ["wheel_frequency", "wheel_eccentricity", "ambient_modal_forces"]


def wheel_frequency(speed: float, diameter: float) -> float:
    """f_w = v / (pi D) [Hz]."""
    return speed / (np.pi * diameter)


def wheel_eccentricity(rng: np.random.Generator, t: np.ndarray, n_axles: int, speed: float, diameter: float,
                       amplitude: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Contact displacement, velocity and acceleration (shape ``(n_steps, n_axles)``) and f_w."""
    fw = wheel_frequency(speed, diameter)
    w = 2 * np.pi * fw
    psi = rng.uniform(0.0, 2.0 * np.pi, n_axles)
    arg = w * np.asarray(t, float)[:, None] + psi
    return amplitude * np.sin(arg), amplitude * w * np.cos(arg), -amplitude * w**2 * np.sin(arg), fw


def ambient_modal_forces(rng: np.random.Generator, bridge: Bridge, n_steps: int, fs: float,
                         modal_accel_rms: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """White-noise modal forces ``(n_steps, n_modes)`` [N], stationary q0, q'0 and sigma_q per mode."""
    if np.any(bridge.damping_ratios <= 0):
        raise ValueError("Ambient vibration needs a positive damping ratio in every bridge mode "
                         "(an undamped mode has no stationary state)")
    w = bridge.omegas
    sigma_q = modal_accel_rms / w**2
    sigma_f = sigma_q * bridge.modal_mass * np.sqrt(4.0 * bridge.damping_ratios * w**3 * fs)
    forces = rng.standard_normal((n_steps, bridge.n_modes)) * sigma_f
    q0 = rng.standard_normal(bridge.n_modes) * sigma_q
    qd0 = rng.standard_normal(bridge.n_modes) * sigma_q * w
    return forces, q0, qd0, sigma_q
