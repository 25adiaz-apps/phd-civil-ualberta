"""Simply supported Euler-Bernoulli beam described by modal superposition.

Sign convention (used everywhere in the package)
------------------------------------------------
* Vertical displacements (bridge deflection ``w``, vehicle DOFs, roughness ``r``) are positive
  **upwards**.
* Vertical forces are positive upwards. The static weight of an axle therefore enters the bridge
  as a negative force ``P = -W`` and produces a negative (downward) deflection.

Modal model::

    phi_n(x) = sin(n pi x / L)                    (zero outside 0 <= x <= L)
    omega_n  = (n pi / L)^2 sqrt(EI / m)           ->  f_n = n^2 f_1
    M_n      = m L / 2
    q_n'' + 2 zeta_n omega_n q_n' + omega_n^2 q_n = (1/M_n) sum_a phi_n(x_a) P_a
    w(x, t)  = sum_n phi_n(x) q_n(t)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["Bridge", "nominal_f1", "resolve_f_max", "count_modes", "flexural_rigidity_from_f1",
           "f1_from_flexural_rigidity"]


def flexural_rigidity_from_f1(span: float, f1: float, mass_per_length: float) -> float:
    """EI = m (2 L^2 f1 / pi)^2, from f1 = (pi / (2 L^2)) sqrt(EI / m)."""
    return mass_per_length * (2.0 * span**2 * f1 / np.pi) ** 2


def f1_from_flexural_rigidity(span: float, flexural_rigidity: float, mass_per_length: float) -> float:
    """f1 = (pi / (2 L^2)) sqrt(EI / m)."""
    return np.pi / (2.0 * span**2) * np.sqrt(flexural_rigidity / mass_per_length)


def nominal_f1(span: float, table_span=(15.0, 30.0, 50.0, 100.0), table_f1=(8.03, 3.63, 2.05, 0.75)) -> float:
    """Nominal first frequency [Hz] by log-log interpolation (linear extrapolation in log-log outside).

    Default table: Talebi-Kalaleh & Mei (2026), table 5.
    """
    ls, lf = np.log(np.asarray(table_span, float)), np.log(np.asarray(table_f1, float))
    x = np.log(float(span))
    if x <= ls[0]:
        slope = (lf[1] - lf[0]) / (ls[1] - ls[0])
        return float(np.exp(lf[0] + slope * (x - ls[0])))
    if x >= ls[-1]:
        slope = (lf[-1] - lf[-2]) / (ls[-1] - ls[-2])
        return float(np.exp(lf[-1] + slope * (x - ls[-1])))
    return float(np.exp(np.interp(x, ls, lf)))


def resolve_f_max(f_max, fs_out: float) -> float:
    """``'auto'`` -> min(0.45 fs_out, 40 Hz)."""
    if f_max is None or f_max == "auto":
        return min(0.45 * fs_out, 40.0)
    return float(f_max)


def count_modes(f1: float, f_max: float, max_modes: int) -> int:
    """Number of modes with f_n = n^2 f1 <= f_max, between 1 and ``max_modes``."""
    n = int(np.floor(np.sqrt(f_max / f1) + 1e-12))
    return int(min(max(n, 1), max_modes))


@dataclass(frozen=True)
class Bridge:
    """Simply supported beam in modal coordinates.

    Parameters
    ----------
    span : L [m]
    f1 : first natural frequency [Hz]
    mass_per_length : m [kg/m]
    damping_ratios : one damping ratio per retained mode [-]
    """

    span: float
    f1: float
    mass_per_length: float
    damping_ratios: np.ndarray = field(repr=False)

    def __post_init__(self):
        z = np.atleast_1d(np.asarray(self.damping_ratios, dtype=float)).copy()
        z.setflags(write=False)
        object.__setattr__(self, "damping_ratios", z)
        if not (self.span > 0 and self.f1 > 0 and self.mass_per_length > 0):
            raise ValueError("Bridge span, f1 and mass_per_length must be positive")
        if z.size < 1 or np.any(z < 0) or np.any(z >= 1):
            raise ValueError("Bridge damping ratios must be in [0, 1) and at least one mode is needed")

    # ---- derived properties -----------------------------------------------------------------
    @property
    def n_modes(self) -> int:
        return int(self.damping_ratios.size)

    @property
    def mode_numbers(self) -> np.ndarray:
        return np.arange(1, self.n_modes + 1)

    @property
    def flexural_rigidity(self) -> float:
        """EI [N m^2]."""
        return flexural_rigidity_from_f1(self.span, self.f1, self.mass_per_length)

    @property
    def frequencies(self) -> np.ndarray:
        """f_n = n^2 f_1 [Hz]."""
        return self.mode_numbers**2 * self.f1

    @property
    def omegas(self) -> np.ndarray:
        """omega_n = (n pi / L)^2 sqrt(EI / m) [rad/s]."""
        return (self.mode_numbers * np.pi / self.span) ** 2 * np.sqrt(self.flexural_rigidity / self.mass_per_length)

    @property
    def modal_mass(self) -> float:
        """M_n = m L / 2 [kg] (same for every mode)."""
        return 0.5 * self.mass_per_length * self.span

    @property
    def modal_stiffness(self) -> np.ndarray:
        return self.modal_mass * self.omegas**2

    @property
    def modal_damping(self) -> np.ndarray:
        return 2.0 * self.damping_ratios * self.omegas * self.modal_mass

    @property
    def total_mass(self) -> float:
        return self.mass_per_length * self.span

    # ---- mode shapes ------------------------------------------------------------------------
    def _k(self) -> np.ndarray:
        return self.mode_numbers * np.pi / self.span

    def on_bridge(self, x) -> np.ndarray:
        x = np.asarray(x, float)
        return (x >= 0.0) & (x <= self.span)

    def shapes(self, x) -> np.ndarray:
        """phi_n(x), shape ``x.shape + (n_modes,)``; zero outside the span."""
        x = np.asarray(x, float)
        out = np.sin(x[..., None] * self._k())
        return np.where(self.on_bridge(x)[..., None], out, 0.0)

    def shape_slopes(self, x) -> np.ndarray:
        """d phi_n / dx; zero outside the span."""
        x = np.asarray(x, float)
        out = self._k() * np.cos(x[..., None] * self._k())
        return np.where(self.on_bridge(x)[..., None], out, 0.0)

    def shape_curvatures(self, x) -> np.ndarray:
        """d^2 phi_n / dx^2; zero outside the span."""
        x = np.asarray(x, float)
        out = -self._k() ** 2 * np.sin(x[..., None] * self._k())
        return np.where(self.on_bridge(x)[..., None], out, 0.0)

    # ---- static solutions (verification) ----------------------------------------------------
    def static_modal_response(self, x_load: float, force: float) -> np.ndarray:
        """Static q_n under a vertical force ``force`` (positive up) at ``x_load``."""
        return self.shapes(x_load) * force / self.modal_stiffness

    def static_deflection(self, x, x_load: float, force: float) -> np.ndarray:
        """Static deflection w(x) with the retained modes (positive up)."""
        return self.shapes(x) @ self.static_modal_response(x_load, force)

    def exact_midspan_static_deflection(self, force: float) -> float:
        """Closed form w(L/2) = P L^3 / (48 EI) for a force P (positive up) at midspan."""
        return force * self.span**3 / (48.0 * self.flexural_rigidity)

    def summary(self) -> dict:
        return {
            "span": self.span,
            "f1": self.f1,
            "mass_per_length": self.mass_per_length,
            "flexural_rigidity": self.flexural_rigidity,
            "n_modes": self.n_modes,
            "frequencies": self.frequencies.tolist(),
            "damping_ratios": self.damping_ratios.tolist(),
        }
