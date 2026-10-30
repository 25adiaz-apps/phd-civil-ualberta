"""Linear vehicle models in a common formulation.

Every model is written as::

    M z'' + C z' + K z = sum_a b_a (k_t,a u_a + c_t,a u_a') + e F_e(t)

* ``z``: vehicle DOFs measured from the static equilibrium on a rigid, smooth road (positive up),
  so gravity cancels out.
* ``u_a``: vertical displacement of the contact point of axle ``a`` (bridge deflection + roughness
  + wheel eccentricity), positive up.
* ``b_a``: unit vector of the DOF connected to the road through the tyre (k_t,a, c_t,a); ``K`` and
  ``C`` already include the tyre spring and damper.
* ``e``: distribution of the engine force on the DOFs.

Vertical force applied by axle ``a`` on the bridge (positive up, Talebi-Kalaleh & Mei 2026)::

    P_a = -W_a + k_t,a (z_dof(a) - u_a) + c_t,a (z_dof(a)' - u_a')

The contact (compressive) force is ``-P_a``; wheel lift-off happens when it becomes negative.
The phone measures ``s^T z''`` (``s`` = sensor vector).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import linalg

from .config import ConfigError

__all__ = ["VehicleModel", "VehicleInfeasibleError", "sprung_mass", "quarter_car", "half_car",
           "quarter_car_stiffnesses", "build_vehicle", "G"]

G = 9.80665  # [m/s^2] standard gravity


class VehicleInfeasibleError(ConfigError):
    """The requested frequency targets cannot be produced by positive stiffnesses."""


@dataclass(frozen=True)
class VehicleModel:
    name: str
    dof_names: tuple[str, ...]
    M: np.ndarray = field(repr=False)
    C: np.ndarray = field(repr=False)
    K: np.ndarray = field(repr=False)
    axle_dofs: tuple[int, ...]
    axle_offsets: np.ndarray = field(repr=False)   # [m] axle position relative to the front axle (<= 0)
    k_tyre: np.ndarray = field(repr=False)
    c_tyre: np.ndarray = field(repr=False)
    static_axle_loads: np.ndarray = field(repr=False)  # [N] weight carried by each axle (> 0)
    sensor_vector: np.ndarray = field(repr=False)
    engine_vector: np.ndarray = field(repr=False)
    parameters: dict = field(default_factory=dict)

    def __post_init__(self):
        n = len(self.dof_names)
        for name in ("M", "C", "K"):
            if getattr(self, name).shape != (n, n):
                raise ValueError(f"{name} must be {n}x{n}")
        if not np.all(np.linalg.eigvalsh(self.M) > 0):
            raise ValueError("Vehicle mass matrix must be positive definite")
        if np.any(self.k_tyre <= 0):
            raise ValueError("Tyre (contact) stiffness must be positive")

    @property
    def n_dof(self) -> int:
        return len(self.dof_names)

    @property
    def n_axles(self) -> int:
        return len(self.axle_dofs)

    @property
    def total_mass(self) -> float:
        return float(self.static_axle_loads.sum() / G)

    @property
    def wheelbase(self) -> float:
        return float(-self.axle_offsets.min())

    def contact_matrices(self) -> tuple[np.ndarray, np.ndarray]:
        """B_k, B_c (n_dof x n_axles): generalised force per unit contact displacement / velocity."""
        bk = np.zeros((self.n_dof, self.n_axles))
        bc = np.zeros((self.n_dof, self.n_axles))
        for a, dof in enumerate(self.axle_dofs):
            bk[dof, a] = self.k_tyre[a]
            bc[dof, a] = self.c_tyre[a]
        return bk, bc

    def natural_frequencies(self) -> np.ndarray:
        """Undamped natural frequencies on a rigid road [Hz], ascending."""
        w2 = linalg.eigh(self.K, self.M, eigvals_only=True)
        return np.sqrt(np.clip(w2, 0.0, None)) / (2 * np.pi)

    def modal_damping_ratios(self) -> np.ndarray:
        """Damping ratio of each mode from the complex eigenvalues (sorted like natural_frequencies)."""
        n = self.n_dof
        a = np.block([[np.zeros((n, n)), np.eye(n)],
                      [-np.linalg.solve(self.M, self.K), -np.linalg.solve(self.M, self.C)]])
        lam = np.linalg.eigvals(a)
        lam = lam[np.imag(lam) >= 0]
        wn = np.abs(lam)
        order = np.argsort(wn)
        lam, wn = lam[order][:n], wn[order][:n]
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(wn > 0, -np.real(lam) / wn, 0.0)

    def _receptance(self, omega: float) -> np.ndarray:
        return np.linalg.inv(self.K - omega**2 * self.M + 1j * omega * self.C)

    def frf_contact_to_sensor(self, freqs, axle: int = 0) -> np.ndarray:
        """Sensor acceleration per unit contact displacement of ``axle`` (complex, [1/s^2])."""
        out = np.empty(len(np.atleast_1d(freqs)), complex)
        dof = self.axle_dofs[axle]
        for i, f in enumerate(np.atleast_1d(freqs)):
            w = 2 * np.pi * f
            force = np.zeros(self.n_dof, complex)
            force[dof] = self.k_tyre[axle] + 1j * w * self.c_tyre[axle]
            out[i] = -(w**2) * self.sensor_vector @ (self._receptance(w) @ force)
        return out

    def frf_engine_to_sensor(self, freqs) -> np.ndarray:
        """Sensor acceleration per unit engine force (complex, [m/s^2/N])."""
        out = np.empty(len(np.atleast_1d(freqs)), complex)
        for i, f in enumerate(np.atleast_1d(freqs)):
            w = 2 * np.pi * f
            out[i] = -(w**2) * self.sensor_vector @ (self._receptance(w) @ self.engine_vector)
        return out

    def quasi_static_state(self, u, du) -> tuple[np.ndarray, np.ndarray]:
        """State that follows the contact input quasi-statically: z = K^-1 B_k u, z' = K^-1 B_k u'.

        ``u`` and ``du`` have shape ``(n_axles, n_cols)``. Used as initial condition so that the
        simulation does not start with a compressed suspension.
        """
        bk, _ = self.contact_matrices()
        return np.linalg.solve(self.K, bk @ u), np.linalg.solve(self.K, bk @ du)

    def summary(self) -> dict:
        out = {"model": self.name, "total_mass": self.total_mass, "n_axles": self.n_axles}
        out.update(self.parameters)
        out["natural_frequencies"] = self.natural_frequencies().tolist()
        out["modal_damping_ratios"] = self.modal_damping_ratios().tolist()
        return out


# --------------------------------------------------------------------------------------------
# Builders
# --------------------------------------------------------------------------------------------
def quarter_car_stiffnesses(body_mass: float, unsprung_mass: float, body_frequency: float,
                            wheel_frequency: float) -> tuple[float, float]:
    """Suspension and tyre stiffness from target ride and wheel-hop frequencies.

    Uses the standard definitions (Gillespie 1992, *Fundamentals of Vehicle Dynamics*, ch. 5)::

        ride rate   RR = k_s k_t / (k_s + k_t) = m_s (2 pi f_body)^2
        wheel hop   k_s + k_t = m_u (2 pi f_wheel)^2

    so k_s and k_t are the roots of z^2 - a z + a b = 0 with a = m_u w_wheel^2, b = m_s w_body^2;
    the tyre takes the larger root. A real solution needs a >= 4 b.
    """
    a = unsprung_mass * (2 * np.pi * wheel_frequency) ** 2
    b = body_mass * (2 * np.pi * body_frequency) ** 2
    disc = a * a - 4 * a * b
    if disc < 0:
        raise VehicleInfeasibleError(
            f"No positive stiffnesses give f_body={body_frequency:.3g} Hz and f_wheel={wheel_frequency:.3g} Hz with "
            f"m_s={body_mass:.4g} kg, m_u={unsprung_mass:.4g} kg: it needs (m_u/m_s)(f_wheel/f_body)^2 >= 4, "
            f"got {(unsprung_mass / body_mass) * (wheel_frequency / body_frequency) ** 2:.3g}")
    k_t = 0.5 * (a + np.sqrt(disc))
    k_s = a - k_t
    return float(k_s), float(k_t)


def sprung_mass(mass: float, frequency: float | None = None, damping_ratio: float = 0.0,
                stiffness: float | None = None, damping: float | None = None) -> VehicleModel:
    """1-DOF moving sprung mass (Yang et al. 2004). The spring/damper is the contact element."""
    k = float(stiffness) if stiffness is not None else mass * (2 * np.pi * frequency) ** 2
    c = float(damping) if damping is not None else 2 * damping_ratio * np.sqrt(k * mass)
    return VehicleModel(
        name="sprung_mass", dof_names=("y_v",),
        M=np.array([[mass]], float), C=np.array([[c]], float), K=np.array([[k]], float),
        axle_dofs=(0,), axle_offsets=np.zeros(1), k_tyre=np.array([k]), c_tyre=np.array([c]),
        static_axle_loads=np.array([mass * G]), sensor_vector=np.array([1.0]), engine_vector=np.array([1.0]),
        parameters={"mass": mass, "stiffness": k, "damping": c,
                    "frequency_target": float(np.sqrt(k / mass) / (2 * np.pi))},
    )


def quarter_car(body_mass: float, unsprung_mass: float, body_frequency: float | None = None,
                wheel_frequency: float | None = None, suspension_damping_ratio: float = 0.3,
                tyre_damping: float = 0.0, suspension_stiffness: float | None = None,
                tyre_stiffness: float | None = None, suspension_damping: float | None = None) -> VehicleModel:
    """2-DOF quarter car, DOFs ``[y_u, y_s]`` (equation 3 of Talebi-Kalaleh & Mei 2026)."""
    if (suspension_stiffness is None) != (tyre_stiffness is None):
        raise ConfigError("Give both suspension_stiffness and tyre_stiffness, or neither")
    if suspension_stiffness is None:
        if body_frequency is None or wheel_frequency is None:
            raise ConfigError("quarter_car needs body_frequency and wheel_frequency (or both stiffnesses)")
        k_s, k_t = quarter_car_stiffnesses(body_mass, unsprung_mass, body_frequency, wheel_frequency)
    else:
        k_s, k_t = float(suspension_stiffness), float(tyre_stiffness)
    c_s = float(suspension_damping) if suspension_damping is not None else \
        2 * suspension_damping_ratio * np.sqrt(k_s * body_mass)
    c_t = float(tyre_damping)
    m = np.diag([unsprung_mass, body_mass]).astype(float)
    c = np.array([[c_t + c_s, -c_s], [-c_s, c_s]])
    k = np.array([[k_t + k_s, -k_s], [-k_s, k_s]])
    return VehicleModel(
        name="quarter_car", dof_names=("y_u", "y_s"), M=m, C=c, K=k,
        axle_dofs=(0,), axle_offsets=np.zeros(1), k_tyre=np.array([k_t]), c_tyre=np.array([c_t]),
        static_axle_loads=np.array([(body_mass + unsprung_mass) * G]),
        sensor_vector=np.array([0.0, 1.0]), engine_vector=np.array([0.0, 1.0]),
        parameters={"body_mass": body_mass, "unsprung_mass": unsprung_mass,
                    "body_frequency_target": body_frequency, "wheel_frequency_target": wheel_frequency,
                    "suspension_stiffness": k_s, "tyre_stiffness": k_t,
                    "suspension_damping": c_s, "tyre_damping": c_t},
    )


def half_car(body_mass: float, dynamic_index: float, wheelbase: float, cg_to_front_ratio: float,
             unsprung_mass: float, body_frequency: float, wheel_frequency: float,
             suspension_damping_ratio: float = 0.3, tyre_damping: float = 0.0, sensor_position: float = 0.0,
             engine_position: float | None = None) -> VehicleModel:
    """4-DOF half car (bounce y_v, pitch theta, front axle y_u1, rear axle y_u2).

    Geometry: front axle at d1 = ratio * d ahead of the CG, rear axle at d2 = d - d1 behind it.
    ``theta`` is positive nose-up, so the vertical displacement of a body point at distance x ahead
    of the CG is ``y_v + x theta`` and the phone (at x_p) measures ``y_v'' + x_p theta''``.
    Each axle gets the quarter-car stiffness derivation with its static share of the body mass.
    """
    d = float(wheelbase)
    d1 = cg_to_front_ratio * d
    d2 = d - d1
    shares = (body_mass * d2 / d, body_mass * d1 / d)  # body mass carried by front / rear suspension
    ks, kt, cs = [], [], []
    for share in shares:
        k_s, k_t = quarter_car_stiffnesses(share, unsprung_mass, body_frequency, wheel_frequency)
        ks.append(k_s)
        kt.append(k_t)
        cs.append(2 * suspension_damping_ratio * np.sqrt(k_s * share))
    inertia = dynamic_index * body_mass * d1 * d2
    g1 = np.array([1.0, d1, -1.0, 0.0])   # front suspension deflection
    g2 = np.array([1.0, -d2, 0.0, -1.0])  # rear suspension deflection
    e3, e4 = np.eye(4)[2], np.eye(4)[3]
    k = ks[0] * np.outer(g1, g1) + ks[1] * np.outer(g2, g2) + kt[0] * np.outer(e3, e3) + kt[1] * np.outer(e4, e4)
    c = cs[0] * np.outer(g1, g1) + cs[1] * np.outer(g2, g2) + tyre_damping * (np.outer(e3, e3) + np.outer(e4, e4))
    x_e = d1 if engine_position is None else float(engine_position)
    loads = np.array([shares[0] + unsprung_mass, shares[1] + unsprung_mass]) * G
    return VehicleModel(
        name="half_car", dof_names=("y_v", "theta", "y_u1", "y_u2"),
        M=np.diag([body_mass, inertia, unsprung_mass, unsprung_mass]).astype(float), C=c, K=k,
        axle_dofs=(2, 3), axle_offsets=np.array([0.0, -d]), k_tyre=np.array(kt),
        c_tyre=np.array([tyre_damping, tyre_damping], float), static_axle_loads=loads,
        sensor_vector=np.array([1.0, float(sensor_position), 0.0, 0.0]),
        engine_vector=np.array([1.0, x_e, 0.0, 0.0]),
        parameters={"body_mass": body_mass, "pitch_inertia": inertia, "wheelbase": d, "cg_to_front": d1,
                    "unsprung_mass": unsprung_mass, "body_frequency_target": body_frequency,
                    "wheel_frequency_target": wheel_frequency, "suspension_stiffness_front": ks[0],
                    "suspension_stiffness_rear": ks[1], "tyre_stiffness_front": kt[0], "tyre_stiffness_rear": kt[1],
                    "suspension_damping_front": cs[0], "suspension_damping_rear": cs[1],
                    "tyre_damping": tyre_damping, "sensor_position": float(sensor_position),
                    "engine_position": x_e},
    )


def build_vehicle(vehicle_cfg: dict) -> VehicleModel:
    """Build the vehicle selected by ``vehicle_cfg['model']`` from a fixed configuration section."""
    model = vehicle_cfg["model"]
    p = vehicle_cfg[model]
    if model == "sprung_mass":
        return sprung_mass(p["mass"], None if p["stiffness"] is not None else p["frequency"],
                           0.0 if p["damping"] is not None else p["damping_ratio"], p["stiffness"], p["damping"])
    if model == "quarter_car":
        direct = p["suspension_stiffness"] is not None
        return quarter_car(p["body_mass"], p["unsprung_mass"], None if direct else p["body_frequency"],
                           None if direct else p["wheel_frequency"],
                           0.0 if p["suspension_damping"] is not None else p["suspension_damping_ratio"],
                           p["tyre_damping"], p["suspension_stiffness"], p["tyre_stiffness"], p["suspension_damping"])
    if model == "half_car":
        return half_car(p["body_mass"], p["dynamic_index"], p["wheelbase"], p["cg_to_front_ratio"],
                        p["unsprung_mass"], p["body_frequency"], p["wheel_frequency"],
                        p["suspension_damping_ratio"], p["tyre_damping"], p["sensor_position"],
                        p["engine_position"])
    raise ConfigError(f"Unknown vehicle model '{model}'")
