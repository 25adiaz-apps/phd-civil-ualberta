"""Vehicle-bridge interaction solvers (Newmark average acceleration).

Both solvers work with several *input columns* at once. Each column is one excitation source
(static weight, roughness, wheel eccentricity, engine force, ambient forces). Given the trajectory,
the problem is linear in the inputs, so the response to every input is computed in the same pass
(same effective matrices, a right-hand side with one column per input) and the physical response
is a weighted sum of the columns.

Coupled formulation (state X = [q (bridge modes); z (vehicle DOFs)]). For each axle ``a`` on the
bridge, with phi = phi(x_a), phi' = d phi/dx (x_a), k = k_t,a, c = c_t,a, j = vehicle DOF of the axle::

    contact displacement  u_a  = phi^T q + r_a
    contact velocity      u_a' = phi^T q' + v phi'^T q + r_a'          (exact chain rule)

    bridge rows:   M_b q'' + C_b q' + K_b q
                   + k phi phi^T q + c v phi phi'^T q + c phi phi^T q' - k phi z_j - c phi z_j'
                 = -phi W_a - phi (k r_a + c r_a') + f_ambient
    vehicle rows:  M_v z'' + C_v z' + K_v z - b_j (k phi^T q + c phi^T q' + c v phi'^T q)
                 = b_j (k r_a + c r_a') + e F_e

The matrices change with time; they are assembled for every on-bridge step and the system is
solved implicitly (gamma = 1/2, beta = 1/4). The decoupled solver (Algorithm 2 of Talebi-Kalaleh
& Mei 2026) applies the static axle weights as moving loads, then drives the vehicle with the
resulting deflection plus roughness.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .bridge import Bridge
from .vehicle import VehicleModel

__all__ = ["Trajectory", "InputColumn", "Response", "make_trajectory", "newmark_lti", "solve_coupled",
           "solve_decoupled", "vehicle_response_to_contact", "contact_inputs_from_bridge"]


@dataclass
class Trajectory:
    """Constant-speed trajectory of the axles on the simulation time grid."""

    t: np.ndarray                 # [s] (n_steps,), t = 0 at the start of the output window
    dt: float
    speed: float
    positions: np.ndarray         # [m] (n_steps, n_axles), road coordinate (bridge from 0 to L)
    on_bridge: np.ndarray         # bool (n_steps, n_axles)
    phi: np.ndarray = field(repr=False)    # (n_steps, n_axles, n_modes)
    dphi: np.ndarray = field(repr=False)   # d phi / dx
    ddphi: np.ndarray = field(repr=False)  # d^2 phi / dx^2

    @property
    def n_steps(self) -> int:
        return self.t.size

    @property
    def any_on(self) -> np.ndarray:
        return self.on_bridge.any(axis=1)


def make_trajectory(bridge: Bridge, vehicle: VehicleModel, t: np.ndarray, speed: float,
                    entry_time: float) -> Trajectory:
    """Front axle at x = 0 when t = entry_time; the other axles follow at their offsets."""
    t = np.asarray(t, float)
    front = speed * (t - entry_time)
    pos = front[:, None] + vehicle.axle_offsets[None, :]
    return Trajectory(t=t, dt=float(t[1] - t[0]), speed=float(speed), positions=pos,
                      on_bridge=bridge.on_bridge(pos), phi=bridge.shapes(pos), dphi=bridge.shape_slopes(pos),
                      ddphi=bridge.shape_curvatures(pos))


@dataclass
class InputColumn:
    """One excitation source. Unused fields stay ``None``."""

    name: str
    contact_disp: np.ndarray | None = None   # (n_steps, n_axles) road-type contact displacement [m]
    contact_vel: np.ndarray | None = None    # (n_steps, n_axles) its time derivative [m/s]
    engine_force: np.ndarray | None = None   # (n_steps,) [N]
    modal_force: np.ndarray | None = None    # (n_steps, n_modes) [N]
    axle_weights: bool = False               # static axle weights as forces on the bridge
    q0: np.ndarray | None = None             # initial bridge state (n_modes,)
    qd0: np.ndarray | None = None


@dataclass
class Response:
    """Response to every input column: arrays (n_steps, n_dof, n_cols)."""

    names: list[str]
    q: np.ndarray = field(repr=False)
    qd: np.ndarray = field(repr=False)
    qdd: np.ndarray = field(repr=False)
    z: np.ndarray = field(repr=False)
    zd: np.ndarray = field(repr=False)
    zdd: np.ndarray = field(repr=False)

    def col(self, name: str) -> int:
        return self.names.index(name)


# --------------------------------------------------------------------------------------------
# Generic linear time-invariant Newmark
# --------------------------------------------------------------------------------------------
def newmark_lti(M, C, K, F, X0, V0, dt, gamma=0.5, beta=0.25):
    """Newmark integration of M X'' + C X' + K X = F for constant matrices.

    ``F`` has shape (n_steps, n, n_cols); ``X0``, ``V0`` have shape (n, n_cols).
    Returns X, V, A with shape (n_steps, n, n_cols).
    """
    n_steps = F.shape[0]
    X = np.empty_like(F)
    V = np.empty_like(F)
    A = np.empty_like(F)
    X[0], V[0] = X0, V0
    A[0] = np.linalg.solve(M, F[0] - C @ V0 - K @ X0)
    keff_inv = np.linalg.inv(M + gamma * dt * C + beta * dt**2 * K)
    c1, c2 = (0.5 - beta) * dt**2, (1.0 - gamma) * dt
    b1, b2 = beta * dt**2, gamma * dt
    x, v, a = X[0].copy(), V[0].copy(), A[0].copy()
    for k in range(1, n_steps):
        xp = x + dt * v + c1 * a
        vp = v + c2 * a
        a = keff_inv @ (F[k] - C @ vp - K @ xp)
        x = xp + b1 * a
        v = vp + b2 * a
        X[k], V[k], A[k] = x, v, a
    return X, V, A


# --------------------------------------------------------------------------------------------
# Right-hand sides
# --------------------------------------------------------------------------------------------
def _bridge_rhs(traj: Trajectory, vehicle: VehicleModel, col: InputColumn, n_modes: int) -> np.ndarray:
    f = np.zeros((traj.n_steps, n_modes))
    if col.axle_weights:
        f -= np.einsum("a,sai->si", vehicle.static_axle_loads, traj.phi)
    if col.contact_disp is not None:
        load = col.contact_disp * vehicle.k_tyre + col.contact_vel * vehicle.c_tyre  # (n_steps, n_axles)
        f -= np.einsum("sa,sai->si", load, traj.phi)
    if col.modal_force is not None:
        f += col.modal_force
    return f


def _vehicle_rhs(vehicle: VehicleModel, n_steps: int, contact_disp=None, contact_vel=None,
                 engine_force=None) -> np.ndarray:
    f = np.zeros((n_steps, vehicle.n_dof))
    bk, bc = vehicle.contact_matrices()
    if contact_disp is not None:
        f += contact_disp @ bk.T + contact_vel @ bc.T
    if engine_force is not None:
        f += np.outer(engine_force, vehicle.engine_vector)
    return f


def _vehicle_initial_state(vehicle: VehicleModel, col: InputColumn) -> tuple[np.ndarray, np.ndarray]:
    if col.contact_disp is None:
        return np.zeros(vehicle.n_dof), np.zeros(vehicle.n_dof)
    z0, zd0 = vehicle.quasi_static_state(col.contact_disp[0][:, None], col.contact_vel[0][:, None])
    return z0[:, 0], zd0[:, 0]


# --------------------------------------------------------------------------------------------
# Coupled solver
# --------------------------------------------------------------------------------------------
def _coupling_matrices(bridge: Bridge, vehicle: VehicleModel, traj: Trajectory, steps: np.ndarray):
    """K(t), C(t) of the coupled system at the given step indices, shape (len(steps), n, n)."""
    nb, nv = bridge.n_modes, vehicle.n_dof
    n = nb + nv
    s = steps.size
    k0 = np.zeros((n, n))
    c0 = np.zeros((n, n))
    k0[:nb, :nb] = np.diag(bridge.modal_stiffness)
    c0[:nb, :nb] = np.diag(bridge.modal_damping)
    k0[nb:, nb:] = vehicle.K
    c0[nb:, nb:] = vehicle.C
    kt = np.broadcast_to(k0, (s, n, n)).copy()
    ct = np.broadcast_to(c0, (s, n, n)).copy()
    phi, dphi = traj.phi[steps], traj.dphi[steps]  # (s, A, N), zero for axles off the bridge
    v = traj.speed
    for a, dof in enumerate(vehicle.axle_dofs):
        k, c = vehicle.k_tyre[a], vehicle.c_tyre[a]
        p, dp = phi[:, a, :], dphi[:, a, :]
        j = nb + dof
        kt[:, :nb, :nb] += k * np.einsum("si,sj->sij", p, p) + c * v * np.einsum("si,sj->sij", p, dp)
        ct[:, :nb, :nb] += c * np.einsum("si,sj->sij", p, p)
        kt[:, :nb, j] -= k * p
        ct[:, :nb, j] -= c * p
        kt[:, j, :nb] -= k * p + c * v * dp
        ct[:, j, :nb] -= c * p
    return k0, c0, kt, ct


def solve_coupled(bridge: Bridge, vehicle: VehicleModel, traj: Trajectory, columns: list[InputColumn],
                  gamma: float = 0.5, beta: float = 0.25) -> Response:
    """Monolithic coupled VBI solution for every input column."""
    nb, nv = bridge.n_modes, vehicle.n_dof
    n, ncol, ns, dt = nb + nv, len(columns), traj.n_steps, traj.dt
    F = np.zeros((ns, n, ncol))
    X0 = np.zeros((n, ncol))
    V0 = np.zeros((n, ncol))
    for c, col in enumerate(columns):
        F[:, :nb, c] = _bridge_rhs(traj, vehicle, col, nb)
        F[:, nb:, c] = _vehicle_rhs(vehicle, ns, col.contact_disp, col.contact_vel, col.engine_force)
        if col.q0 is not None:
            X0[:nb, c], V0[:nb, c] = col.q0, col.qd0
        X0[nb:, c], V0[nb:, c] = _vehicle_initial_state(vehicle, col)

    M = np.zeros((n, n))
    M[:nb, :nb] = np.eye(nb) * bridge.modal_mass
    M[nb:, nb:] = vehicle.M
    on_steps = np.flatnonzero(traj.any_on)
    k0, c0, kt, ct = _coupling_matrices(bridge, vehicle, traj, on_steps)
    slot = np.full(ns, -1)
    slot[on_steps] = np.arange(on_steps.size)
    kinv_on = np.linalg.inv(M + gamma * dt * ct + beta * dt**2 * kt) if on_steps.size else None
    kinv_off = np.linalg.inv(M + gamma * dt * c0 + beta * dt**2 * k0)

    X = np.empty((ns, n, ncol))
    V = np.empty((ns, n, ncol))
    A = np.empty((ns, n, ncol))
    X[0], V[0] = X0, V0
    kk, cc = (kt[slot[0]], ct[slot[0]]) if slot[0] >= 0 else (k0, c0)
    A[0] = np.linalg.solve(M, F[0] - cc @ V0 - kk @ X0)
    c1, c2, b1, b2 = (0.5 - beta) * dt**2, (1.0 - gamma) * dt, beta * dt**2, gamma * dt
    x, v, a = X[0].copy(), V[0].copy(), A[0].copy()
    for k in range(1, ns):
        xp = x + dt * v + c1 * a
        vp = v + c2 * a
        s = slot[k]
        if s >= 0:
            a = kinv_on[s] @ (F[k] - ct[s] @ vp - kt[s] @ xp)
        else:
            a = kinv_off @ (F[k] - c0 @ vp - k0 @ xp)
        x = xp + b1 * a
        v = vp + b2 * a
        X[k], V[k], A[k] = x, v, a
    names = [c.name for c in columns]
    return Response(names, X[:, :nb], V[:, :nb], A[:, :nb], X[:, nb:], V[:, nb:], A[:, nb:])


# --------------------------------------------------------------------------------------------
# Decoupled solver and vehicle-only response
# --------------------------------------------------------------------------------------------
def contact_inputs_from_bridge(traj: Trajectory, q: np.ndarray, qd: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Bridge part of the contact displacement and velocity: (n_steps, n_axles, n_cols).

    u = phi^T q,  u' = phi^T q' + v phi'^T q  (phi = 0 off the bridge).
    """
    u = np.einsum("sai,sic->sac", traj.phi, q)
    du = np.einsum("sai,sic->sac", traj.phi, qd) + traj.speed * np.einsum("sai,sic->sac", traj.dphi, q)
    return u, du


def vehicle_response_to_contact(vehicle: VehicleModel, dt: float, u: np.ndarray, du: np.ndarray,
                                engine_force: np.ndarray | None, z0: np.ndarray, zd0: np.ndarray,
                                gamma: float = 0.5, beta: float = 0.25):
    """Vehicle-only response to prescribed contact inputs ``u``, ``du`` (n_steps, n_axles, n_cols)
    and engine forces (n_steps, n_cols). Returns z, z', z'' with shape (n_steps, n_dof, n_cols)."""
    bk, bc = vehicle.contact_matrices()
    F = np.einsum("da,sac->sdc", bk, u) + np.einsum("da,sac->sdc", bc, du)
    if engine_force is not None:
        F += vehicle.engine_vector[None, :, None] * engine_force[:, None, :]
    return newmark_lti(vehicle.M, vehicle.C, vehicle.K, F, z0, zd0, dt, gamma, beta)


def solve_decoupled(bridge: Bridge, vehicle: VehicleModel, traj: Trajectory, columns: list[InputColumn],
                    gamma: float = 0.5, beta: float = 0.25) -> Response:
    """Moving-load bridge solution, then vehicle driven by deflection + roughness (no feedback)."""
    nb, ns, ncol = bridge.n_modes, traj.n_steps, len(columns)
    Fb = np.zeros((ns, nb, ncol))
    q0 = np.zeros((nb, ncol))
    qd0 = np.zeros((nb, ncol))
    for c, col in enumerate(columns):
        if col.axle_weights:
            Fb[:, :, c] -= np.einsum("a,sai->si", vehicle.static_axle_loads, traj.phi)
        if col.modal_force is not None:
            Fb[:, :, c] += col.modal_force
        if col.q0 is not None:
            q0[:, c], qd0[:, c] = col.q0, col.qd0
    Mb = np.eye(nb) * bridge.modal_mass
    q, qd, qdd = newmark_lti(Mb, np.diag(bridge.modal_damping), np.diag(bridge.modal_stiffness), Fb, q0, qd0,
                             traj.dt, gamma, beta)
    u, du = contact_inputs_from_bridge(traj, q, qd)
    fe = np.zeros((ns, ncol))
    z0 = np.zeros((vehicle.n_dof, ncol))
    zd0 = np.zeros((vehicle.n_dof, ncol))
    for c, col in enumerate(columns):
        if col.contact_disp is not None:
            u[:, :, c] += col.contact_disp
            du[:, :, c] += col.contact_vel
        if col.engine_force is not None:
            fe[:, c] = col.engine_force
        z0[:, c], zd0[:, c] = _vehicle_initial_state(vehicle, col)
    z, zd, zdd = vehicle_response_to_contact(vehicle, traj.dt, u, du, fe, z0, zd0, gamma, beta)
    return Response([c.name for c in columns], q, qd, qdd, z, zd, zdd)
