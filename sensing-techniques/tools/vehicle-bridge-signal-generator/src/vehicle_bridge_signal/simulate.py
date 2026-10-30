"""Single-signal simulation pipeline and exact decomposition into components.

Outline of :func:`simulate`
---------------------------
1. Build the bridge, the vehicle and the constant-speed trajectory (with a warm-up before the
   output window and a short tail after it).
2. Build one *input column* per excitation source: static weight, roughness, wheel eccentricity,
   engine force and ambient modal forces.
3. Solve the coupled (or decoupled) problem for all columns at once.
4. Path decomposition: given the solution, the vehicle is linear and time invariant with input
   u_c = sum_n phi_n(x_a) q_n + r(x_a) + e(t) plus the engine force. Each part is passed again
   through the vehicle model, giving the body response through each path (bridge mode n,
   roughness, wheel, engine). In the coupled case the q_n already contain the feedback.
5. Amplitudes: ``physical`` (engine scaled to its target body RMS) or ``snr_controlled`` (inputs
   scaled to target bridge-to-disturbance ratios on the on-bridge window).
6. Measurement chain. Clean components go through the same deterministic linear stages as the
   measurement (mount, anti-alias, decimation) but not through noise, jitter or quantisation, so
   ``measured = sum(components)`` holds exactly, with ``noise`` defined as the residual.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from . import sensor
from .bridge import Bridge, count_modes, f1_from_flexural_rigidity, nominal_f1, resolve_f_max
from .config import ConfigError, load_config, require_fixed
from .disturbances import ambient_modal_forces, wheel_eccentricity
from .engine import EngineOrders, engine_force, firing_frequency
from .roughness import RoughnessProfile, class_mean, classify
from .vbi import (InputColumn, contact_inputs_from_bridge, make_trajectory, solve_coupled, solve_decoupled,
                  vehicle_response_to_contact)
from .vehicle import G, VehicleModel, build_vehicle

__all__ = ["simulate", "SimulationResult", "SimulationError", "CrossingDoesNotFitError", "component_keys",
           "contact_keys", "STREAMS", "make_rng", "build_bridge", "crossing_window"]

# Independent random streams per source, so changing one parameter (e.g. the speed) keeps the
# realisation of the others. Never renumber: the indices are part of the reproducibility contract.
STREAMS = {"crossing": 0, "roughness": 1, "engine": 2, "wheel": 3, "ambient": 4, "white_noise": 5,
           "pink_noise": 6, "drift": 7, "jitter": 8}

INPUT_ORDER = ("weight", "roughness", "wheel", "engine", "ambient")


class SimulationError(RuntimeError):
    """The simulation produced an invalid result (non-finite values...)."""


class CrossingDoesNotFitError(ConfigError):
    """The crossing plus its margins does not fit in the output window."""


def make_rng(seed: int, stream: str) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence(entropy=int(seed), spawn_key=(STREAMS[stream],)))


def _check_seed(seed) -> int:
    if isinstance(seed, (bool, float)) or not isinstance(seed, (int, np.integer)):
        raise TypeError(f"seed must be a non-negative integer, got {seed!r}")
    if seed < 0:
        raise ValueError("seed must be non-negative")
    return int(seed)


def component_keys(n_label_modes: int) -> list[str]:
    """Label keys in the order they are stored."""
    return [f"bridge_mode_{k}" for k in range(1, n_label_modes + 1)] + \
        ["bridge_higher_modes", "roughness", "engine", "wheel", "noise"]


def contact_keys(n_label_modes: int) -> list[str]:
    return [f"bridge_mode_{k}" for k in range(1, n_label_modes + 1)] + \
        ["bridge_higher_modes", "roughness", "engine", "wheel", "total"]


# --------------------------------------------------------------------------------------------
# Scenario assembly
# --------------------------------------------------------------------------------------------
def build_bridge(bcfg: dict, fs_out: float) -> tuple[Bridge, list[str]]:
    """Bridge from a fixed ``bridge`` section. Returns the bridge and warnings."""
    warns = []
    span, m = float(bcfg["span"]), float(bcfg["mass_per_length"])
    if bcfg["flexural_rigidity"] is not None:
        f1 = f1_from_flexural_rigidity(span, float(bcfg["flexural_rigidity"]), m)
    elif bcfg["f1"] is None:
        table = bcfg["nominal_f1_table"]
        f1 = nominal_f1(span, table["span"], table["f1"]) * float(bcfg["f1_factor"])
    else:
        f1 = float(bcfg["f1"])
    f_max = resolve_f_max(bcfg["f_max"], fs_out)
    n = count_modes(f1, f_max, int(bcfg["max_modes"]))
    if f1 > f_max:
        warns.append(f"f1={f1:.3g} Hz is above f_max={f_max:.3g} Hz; mode 1 is kept anyway")
    zeta = bcfg["damping_ratio"]
    if isinstance(zeta, (list, tuple, np.ndarray)):
        if len(zeta) < n:
            raise ConfigError(f"bridge.damping_ratio has {len(zeta)} values but {n} modes are retained")
        zeta = np.asarray(zeta[:n], float)
    else:
        zeta = np.full(n, float(zeta))
    return Bridge(span, f1, m, zeta), warns


def crossing_window(window: float, duration: float, approach: float, exit_margin: float) -> tuple[float, float]:
    """Interval of admissible entry times, or raise CrossingDoesNotFitError."""
    lo, hi = approach, window - exit_margin - duration
    if hi < lo - 1e-12:
        raise CrossingDoesNotFitError(
            f"The crossing (duration {duration:.3f} s) plus margins ({approach:g} s + {exit_margin:g} s) does not fit "
            f"in the output window of {window:.3f} s; use a higher speed, a shorter span or more samples")
    return lo, max(lo, hi)


@dataclass
class _Grid:
    fs_sim: float
    fs_out: float
    factor: int
    n_out: int
    warm: int
    tail: int
    t: np.ndarray

    @property
    def n_steps(self) -> int:
        return self.t.size

    @property
    def window(self) -> float:
        return self.n_out / self.fs_out

    @property
    def out_index(self) -> np.ndarray:
        return self.warm + self.factor * np.arange(self.n_out)

    @property
    def win(self) -> slice:
        return slice(self.warm, self.warm + self.factor * self.n_out)


def _grid(scfg: dict) -> _Grid:
    fs_sim, fs_out = float(scfg["fs_sim"]), float(scfg["fs_out"])
    factor = int(round(fs_sim / fs_out))
    n_out = int(scfg["n_samples_out"])
    warm = int(math.ceil(float(scfg["warmup_s"]) * fs_out - 1e-9)) * factor
    tail = int(math.ceil(float(scfg["tail_s"]) * fs_out - 1e-9)) * factor
    n_steps = warm + factor * n_out + tail
    t = (np.arange(n_steps) - warm) / fs_sim
    return _Grid(fs_sim, fs_out, factor, n_out, warm, tail, t)


# --------------------------------------------------------------------------------------------
# Result container
# --------------------------------------------------------------------------------------------
@dataclass
class SimulationResult:
    """One simulated crossing (all arrays at fs_out, float64)."""

    t: np.ndarray
    fs: float
    measured: np.ndarray
    components: dict[str, np.ndarray]
    contact: dict[str, np.ndarray]                 # each (n_axles, n_out)
    on_bridge: np.ndarray                          # bool (n_out,): any axle on the bridge
    on_bridge_axles: np.ndarray                    # bool (n_axles, n_out)
    bridge_midspan_displacement: np.ndarray
    bridge_midspan_acceleration: np.ndarray
    metadata: dict[str, Any]
    warnings: list[str]
    config: dict = field(repr=False)
    seed: int = 0
    internal: dict | None = field(default=None, repr=False)

    @property
    def component_keys(self) -> list[str]:
        return list(self.components)

    def component_sum(self) -> np.ndarray:
        return np.sum([v for v in self.components.values()], axis=0)

    def identity_error(self) -> float:
        """max |measured - sum(components)| / RMS(measured)."""
        rms = np.sqrt(np.mean(self.measured**2))
        return float(np.max(np.abs(self.measured - self.component_sum())) / rms) if rms > 0 else 0.0

    def bridge_component(self) -> np.ndarray:
        return np.sum([v for k, v in self.components.items() if k.startswith("bridge_")], axis=0)


def _rms(x, mask=None) -> float:
    x = np.asarray(x, float)
    if mask is not None:
        x = x[..., mask]
    return float(np.sqrt(np.mean(x**2))) if x.size else 0.0


def _db(a: float, b: float) -> float:
    if b <= 0:
        return float("nan")
    if a <= 0:
        return float("-inf")
    return 20.0 * math.log10(a / b)


# --------------------------------------------------------------------------------------------
# Main entry point
# --------------------------------------------------------------------------------------------
def simulate(config: Any, seed: int = 0, *, return_internal: bool = False) -> SimulationResult:
    """Simulate one crossing with fully fixed parameters.

    Parameters
    ----------
    config
        Preset name, YAML path or mapping. Every parameter must be fixed (no distributions); use
        :func:`vehicle_bridge_signal.sample_scenario` to draw one from a random configuration.
    seed
        Non-negative integer. It drives the random realisations (roughness phases, engine rpm and
        phases, noise, ...) through independent streams; the same (config, seed) always gives the
        same arrays bit for bit.
    return_internal
        Also keep the high-rate (fs_sim) intermediate signals in ``result.internal`` (for
        teaching/verification; uses more memory).
    """
    cfg = load_config(config)
    require_fixed(cfg)
    seed = _check_seed(seed)
    scfg = cfg["simulation"]
    grid = _grid(scfg)
    warns: list[str] = []

    bridge, w = build_bridge(cfg["bridge"], grid.fs_out)
    warns += w
    vehicle = build_vehicle(cfg["vehicle"])
    speed = float(cfg["crossing"]["speed"])
    duration = (bridge.span + vehicle.wheelbase) / speed
    ccfg = cfg["crossing"]
    lo, hi = crossing_window(grid.window, duration, float(ccfg["approach_margin_s"]), float(ccfg["exit_margin_s"]))
    placement = ccfg["placement"]
    if placement == "center":
        entry = 0.5 * (lo + hi)
    elif placement == "random":
        entry = float(make_rng(seed, "crossing").uniform(lo, hi)) if hi > lo else lo
    else:
        entry = float(ccfg["entry_time_s"])
        if not lo - 1e-9 <= entry <= hi + 1e-9:
            raise CrossingDoesNotFitError(f"crossing.entry_time_s={entry} s must be in [{lo:.4g}, {hi:.4g}] s so the "
                                          "crossing and its margins fit in the window")
    traj = make_trajectory(bridge, vehicle, grid.t, speed, entry)
    v, dt, ns, na = speed, traj.dt, grid.n_steps, vehicle.n_axles
    meta: dict[str, Any] = {}

    # ---- input columns -----------------------------------------------------------------------
    columns = [InputColumn("weight", axle_weights=True)]
    contact_acc_input: dict[str, np.ndarray] = {}
    profile = None
    rcfg = cfg["roughness"]
    if rcfg["enabled"]:
        gd = float(rcfg["gd_n0"]) if rcfg["gd_n0"] is not None else class_mean(rcfg["class"])
        period = float(rcfg["profile_period_m"])
        travelled = float(traj.positions.max() - traj.positions.min())
        while period < 2.0 * travelled:
            period *= 2.0
        if period != float(rcfg["profile_period_m"]):
            warns.append(f"roughness profile period extended to {period:g} m (>= 2 x travelled distance)")
        rseed = seed if rcfg["seed"] is None else int(rcfg["seed"])
        smooth = float(rcfg["smoothing"]["length_m"]) if rcfg["smoothing"]["enabled"] else 0.0
        profile = RoughnessProfile.generate(make_rng(rseed, "roughness"), gd, float(rcfg["waviness"]),
                                            float(rcfg["n0"]), float(rcfg["n_min"]), float(rcfg["n_max"]),
                                            period, smooth)
        r = np.empty((ns, na))
        r1 = np.empty((ns, na))
        r2 = np.empty((ns, na))
        for a in range(na):
            x0 = float(traj.positions[0, a])
            r[:, a] = profile.evaluate_uniform(x0, v * dt, ns, 0)
            r1[:, a] = profile.evaluate_uniform(x0, v * dt, ns, 1)
            r2[:, a] = profile.evaluate_uniform(x0, v * dt, ns, 2)
        columns.append(InputColumn("roughness", contact_disp=r, contact_vel=v * r1))
        contact_acc_input["roughness"] = v * v * r2
        meta.update(roughness_gd_n0=gd, roughness_class=classify(gd), roughness_waviness=float(rcfg["waviness"]),
                    roughness_period_m=period, roughness_smoothing_m=smooth)
    wcfg = cfg["wheel"]
    if wcfg["enabled"] and float(wcfg["eccentricity_m"]) > 0:
        e, ed, edd, fw = wheel_eccentricity(make_rng(seed, "wheel"), grid.t, na, v, float(wcfg["diameter_m"]),
                                            float(wcfg["eccentricity_m"]))
        columns.append(InputColumn("wheel", contact_disp=e, contact_vel=ed))
        contact_acc_input["wheel"] = edd
        meta.update(wheel_frequency=fw)
    ecfg = cfg["engine"]
    engine_unit = None
    if ecfg["enabled"]:
        n_cyl = int(ecfg["n_cylinders"])
        orders = EngineOrders.from_config(n_cyl, float(ecfg["firing_amplitude"]),
                                          float(ecfg["double_firing_amplitude"]), float(ecfg["half_order_amplitude"]))
        if orders.orders:
            engine_unit, rpm = engine_force(make_rng(seed, "engine"), grid.t, orders, float(ecfg["rpm_mean"]),
                                            float(ecfg["rpm_relative_std"]), float(ecfg["rpm_correlation_time_s"]))
            f_top = float(rpm.max()) * max(orders.orders) / 60.0
            if f_top >= 0.5 * grid.fs_sim:
                raise ConfigError(f"Engine order frequency {f_top:.1f} Hz >= fs_sim/2; raise simulation.fs_sim")
            columns.append(InputColumn("engine", engine_force=engine_unit))
            meta.update(engine_rpm_min=float(rpm[grid.win].min()), engine_rpm_max=float(rpm[grid.win].max()),
                        engine_firing_frequency=float(firing_frequency(ecfg["rpm_mean"], n_cyl)),
                        engine_orders=list(orders.orders))
    acfg = cfg["ambient"]
    if acfg["enabled"] and float(acfg["modal_accel_rms"]) > 0:
        fa, q0, qd0, _ = ambient_modal_forces(make_rng(seed, "ambient"), bridge, ns, grid.fs_sim,
                                              float(acfg["modal_accel_rms"]))
        columns.append(InputColumn("ambient", modal_force=fa, q0=q0, qd0=qd0))
    names = [c.name for c in columns]

    # ---- solve -------------------------------------------------------------------------------
    gamma, beta = float(scfg["newmark_gamma"]), float(scfg["newmark_beta"])
    solver = solve_coupled if scfg["interaction"] == "coupled" else solve_decoupled
    resp = solver(bridge, vehicle, traj, columns, gamma, beta)

    # ---- path decomposition --------------------------------------------------------------------
    nb = bridge.n_modes
    paths: list[tuple[str, str, int]] = []   # (kind, input name, mode index or -1)
    for j, name in enumerate(names):
        for n in range(nb):
            if np.any(resp.q[:, n, j]) or np.any(resp.qd[:, n, j]):
                paths.append(("bridge", name, n))
    for kind in ("roughness", "wheel", "engine"):
        if kind in names:
            paths.append((kind, kind, -1))
    npaths = len(paths)
    u = np.zeros((ns, na, npaths))
    du = np.zeros((ns, na, npaths))
    fe = np.zeros((ns, npaths))
    z0 = np.zeros((vehicle.n_dof, npaths))
    zd0 = np.zeros((vehicle.n_dof, npaths))
    cacc = np.zeros((ns, na, npaths))   # contact-point acceleration per path
    for p, (kind, name, n) in enumerate(paths):
        if kind == "bridge":
            j = names.index(name)
            q, qd, qdd = resp.q[:, n, j], resp.qd[:, n, j], resp.qdd[:, n, j]
            u[:, :, p] = traj.phi[:, :, n] * q[:, None]
            du[:, :, p] = traj.phi[:, :, n] * qd[:, None] + v * traj.dphi[:, :, n] * q[:, None]
            cacc[:, :, p] = (traj.phi[:, :, n] * qdd[:, None] + 2 * v * traj.dphi[:, :, n] * qd[:, None]
                             + v * v * traj.ddphi[:, :, n] * q[:, None])
        elif kind == "engine":
            fe[:, p] = engine_unit
        else:
            col = columns[names.index(kind)]
            u[:, :, p], du[:, :, p] = col.contact_disp, col.contact_vel
            cacc[:, :, p] = contact_acc_input[kind]
            zq, zqd = vehicle.quasi_static_state(col.contact_disp[0][:, None], col.contact_vel[0][:, None])
            z0[:, p], zd0[:, p] = zq[:, 0], zqd[:, 0]
    _, _, zdd_p = vehicle_response_to_contact(vehicle, dt, u, du, fe, z0, zd0, gamma, beta)
    sv = vehicle.sensor_vector
    y_path = np.einsum("d,sdp->sp", sv, zdd_p)          # true body acceleration per path (fs_sim)
    y_col = np.einsum("d,sdc->sc", sv, resp.zdd)        # coupled body acceleration per input column

    # ---- deterministic linear chain (mount, anti-alias) at fs_sim -----------------------------
    mcfg, aacfg = cfg["mount"], cfg["sensor"]["anti_alias"]
    mount_sos = sensor.mount_sos(float(mcfg["frequency_hz"]), float(mcfg["damping_ratio"]), grid.fs_sim) \
        if mcfg["enabled"] else None
    aa_sos = sensor.antialias_sos(int(aacfg["order"]), float(aacfg["cutoff_ratio"]) * grid.fs_out, grid.fs_sim) \
        if aacfg["enabled"] else None
    zero_phase = bool(aacfg["zero_phase"])

    def phone_chain(x):   # stages 2 and 4 (time axis 0)
        if mount_sos is not None:
            x = sensor.apply_sos(mount_sos, x, zero_phase=False, axis=0)
        if aa_sos is not None:
            x = sensor.apply_sos(aa_sos, x, zero_phase, axis=0)
        return x

    def aa_only(x):
        return sensor.apply_sos(aa_sos, x, zero_phase, axis=0) if aa_sos is not None else x

    y_path_f = phone_chain(y_path)
    oi = grid.out_index
    y_path_out = y_path_f[oi]
    mask_out = traj.any_on[oi]
    mask_axles = traj.on_bridge[oi].T

    # ---- amplitudes ----------------------------------------------------------------------------
    scales = {name: 1.0 for name in names}
    snr_mode = scfg["amplitude_mode"] == "snr_controlled"
    if "engine" in names:
        rms_unit = _rms(y_col[grid.win, names.index("engine")])
        if rms_unit <= 0:
            raise SimulationError("The engine force produces no body acceleration")
        # physical: target body RMS; snr_controlled: only the starting point of the iteration.
        scales["engine"] = (0.01 if snr_mode else float(ecfg["body_accel_rms"])) / rms_unit
    bridge_paths = [p for p, (k, _, _) in enumerate(paths) if k == "bridge"]
    disturbance_path = {k: p for p, (k, _, _) in enumerate(paths) if k != "bridge"}

    def bridge_total(sc):
        out = np.zeros(grid.n_out)
        for p in bridge_paths:
            out += sc[paths[p][1]] * y_path_out[:, p]
        return out

    targets = cfg["snr_control"]["targets_db"]
    if snr_mode:
        converged = not disturbance_path
        for _ in range(int(cfg["snr_control"]["max_iterations"])):
            rms_b = _rms(bridge_total(scales), mask_out)
            change = 0.0
            for kind, p in disturbance_path.items():
                rms_d = _rms(y_path_out[:, p], mask_out)
                if rms_d <= 0:
                    continue
                new = rms_b / (10 ** (float(targets[kind]) / 20.0) * rms_d)
                change = max(change, abs(new - scales[kind]) / max(new, 1e-300))
                scales[kind] = new
            if change < 1e-10:
                converged = True
                break
        if not converged:
            warns.append("snr_controlled scaling did not converge; achieved ratios are stored in the metadata")

    # ---- noise (stage 3), anti-alias (4), decimation (5), jitter (6), quantisation (7) -----------
    scfg_s = cfg["sensor"]
    noise_sim = np.zeros(ns)
    nd = float(scfg_s["white_noise"]["density_ug"]) if scfg_s["white_noise"]["enabled"] and not snr_mode else 0.0
    white_unit = None
    if scfg_s["white_noise"]["enabled"]:
        white_unit = make_rng(seed, "white_noise").standard_normal(ns) * sensor.white_noise_sigma(1.0, grid.fs_sim)
        if snr_mode:
            rms_b = _rms(bridge_total(scales), mask_out)
            rms_n1 = _rms(aa_only(white_unit)[oi], mask_out)
            nd = rms_b / (10 ** (float(targets["sensor_noise"]) / 20.0) * rms_n1)
        noise_sim += nd * white_unit
    if scfg_s["pink_noise"]["enabled"]:
        noise_sim += sensor.pink_noise(make_rng(seed, "pink_noise"), ns,
                                       float(scfg_s["pink_noise"]["density_at_1hz_ug"]), grid.fs_sim)
    if scfg_s["bias"]["enabled"]:
        noise_sim += float(scfg_s["bias"]["value_m_s2"])
    if scfg_s["drift"]["enabled"]:
        noise_sim += sensor.random_walk(make_rng(seed, "drift"), ns,
                                        float(scfg_s["drift"]["random_walk_m_s2_per_sqrt_s"]), grid.fs_sim)

    col_scale = np.array([scales[nm] for nm in names])
    path_scale = np.array([scales[name] for _, name, _ in paths])
    # Stage 1, the true body acceleration, is the sum of the path responses. The coupled solution
    # gives the same signal independently; their difference is a numerical check of the decomposition.
    true_sim = y_path @ path_scale
    true_coupled = y_col @ col_scale
    post = slice(grid.warm, ns)
    ref_rms = _rms(true_coupled[grid.win])
    decomposition_residual = float(np.max(np.abs(true_sim[post] - true_coupled[post])) / ref_rms) if ref_rms > 0 \
        else 0.0
    if decomposition_residual > 1e-6:
        warns.append(f"path decomposition residual {decomposition_residual:.2e} (expected round-off)")
    t_out = grid.t[oi]
    jcfg = scfg_s["jitter"]
    jitter_on = bool(jcfg["enabled"]) and float(jcfg["std_s"]) > 0
    qcfg = scfg_s["quantization"]
    noise_stage_on = bool(np.any(noise_sim)) or jitter_on or bool(qcfg["enabled"])

    # ---- labels --------------------------------------------------------------------------------
    K = int(cfg["bridge"]["n_label_modes"])
    comps = {key: np.zeros(grid.n_out) for key in component_keys(K)}
    contact_f = aa_only(cacc.reshape(ns, na * npaths)).reshape(ns, na, npaths)[oi]   # (n_out, na, npaths)
    ckeys = contact_keys(K)
    contact = {key: np.zeros((na, grid.n_out)) for key in ckeys}
    for p, (kind, name, n) in enumerate(paths):
        key = (f"bridge_mode_{n + 1}" if n < K else "bridge_higher_modes") if kind == "bridge" else kind
        s = scales[name]
        comps[key] += s * y_path_out[:, p]
        contact[key] += s * contact_f[:, :, p].T
    clean_sum = np.sum([comps[k] for k in comps if k != "noise"], axis=0)
    contact["total"] = np.sum([contact[k] for k in ckeys if k != "total"], axis=0)

    # ---- measurement: noise (3), anti-alias (4), decimation (5), jitter (6), quantisation (7) ----
    noise_f = aa_only(noise_sim)
    measured_f = y_path_f @ path_scale + noise_f     # stages 2, 3, 4 at fs_sim
    jitter_delta = None
    n_sat = 0
    if not noise_stage_on:
        measured = clean_sum.copy()                  # no measurement effect: noise is exactly zero
    else:
        if jitter_on:
            measured, jitter_delta = sensor.jitter_resample(measured_f, grid.t, t_out, make_rng(seed, "jitter"),
                                                            float(jcfg["std_s"]), grid.fs_out)
        else:
            measured = measured_f[oi].copy()
        if qcfg["enabled"]:
            measured, n_sat = sensor.quantize(measured, int(qcfg["bits"]), float(qcfg["range_g"]),
                                              bool(qcfg["include_gravity"]))
            if n_sat:
                warns.append(f"sensor saturation in {n_sat} samples (range +/-{qcfg['range_g']} g)")
        comps["noise"] = measured - clean_sum

    # ---- bridge response, contact force, metadata -----------------------------------------------
    q_tot = resp.q @ col_scale
    qd_tot = resp.qd @ col_scale
    qdd_tot = resp.qdd @ col_scale
    phi_mid = bridge.shapes(0.5 * bridge.span)
    w_mid = aa_only(q_tot @ phi_mid)[oi]
    a_mid = aa_only(qdd_tot @ phi_mid)[oi]
    u_b, du_b = contact_inputs_from_bridge(traj, q_tot[:, :, None], qd_tot[:, :, None])
    u_tot, du_tot = u_b[:, :, 0], du_b[:, :, 0]
    for c in columns:
        if c.contact_disp is not None:
            u_tot += scales[c.name] * c.contact_disp
            du_tot += scales[c.name] * c.contact_vel
    z_tot = resp.z @ col_scale
    zd_tot = resp.zd @ col_scale
    post = slice(grid.warm, ns)
    ratios = []
    for a, dof in enumerate(vehicle.axle_dofs):
        compression = (vehicle.static_axle_loads[a] - vehicle.k_tyre[a] * (z_tot[post, dof] - u_tot[post, a])
                       - vehicle.c_tyre[a] * (zd_tot[post, dof] - du_tot[post, a]))
        ratios.append(float(compression.min() / vehicle.static_axle_loads[a]))
    min_ratio = min(ratios)
    if min_ratio < 0:
        warns.append(f"wheel lift-off: contact force reaches {min_ratio:.2f} x static load (linear model "
                     "keeps the wheel attached)")

    for arr_name, arr in (("measured", measured), *comps.items()):
        if not np.all(np.isfinite(arr)):
            raise SimulationError(f"Non-finite values in '{arr_name}' (seed={seed})")

    rms_bridge = _rms(np.sum([comps[k] for k in comps if k.startswith("bridge_")], axis=0), mask_out)
    vf = vehicle.natural_frequencies()
    meta.update(
        seed=seed, vehicle_model=vehicle.name, interaction=scfg["interaction"], amplitude_mode=scfg["amplitude_mode"],
        fs_out=grid.fs_out, fs_sim=grid.fs_sim, n_samples_out=grid.n_out,
        span=bridge.span, f1=bridge.f1, mass_per_length=bridge.mass_per_length,
        flexural_rigidity=bridge.flexural_rigidity, n_modes=nb, bridge_frequencies=bridge.frequencies.tolist(),
        bridge_damping_ratios=bridge.damping_ratios.tolist(),
        vehicle_mass=vehicle.total_mass, vehicle_frequencies=vf.tolist(),
        vehicle_damping_ratios=vehicle.modal_damping_ratios().tolist(), n_axles=na, wheelbase=vehicle.wheelbase,
        mass_ratio=vehicle.total_mass / bridge.total_mass, speed=v, entry_time=entry, exit_time=entry + duration,
        crossing_duration=duration, driving_frequency=v / (2 * bridge.span),
        roughness_enabled=profile is not None, engine_enabled="engine" in names, wheel_enabled="wheel" in names,
        ambient_enabled="ambient" in names, mount_enabled=mount_sos is not None, antialias_enabled=aa_sos is not None,
        white_noise_enabled=white_unit is not None, pink_noise_enabled=bool(scfg_s["pink_noise"]["enabled"]),
        bias_enabled=bool(scfg_s["bias"]["enabled"]), drift_enabled=bool(scfg_s["drift"]["enabled"]),
        jitter_enabled=jitter_delta is not None, quantization_enabled=bool(qcfg["enabled"]),
        noise_density_ug=nd, n_saturated=n_sat, min_contact_force_ratio=min_ratio, wheel_lift_off=min_ratio < 0,
        rms_bridge_on=rms_bridge, decomposition_residual=decomposition_residual,
    )
    for key, val in vehicle.parameters.items():
        meta[f"vehicle_{key}"] = val
    if "engine" in names:
        meta["engine_n_cylinders"] = int(ecfg["n_cylinders"])
        meta["engine_rpm_mean"] = float(ecfg["rpm_mean"])
        meta["engine_body_accel_rms"] = _rms(y_col[grid.win, names.index("engine")]) * scales["engine"]
        meta["engine_force_scale"] = scales["engine"]
    for kind in ("roughness", "engine", "wheel"):
        rms_d = _rms(comps[kind], mask_out)
        meta[f"rms_{kind}_on"] = rms_d
        meta[f"ratio_db_bridge_to_{kind}"] = _db(rms_bridge, rms_d) if kind in names else float("nan")
        meta[f"scale_{kind}"] = scales.get(kind, 0.0)
    rms_noise = _rms(comps["noise"], mask_out)
    meta["rms_noise_on"] = rms_noise
    meta["ratio_db_bridge_to_noise"] = _db(rms_bridge, rms_noise)
    # White sensor noise alone (no quantisation, pink noise, bias or drift): the quantity that the
    # sensor_noise target of snr_controlled acts on.
    rms_white = _rms(nd * aa_only(white_unit)[oi], mask_out) if white_unit is not None else 0.0
    meta["rms_white_noise_on"] = rms_white
    meta["ratio_db_bridge_to_white_noise"] = _db(rms_bridge, rms_white)
    for name in names:   # bridge component by source (useful to see the feedback of each input)
        b = np.zeros(grid.n_out)
        for p in bridge_paths:
            if paths[p][1] == name:
                b += scales[name] * y_path_out[:, p]
        meta[f"rms_bridge_from_{name}_on"] = _rms(b, mask_out)
    if snr_mode:
        for kind in [k for k in disturbance_path] + (["sensor_noise"] if white_unit is not None else []):
            meta[f"snr_target_db_{kind}"] = float(targets[kind])
    meta["warnings"] = list(warns)

    internal = None
    if return_internal:
        internal = dict(t_sim=grid.t, out_index=oi, window=grid.win, trajectory=traj, bridge=bridge, vehicle=vehicle,
                        response=resp, columns=names, scales=scales, paths=paths, y_path=y_path, y_col=y_col,
                        true_sim=true_sim, true_coupled_sim=true_coupled, clean_filtered_sim=y_path_f @ path_scale, noise_sim=noise_sim, noise_filtered_sim=noise_f,
                        measured_filtered_sim=measured_f, roughness_profile=profile, jitter_delta=jitter_delta,
                        mount_sos=mount_sos, antialias_sos=aa_sos, contact_true_sim=cacc,
                        engine_force_unit=engine_unit)
    return SimulationResult(t=t_out, fs=grid.fs_out, measured=measured, components=comps, contact=contact,
                            on_bridge=mask_out, on_bridge_axles=mask_axles, bridge_midspan_displacement=w_mid,
                            bridge_midspan_acceleration=a_mid, metadata=meta, warnings=warns, config=cfg, seed=seed,
                            internal=internal)
