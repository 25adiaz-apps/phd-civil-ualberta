"""Exact decomposition into components, superposition of inputs, analytical path checks, switches."""
import numpy as np
import pytest

import vehicle_bridge_signal as vbs

from conftest import rms

ALL_ON = {
    "wheel.enabled": True, "wheel.diameter_m": 0.65, "wheel.eccentricity_m": 3e-4,
    "ambient.enabled": True, "ambient.modal_accel_rms": 0.005,
    "mount.enabled": True, "mount.frequency_hz": 25.0, "mount.damping_ratio": 0.1,
}


@pytest.mark.parametrize("model", ["typical_car_smartphone", "half_car_typical"])
@pytest.mark.parametrize("interaction", ["coupled", "decoupled"])
def test_identity_measured_equals_sum_of_components(model, interaction):
    """Validation 7a: measured = sum(components) (< 1e-6 RMS, also after the float32 cast), and the path
    decomposition reproduces the coupled solution to round-off."""
    cfg = vbs.load_preset(model, {**ALL_ON, "simulation.interaction": interaction, "sensor.jitter.enabled": True})
    res = vbs.simulate(cfg, 5)
    assert res.identity_error() < 1e-6
    m32 = res.measured.astype(np.float32)
    s32 = np.sum([v.astype(np.float32) for v in res.components.values()], axis=0)
    assert np.max(np.abs(m32 - s32)) < 1e-6 * rms(res.measured)
    assert res.metadata["decomposition_residual"] < 1e-9


def test_superposition_of_per_input_simulations(quiet):
    """Validation 7b: sum of the simulations with one input at a time equals the full simulation.

    Every simulation contains the static weight, so full = weight + sum_j (only_j - weight). The random
    streams of each source are independent, so the realisations match.
    """
    base = vbs.with_overrides(quiet, {"sensor.anti_alias.enabled": True})
    inputs = {
        "roughness": {"roughness.enabled": True},
        "engine": {"engine.enabled": True},
        "wheel": {k: v for k, v in ALL_ON.items() if k.startswith("wheel")},
        "ambient": {k: v for k, v in ALL_ON.items() if k.startswith("ambient")},
    }
    weight = vbs.simulate(base, 9).measured
    full_over = {k: v for over in inputs.values() for k, v in over.items()}
    full = vbs.simulate(vbs.with_overrides(base, full_over), 9)
    total = weight.copy()
    for over in inputs.values():
        total += vbs.simulate(vbs.with_overrides(base, over), 9).measured - weight
    assert np.max(np.abs(total - full.measured)) < 1e-10 * rms(full.measured)
    # The labels add up in the same way.
    for key in full.components:
        if key == "noise":
            continue
        parts = vbs.simulate(vbs.with_overrides(base, inputs.get(key, {})), 9).components[key] \
            if key in inputs else None
        if parts is not None:
            assert np.allclose(parts, full.components[key], rtol=0, atol=1e-12 * rms(full.measured))


def test_roughness_path_equals_analytical_steady_state(quiet):
    """Body acceleration due to roughness = sum_i A_i |H(v n_i)| cos(...) with the vehicle FRF H
    (decoupled, so no bridge feedback; the warm-up removes the initial transient)."""
    cfg = vbs.with_overrides(quiet, {"roughness.enabled": True, "simulation.interaction": "decoupled",
                                     "simulation.fs_sim": 2000.0})
    res = vbs.simulate(cfg, 2, return_internal=True)
    it = res.internal
    prof, veh, traj = it["roughness_profile"], it["vehicle"], it["trajectory"]
    p = [k for k, _, _ in it["paths"]].index("roughness")
    idx = it["out_index"][::8]
    y_sim = it["y_path"][idx, p]
    v = traj.speed
    f = v * prof.spatial_frequencies
    h = veh.frf_contact_to_sensor(f)
    x = traj.positions[idx, 0]
    y_ana = np.real(np.exp(2j * np.pi * np.outer(x, prof.spatial_frequencies)) @ (prof.amplitudes * np.exp(1j * prof.phases) * h))
    assert np.max(np.abs(y_sim - y_ana)) < 0.01 * rms(y_ana)
    # And its RMS agrees with the expected value sum A_i^2 |H_i|^2 / 2 (finite-window scatter).
    assert rms(y_ana) == pytest.approx(np.sqrt(0.5 * np.sum((prof.amplitudes * np.abs(h)) ** 2)), rel=0.25)


def test_engine_path_equals_analytical_steady_state(quiet):
    cfg = vbs.with_overrides(quiet, {"engine.enabled": True, "engine.rpm_relative_std": 0.0, "engine.rpm_mean": 1800.0,
                                     "simulation.interaction": "decoupled"})
    res = vbs.simulate(cfg, 3, return_internal=True)
    it = res.internal
    veh = it["vehicle"]
    p = [k for k, _, _ in it["paths"]].index("engine")
    win = it["window"]
    t = it["t_sim"][win]
    y_sim = it["y_path"][win, p] * it["scales"]["engine"]
    force = it["engine_force_unit"][win] * it["scales"]["engine"]
    # Fit the force as a sum of sines at the engine orders, then predict the steady state with the FRF.
    orders = res.metadata["engine_orders"]
    freqs = np.array(orders) * 1800.0 / 60.0
    a = np.column_stack([fn(2 * np.pi * f * t) for f in freqs for fn in (np.cos, np.sin)])
    c, *_ = np.linalg.lstsq(a, force, rcond=None)
    phasor = c[0::2] - 1j * c[1::2]
    y_ana = np.real(np.exp(2j * np.pi * np.outer(t, freqs)) @ (phasor * veh.frf_engine_to_sensor(freqs)))
    assert np.max(np.abs(y_sim - y_ana)) < 0.01 * rms(y_ana)
    assert res.metadata["engine_body_accel_rms"] == pytest.approx(cfg["engine"]["body_accel_rms"], rel=1e-9)


def test_contact_point_components(quiet):
    """Contact acceleration: roughness part = v^2 r''(x) exactly; bridge part = d^2/dt^2 [phi(x(t)) q(t)]."""
    cfg = vbs.with_overrides(quiet, {"roughness.enabled": True})
    res = vbs.simulate(cfg, 1, return_internal=True)
    it = res.internal
    traj, prof = it["trajectory"], it["roughness_profile"]
    oi = it["out_index"]
    expected = traj.speed**2 * prof.evaluate(traj.positions[oi, 0], 2)
    assert np.allclose(res.contact["roughness"][0], expected, rtol=0, atol=1e-7 * rms(expected))
    # Bridge path: compare with a finite-difference second derivative of the contact displacement.
    resp = it["response"]
    q_total = resp.q @ np.array([it["scales"][n] for n in it["columns"]])   # all inputs (coupled feedback)
    u_b = np.einsum("si,si->s", traj.phi[:, 0, :], q_total)
    dt = traj.dt
    fd = (u_b[2:] - 2 * u_b[1:-1] + u_b[:-2]) / dt**2
    bridge_contact = sum(res.contact[k][0] for k in res.contact if k.startswith("bridge"))
    on = np.flatnonzero(res.on_bridge)[5:-5]
    fd_out = fd[oi[on] - 1]
    assert np.max(np.abs(bridge_contact[on] - fd_out)) < 0.02 * np.max(np.abs(fd_out))
    assert np.allclose(res.contact["total"], sum(v for k, v in res.contact.items() if k != "total"))
    assert not np.any(res.contact["engine"])


def test_bridge_labels_before_entry_and_unused_modes(quiet):
    res = vbs.simulate(vbs.with_overrides(quiet, {"roughness.enabled": True, "bridge.n_label_modes": 5}), 0)
    first = np.flatnonzero(res.on_bridge)[0]
    for k in [k for k in res.components if k.startswith("bridge")]:
        assert not np.any(res.components[k][:first])      # causality: nothing before the entry
    assert res.metadata["n_modes"] == 3
    assert not np.any(res.components["bridge_mode_4"]) and not np.any(res.components["bridge_mode_5"])
    assert np.any(res.components["bridge_mode_3"])


def test_higher_modes_label(quiet):
    res = vbs.simulate(vbs.with_overrides(quiet, {"bridge.n_label_modes": 1, "bridge.f1": 1.5,
                                                  "bridge.f1_factor": 1.0}), 0)
    assert res.metadata["n_modes"] >= 3
    assert np.any(res.components["bridge_higher_modes"])
    assert list(res.components)[:2] == ["bridge_mode_1", "bridge_higher_modes"]


@pytest.mark.parametrize("switch,key", [("roughness.enabled", "roughness"), ("engine.enabled", "engine"),
                                        ("wheel.enabled", "wheel")])
def test_disabled_disturbance_gives_exact_zero(typical, switch, key):
    """Validation 11: a disabled disturbance gives a component exactly equal to zero."""
    cfg = vbs.with_overrides(typical, {**ALL_ON, switch: False})
    res = vbs.simulate(cfg, 0)
    assert not np.any(res.components[key])
    assert not np.any(res.contact[key])
    on = vbs.simulate(vbs.with_overrides(typical, ALL_ON), 0)
    assert np.any(on.components[key])


def test_no_measurement_effects_gives_exact_zero_noise(typical):
    cfg = vbs.with_overrides(typical, {"sensor.white_noise.enabled": False, "sensor.quantization.enabled": False})
    assert not np.any(vbs.simulate(cfg, 0).components["noise"])


def test_ambient_switch_keeps_bridge_quiet_before_entry_and_changes_it_on_the_bridge(quiet):
    off = vbs.simulate(quiet, 0)
    on = vbs.simulate(vbs.with_overrides(quiet, {"ambient.enabled": True, "ambient.modal_accel_rms": 0.01}), 0)
    first = np.flatnonzero(on.on_bridge)[0]
    assert not np.any(on.components["bridge_mode_1"][:first])
    assert not np.allclose(on.components["bridge_mode_1"], off.components["bridge_mode_1"])
    # The bridge already vibrates before the vehicle enters.
    assert rms(on.bridge_midspan_acceleration[:first]) > 10 * rms(off.bridge_midspan_acceleration[:first])


def test_snr_controlled_reaches_the_targets(typical):
    targets = {"roughness": -12.0, "engine": 3.0, "wheel": -4.0, "sensor_noise": 16.0}
    over = {**ALL_ON, "simulation.amplitude_mode": "snr_controlled",
            **{f"snr_control.targets_db.{k}": v for k, v in targets.items()}}
    res = vbs.simulate(vbs.with_overrides(typical, over), 4)
    m = res.metadata
    for kind in ("roughness", "engine", "wheel"):
        assert m[f"ratio_db_bridge_to_{kind}"] == pytest.approx(targets[kind], abs=1e-6)
    assert m["ratio_db_bridge_to_white_noise"] == pytest.approx(16.0, abs=1e-6)
    assert res.identity_error() < 1e-6
    # The sensor stays physical (same density); the bridge is scaled up, so quantisation stays small.
    assert m["noise_density_ug"] == typical["sensor"]["white_noise"]["density_ug"]
    assert m["scale_bridge_inputs"] > 1
    assert m["ratio_db_bridge_to_noise"] == pytest.approx(16.0, abs=0.5)


def test_engine_rms_target_in_physical_mode(typical):
    res = vbs.simulate(vbs.with_overrides(typical, {"engine.body_accel_rms": 0.07}), 0)
    assert res.metadata["engine_body_accel_rms"] == pytest.approx(0.07, rel=1e-9)
