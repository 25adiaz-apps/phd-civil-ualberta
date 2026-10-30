"""Bridge model: frequencies, static deflection (signs and modal mass), moving-load closed form."""
import numpy as np
import pytest

from vehicle_bridge_signal.bridge import (Bridge, count_modes, f1_from_flexural_rigidity, flexural_rigidity_from_f1,
                                          nominal_f1, resolve_f_max)
from vehicle_bridge_signal.vbi import InputColumn, make_trajectory, newmark_lti, solve_decoupled
from vehicle_bridge_signal.vehicle import sprung_mass


def test_frequencies_follow_n2_and_ei():
    """Validation 1: f_n = n^2 f1 and omega_n = (n pi/L)^2 sqrt(EI/m) with EI = m (2 L^2 f1/pi)^2."""
    L, f1, m = 27.0, 4.1, 9000.0
    b = Bridge(L, f1, m, np.full(6, 0.01))
    n = np.arange(1, 7)
    assert np.allclose(b.frequencies, n**2 * f1)
    ei = m * (2 * L**2 * f1 / np.pi) ** 2
    assert b.flexural_rigidity == pytest.approx(ei, rel=1e-12)
    assert np.allclose(b.omegas, (n * np.pi / L) ** 2 * np.sqrt(ei / m))
    assert np.allclose(b.omegas / (2 * np.pi), b.frequencies)
    assert b.modal_mass == pytest.approx(m * L / 2)
    # Yang et al. (2004) bridge: EI = 2.75e10 * 0.12, m = 4800, L = 25 -> f1 = 2.08 Hz
    assert f1_from_flexural_rigidity(25.0, 2.75e10 * 0.12, 4800.0) == pytest.approx(2.084, abs=1e-3)
    assert f1_from_flexural_rigidity(L, flexural_rigidity_from_f1(L, f1, m), m) == pytest.approx(f1)


def test_mode_count_and_nominal_f1():
    assert resolve_f_max("auto", 200.0) == 40.0
    assert resolve_f_max("auto", 50.0) == pytest.approx(22.5)
    assert count_modes(3.63, 40.0, 6) == 3       # 3.63, 14.5, 32.7 Hz
    assert count_modes(1.0, 40.0, 6) == 6        # capped by max_modes
    assert count_modes(50.0, 40.0, 6) == 1       # at least one mode
    for span, f in [(15, 8.03), (30, 3.63), (50, 2.05), (100, 0.75)]:
        assert nominal_f1(span) == pytest.approx(f)
    assert 3.63 > nominal_f1(40) > 2.05


def test_mode_shapes_vanish_outside_and_at_supports():
    b = Bridge(20.0, 5.0, 5000.0, [0.01, 0.01, 0.01])
    x = np.array([-1.0, 0.0, 10.0, 20.0, 21.0])
    phi = b.shapes(x)
    assert np.allclose(phi[[0, 1, 3, 4]], 0.0, atol=1e-12)
    assert phi[2, 0] == pytest.approx(1.0)
    assert np.all(b.shape_slopes(x)[[0, 4]] == 0)


def test_static_midspan_deflection_converges_to_closed_form():
    """Validation 2: w(L/2) -> P L^3/(48 EI) with enough modes; a downward load gives w < 0."""
    L, f1, m, W = 25.0, 2.084, 4800.0, 1200 * 9.80665
    exact = None
    errors = []
    for n_modes in (1, 3, 5, 15, 41):
        b = Bridge(L, f1, m, np.zeros(n_modes))
        w = b.static_deflection(L / 2, L / 2, -W)
        exact = b.exact_midspan_static_deflection(-W)
        errors.append(abs(w - exact) / abs(exact))
        assert np.sign(w) == -1
    assert exact == pytest.approx(-W * L**3 / (48 * b.flexural_rigidity))
    # Modal sum: w(L/2) = (2 P L^3 / (pi^4 EI)) sum_{n odd} n^-4 and sum_{n odd} n^-4 = pi^4/96, so the
    # truncation error with N modes is exactly the tail of that series.
    for n_modes, err in zip((1, 3, 5, 15, 41), errors):
        tail = sum(k**-4.0 for k in range(n_modes + 1, 20001) if k % 2) / (np.pi**4 / 96)
        assert err == pytest.approx(tail, rel=1e-6)
    assert errors[0] < 0.015            # one mode already gives 96/pi^4 = 98.55 % of the exact value
    assert errors[-1] < 3e-6
    assert all(e2 <= e1 for e1, e2 in zip(errors, errors[1:]))


def test_dynamic_solver_reaches_static_deflection():
    """The Newmark modal solver under a constant midspan load settles at the static deflection."""
    L, f1, m, P = 30.0, 3.0, 8000.0, -1.0e5
    b = Bridge(L, f1, m, np.full(5, 0.3))   # heavy damping to settle quickly
    dt, n = 1e-3, 6000
    F = np.repeat((b.shapes(L / 2) * P)[None, :, None], n, axis=0)
    X, _, _ = newmark_lti(np.eye(5) * b.modal_mass, np.diag(b.modal_damping), np.diag(b.modal_stiffness), F,
                          np.zeros((5, 1)), np.zeros((5, 1)), dt)
    w_end = X[-1, :, 0] @ b.shapes(L / 2)
    assert w_end == pytest.approx(b.static_deflection(L / 2, L / 2, P), rel=1e-6)
    assert w_end == pytest.approx(b.exact_midspan_static_deflection(P), rel=2e-3)


def test_moving_force_matches_closed_form():
    """Undamped beam under a moving constant force (Fryba): q_n = 2P/(mL) [sin Wt - (W/w) sin wt]/(w^2 - W^2)."""
    L, f1, m, v = 25.0, 2.084, 4800.0, 10.0
    b = Bridge(L, f1, m, np.zeros(4))
    veh = sprung_mass(1.0e-3, frequency=3.0)          # negligible mass: pure moving force
    t = np.arange(-0.5, 2.5, 1e-4)
    traj = make_trajectory(b, veh, t, v, 0.0)
    P = -1.0e4
    col = InputColumn("weight", axle_weights=True)
    # A moving force of magnitude |P| is obtained by scaling the response to the vehicle weight.
    resp = solve_decoupled(b, veh, traj, [col])
    scale = P / -veh.static_axle_loads[0]
    on = (t >= 0) & (t <= L / v)
    tt = t[on]
    w = b.omegas
    big_w = np.arange(1, 5) * np.pi * v / L
    q_exact = (2 * P / (m * L)) * (np.sin(np.outer(tt, big_w)) - (big_w / w) * np.sin(np.outer(tt, w))) / (w**2 - big_w**2)
    w_mid_exact = q_exact @ b.shapes(L / 2)
    w_mid = scale * resp.q[on, :, 0] @ b.shapes(L / 2)
    assert np.max(np.abs(w_mid - w_mid_exact)) / np.max(np.abs(w_mid_exact)) < 1e-3
    assert np.all(resp.q[t < 0] == 0)
