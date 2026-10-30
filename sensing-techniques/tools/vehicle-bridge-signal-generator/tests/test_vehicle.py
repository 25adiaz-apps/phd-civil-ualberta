"""Vehicle models: stiffness derivation, natural frequencies, FRF against a time simulation."""
import numpy as np
import pytest

from vehicle_bridge_signal.vbi import vehicle_response_to_contact
from vehicle_bridge_signal.vehicle import (G, VehicleInfeasibleError, half_car, quarter_car, quarter_car_stiffnesses,
                                           sprung_mass)


def test_sprung_mass_frequency():
    v = sprung_mass(1200.0, stiffness=5e5, damping=0.0)
    assert v.natural_frequencies()[0] == pytest.approx(np.sqrt(5e5 / 1200) / (2 * np.pi))
    assert v.natural_frequencies()[0] == pytest.approx(3.25, abs=0.01)   # Yang et al. (2004)
    assert v.static_axle_loads[0] == pytest.approx(1200 * G)


def test_quarter_car_stiffness_reproduces_targets():
    ms, mu, fb, fw = 400.0, 50.0, 1.3, 11.0
    ks, kt = quarter_car_stiffnesses(ms, mu, fb, fw)
    assert kt > ks > 0
    assert np.sqrt(ks * kt / (ks + kt) / ms) / (2 * np.pi) == pytest.approx(fb)   # ride rate
    assert np.sqrt((ks + kt) / mu) / (2 * np.pi) == pytest.approx(fw)             # wheel hop
    car = quarter_car(ms, mu, fb, fw, 0.3)
    f = car.natural_frequencies()
    # The coupled 2-DOF frequencies stay close to the targets and are stored in the metadata.
    assert f[0] == pytest.approx(fb, rel=0.03) and f[1] == pytest.approx(fw, rel=0.03)
    # Product of the eigenvalues equals k_s k_t / (m_s m_u) = w_body^2 w_wheel^2 exactly.
    assert np.prod((2 * np.pi * f) ** 2) == pytest.approx((2 * np.pi * fb) ** 2 * (2 * np.pi * fw) ** 2, rel=1e-9)
    z = car.modal_damping_ratios()
    assert 0 < z[0] < 0.5


def test_quarter_car_infeasible_targets_raise():
    with pytest.raises(VehicleInfeasibleError):
        quarter_car_stiffnesses(500.0, 30.0, 2.0, 8.0)   # (m_u/m_s)(f_w/f_b)^2 = 0.96 < 4


def test_frf_matches_time_simulation():
    """Steady-state body acceleration under a harmonic contact displacement equals |H(f)| A."""
    car = quarter_car(400.0, 50.0, 1.3, 11.0, 0.3)
    dt, f, amp = 1e-3, 2.0, 1e-3
    t = np.arange(0, 30, dt)
    u = amp * np.sin(2 * np.pi * f * t)
    du = amp * 2 * np.pi * f * np.cos(2 * np.pi * f * t)
    z, zd, zdd = vehicle_response_to_contact(car, dt, u[:, None, None], du[:, None, None], None,
                                             np.zeros((2, 1)), np.zeros((2, 1)))
    acc = zdd[:, 1, 0]
    tail = t > 20
    sim_amp = 0.5 * (acc[tail].max() - acc[tail].min())
    h = abs(car.frf_contact_to_sensor([f])[0])
    assert sim_amp == pytest.approx(h * amp, rel=2e-3)


def test_quasi_static_initial_state_is_rigid_translation():
    car = quarter_car(400.0, 50.0, 1.3, 11.0, 0.3)
    z0, zd0 = car.quasi_static_state(np.array([[0.01]]), np.array([[0.2]]))
    assert np.allclose(z0[:, 0], 0.01) and np.allclose(zd0[:, 0], 0.2)


def test_half_car_geometry_and_loads():
    hc = half_car(800.0, 1.0, 2.7, 0.45, 50.0, 1.4, 11.0, 0.3, sensor_position=0.5)
    assert hc.n_axles == 2 and hc.wheelbase == pytest.approx(2.7)
    assert hc.static_axle_loads.sum() == pytest.approx((800 + 100) * G)
    d1, d2 = 0.45 * 2.7, 0.55 * 2.7
    # Moment balance about the CG: front load * d1 = rear load * d2 for the body share.
    body_front = hc.static_axle_loads[0] / G - 50.0
    body_rear = hc.static_axle_loads[1] / G - 50.0
    assert body_front * d1 == pytest.approx(body_rear * d2)
    # Dynamic index 1: the half car splits exactly into a front and a rear quarter car, each with its
    # static share of the body mass.
    front = quarter_car(body_front, 50.0, 1.4, 11.0, 0.3).natural_frequencies()
    rear = quarter_car(body_rear, 50.0, 1.4, 11.0, 0.3).natural_frequencies()
    assert np.allclose(np.sort(hc.natural_frequencies()), np.sort(np.concatenate([front, rear])), rtol=1e-9)
    assert hc.sensor_vector.tolist() == [1.0, 0.5, 0.0, 0.0]
    # A rigid heave of both contact points moves every DOF equally (no pitch).
    z0, _ = hc.quasi_static_state(np.array([[0.01], [0.01]]), np.zeros((2, 1)))
    assert np.allclose(z0[[0, 2, 3], 0], 0.01) and abs(z0[1, 0]) < 1e-12
