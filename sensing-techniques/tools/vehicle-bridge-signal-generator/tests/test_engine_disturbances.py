"""Engine orders, rpm process, wheel eccentricity and ambient vibration."""
import numpy as np
import pytest
from scipy.signal import hilbert

from vehicle_bridge_signal.bridge import Bridge
from vehicle_bridge_signal.disturbances import ambient_modal_forces, wheel_frequency
from vehicle_bridge_signal.engine import EngineOrders, engine_force, firing_frequency, smooth_gaussian_process
from vehicle_bridge_signal.spectral import peak_in_band, welch_psd
from vehicle_bridge_signal.vbi import newmark_lti


def test_firing_frequency():
    assert firing_frequency(3000, 4) == pytest.approx(100.0)       # second order of a 4-cylinder engine
    assert firing_frequency(600, 4) == pytest.approx(20.0)         # 600-6000 rpm -> 20-200 Hz (Burdzik)
    assert firing_frequency(6000, 4) == pytest.approx(200.0)
    assert EngineOrders.from_config(6, 1.0, 0.3, 0.1).orders == (3.0, 6.0, 0.5)
    assert EngineOrders.from_config(4, 1.0, 0.0, 0.0).orders == (2.0,)


def test_constant_rpm_tone_at_firing_frequency():
    fs = 1000.0
    t = np.arange(0, 20, 1 / fs)
    force, rpm = engine_force(np.random.default_rng(0), t, EngineOrders.from_config(4, 1.0, 0.0, 0.0), 1500.0, 0.0, 2.0)
    assert np.all(rpm == 1500.0)
    f, p = welch_psd(force, fs, nperseg=4096)
    assert peak_in_band(f, p, 1, 499) == pytest.approx(50.0, abs=fs / 4096)


def test_phase_is_integral_of_instantaneous_frequency():
    """With a varying rpm the instantaneous frequency is o * rpm(t)/60 (not 2 pi f(t) t)."""
    fs = 2000.0
    t = np.arange(0, 30, 1 / fs)
    orders = EngineOrders.from_config(4, 1.0, 0.0, 0.0)
    force, rpm = engine_force(np.random.default_rng(1), t, orders, 2000.0, 0.1, 1.0)
    phase = np.unwrap(np.angle(hilbert(force)))
    f_inst = np.gradient(phase, t) / (2 * np.pi)
    core = slice(4000, -4000)
    expected = 2.0 * rpm / 60.0
    assert np.median(np.abs(f_inst[core] - expected[core]) / expected[core]) < 2e-3
    assert rpm.std() > 50  # it really varies


def test_smooth_process_statistics():
    t = np.arange(0, 1500, 0.1)
    g = smooth_gaussian_process(np.random.default_rng(2), t, 2.0)
    assert g.std() == pytest.approx(1.0, abs=0.1)
    lag = int(2.0 / 0.1)
    rho = np.corrcoef(g[:-lag], g[lag:])[0, 1]
    assert rho == pytest.approx(np.exp(-1.0), abs=0.08)   # covariance exp(-(dt/tau)^2) at dt = tau


def test_wheel_frequency_example():
    # 835 mm tyre at 64 km/h -> about 7 Hz (specification example)
    assert wheel_frequency(64 / 3.6, 0.835) == pytest.approx(6.78, abs=0.01)


def test_ambient_vibration_is_stationary_at_the_requested_level():
    b = Bridge(30.0, 3.0, 8000.0, [0.02, 0.02])
    fs, n = 1000.0, 200_000
    forces, q0, qd0, sigma_q = ambient_modal_forces(np.random.default_rng(3), b, n, fs, 0.005)
    X, _, _ = newmark_lti(np.eye(2) * b.modal_mass, np.diag(b.modal_damping), np.diag(b.modal_stiffness),
                          forces[:, :, None], q0[:, None], qd0[:, None], 1 / fs)
    q = X[:, :, 0]
    # Stationary from the start: the first and the second half have the same variance.
    for k in range(2):
        first, second = q[: n // 2, k].std(), q[n // 2:, k].std()
        assert first == pytest.approx(sigma_q[k], rel=0.2)
        assert second == pytest.approx(sigma_q[k], rel=0.2)
    assert np.allclose(sigma_q * b.omegas**2, 0.005)


def test_ambient_requires_damping():
    with pytest.raises(ValueError):
        ambient_modal_forces(np.random.default_rng(0), Bridge(30.0, 3.0, 8000.0, [0.0]), 10, 1000.0, 0.01)
