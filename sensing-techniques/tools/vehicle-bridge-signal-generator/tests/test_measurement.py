"""Smartphone measurement chain: noise density, anti-alias/aliasing, quantisation, mount, jitter."""
import numpy as np
import pytest
from scipy import signal

import vehicle_bridge_signal as vbs
from vehicle_bridge_signal import sensor
from vehicle_bridge_signal.spectral import peak_in_band, welch_psd
from vehicle_bridge_signal.vehicle import G

from conftest import quiet_overrides, rms


def test_white_noise_sigma_convention():
    """Validation 8: RMS of the simulated noise = ND g sqrt(fs/2) within 5 %."""
    nd, fs = 300.0, 1000.0
    x = sensor.white_noise(np.random.default_rng(0), 200_000, nd, fs)
    expected = nd * 1e-6 * G * np.sqrt(fs / 2)
    assert sensor.white_noise_sigma(nd, fs) == pytest.approx(expected)
    assert rms(x) == pytest.approx(expected, rel=0.05)
    f, p = welch_psd(x, fs, nperseg=4096)
    assert np.mean(p[(f > 10) & (f < 400)]) == pytest.approx((nd * 1e-6 * G) ** 2, rel=0.05)


def test_noise_in_simulation_has_the_requested_density(quiet):
    """In the full chain, the noise component has a flat PSD (ND g)^2 in the anti-alias pass band and
    the fs_sim noise has sigma = ND g sqrt(fs_sim/2)."""
    nd = 250.0
    cfg = vbs.with_overrides(quiet, {"sensor.white_noise.enabled": True, "sensor.white_noise.density_ug": nd,
                                     "sensor.anti_alias.enabled": True, "simulation.n_samples_out": 8192})
    res = vbs.simulate(cfg, 3, return_internal=True)
    assert rms(res.internal["noise_sim"]) == pytest.approx(nd * 1e-6 * G * np.sqrt(1000 / 2), rel=0.05)
    f, p = welch_psd(res.components["noise"], res.fs, nperseg=1024)
    band = (f > 2) & (f < 50)   # well inside the 80 Hz anti-alias cutoff
    assert np.mean(p[band]) == pytest.approx((nd * 1e-6 * G) ** 2, rel=0.1)
    assert res.metadata["noise_density_ug"] == nd


def _engine_tone(aa: bool):
    over = quiet_overrides()
    over.update({"engine.enabled": True, "engine.n_cylinders": 4, "engine.rpm_mean": 4500.0,
                 "engine.rpm_relative_std": 0.0, "engine.double_firing_amplitude": 0.0,
                 "engine.half_order_amplitude": 0.0, "engine.body_accel_rms": 0.05,
                 "sensor.anti_alias.enabled": aa})
    return vbs.simulate(vbs.load_preset("typical_car_smartphone", over), 0)


def test_aliasing_without_and_with_antialias_filter():
    """Validation 9: a 150 Hz firing tone (4 cyl, 4500 rpm) sampled at 200 Hz folds to |150 - 200| = 50 Hz
    without the anti-alias filter; with the filter it is attenuated by |H(150 Hz)|^2 (Butterworth of
    order 4 at 80 Hz applied forward and backward), about -48 dB: not a brick wall."""
    off, on = _engine_tone(False), _engine_tone(True)
    for res in (off, on):
        f, p = welch_psd(res.components["engine"], res.fs, nperseg=1024)
        assert peak_in_band(f, p, 1.0, 99.0) == pytest.approx(50.0, abs=f[1])
    _, h = signal.sosfreqz(sensor.antialias_sos(4, 80.0, 1000.0), worN=[150.0], fs=1000.0)
    ratio = rms(on.components["engine"]) / rms(off.components["engine"])
    assert ratio == pytest.approx(abs(h[0]) ** 2, rel=0.05)
    assert ratio < 0.01
    # Same physical vibration in both cases (only the measurement differs).
    assert off.metadata["engine_body_accel_rms"] == pytest.approx(on.metadata["engine_body_accel_rms"])


def test_antialias_filter_design():
    sos = sensor.antialias_sos(4, 80.0, 1000.0)
    w, h = signal.sosfreqz(sos, worN=[10.0, 80.0, 150.0], fs=1000.0)
    assert abs(h[0]) == pytest.approx(1.0, abs=1e-3)
    assert abs(h[1]) == pytest.approx(1 / np.sqrt(2), rel=1e-3)
    assert abs(h[2]) < 0.1


def test_quantisation_step_and_saturation():
    bits, rng_g = 16, 2.0
    step = 2 * rng_g * G / 2**bits
    assert step == pytest.approx(G / 16384)          # MPU-6500: 16384 LSB/g at +/-2 g
    x = np.linspace(-0.5, 0.5, 1001)
    q, nsat = sensor.quantize(x, bits, rng_g, include_gravity=False)
    assert nsat == 0 and np.max(np.abs(q - x)) <= step / 2 + 1e-15
    assert np.allclose(np.round(q / step), q / step)
    # With gravity the reading is a + g: positive saturation at +2g - g = 1 g of dynamic acceleration.
    # Dynamic range with gravity: [-3 g, +1 g).
    q, nsat = sensor.quantize(np.array([0.0, 1.5 * G, -2.5 * G, -3.5 * G]), bits, rng_g, include_gravity=True)
    assert nsat == 2 and q[1] < 1.0001 * G and q[2] == pytest.approx(-2.5 * G, abs=step)
    assert q[3] == pytest.approx(-3 * G, rel=1e-4)


def test_quantisation_only_noise_is_bounded(quiet):
    cfg = vbs.with_overrides(quiet, {"sensor.quantization.enabled": True, "roughness.enabled": True})
    res = vbs.simulate(cfg, 1)
    step = 4 * G / 2**16
    assert 0 < np.max(np.abs(res.components["noise"])) <= step / 2 + 1e-12


def test_mount_filter_response():
    sos = sensor.mount_sos(25.0, 0.1, 1000.0)
    _, h = signal.sosfreqz(sos, worN=[0.01, 25.0, 200.0], fs=1000.0)
    assert abs(h[0]) == pytest.approx(1.0, abs=1e-6)                      # rigid at low frequency
    assert abs(h[1]) == pytest.approx(np.sqrt(1 + 0.2**2) / 0.2, rel=1e-3)  # transmissibility at resonance
    assert abs(h[2]) < 0.3                                                 # isolation above
    with pytest.raises(ValueError):
        sensor.mount_sos(480.0, 0.1, 1000.0)


def test_jitter_changes_only_the_measurement(typical):
    base = vbs.simulate(typical, 2, return_internal=True)
    jit = vbs.simulate(vbs.with_overrides(typical, {"sensor.jitter.enabled": True, "sensor.jitter.std_s": 3e-4}), 2,
                       return_internal=True)
    for k in base.components:
        if k != "noise":
            assert np.array_equal(base.components[k], jit.components[k])
    delta = jit.internal["jitter_delta"]
    assert np.max(np.abs(delta)) <= 0.45 / 200 and delta.std() == pytest.approx(3e-4, rel=0.1)
    assert not np.array_equal(base.measured, jit.measured)
    assert jit.identity_error() < 1e-12


def test_pink_noise_slope_and_level():
    fs, n = 100.0, 2**18
    x = sensor.pink_noise(np.random.default_rng(4), n, 300.0, fs)
    f, p = welch_psd(x, fs, nperseg=8192)
    band = (f > 0.1) & (f < 20)
    slope = np.polyfit(np.log(f[band]), np.log(p[band]), 1)[0]
    assert slope == pytest.approx(-1.0, abs=0.1)
    level_1hz = np.exp(np.polyval(np.polyfit(np.log(f[band]), np.log(p[band]), 1), 0.0))
    assert level_1hz == pytest.approx((300e-6 * G) ** 2, rel=0.2)


def test_bias_and_drift_go_to_the_noise_component(quiet):
    cfg = vbs.with_overrides(quiet, {"sensor.bias.enabled": True, "sensor.bias.value_m_s2": 0.03,
                                     "sensor.drift.enabled": True, "sensor.drift.random_walk_m_s2_per_sqrt_s": 0.002})
    res = vbs.simulate(cfg, 0)
    assert np.mean(res.components["noise"]) == pytest.approx(0.03, abs=0.02)
    assert res.identity_error() < 1e-12


def test_decimation_alignment(quiet):
    """Without filters the output samples are the fs_sim samples at t = k / fs_out."""
    res = vbs.simulate(vbs.with_overrides(quiet, {"roughness.enabled": True}), 0, return_internal=True)
    it = res.internal
    assert res.t[0] == 0.0 and np.allclose(np.diff(res.t), 1 / 200)
    assert np.allclose(it["t_sim"][it["out_index"]], res.t)
    assert np.allclose(res.measured, it["true_sim"][it["out_index"]], rtol=0, atol=1e-13)
