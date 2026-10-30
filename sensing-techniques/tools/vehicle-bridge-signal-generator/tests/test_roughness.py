"""ISO 8608 roughness: class table, exact evaluation methods, PSD and class of generated profiles."""
import numpy as np
import pytest

import vehicle_bridge_signal as vbs
from vehicle_bridge_signal.roughness import (CLASS_MEANS, RoughnessProfile, class_bounds, classify,
                                             displacement_psd, estimate_gd_n0, temporal_psd)


def test_class_table_cycles_per_metre_convention():
    assert CLASS_MEANS["A"] == pytest.approx(16e-6)
    assert CLASS_MEANS["B"] == pytest.approx(64e-6)
    assert CLASS_MEANS["C"] == pytest.approx(256e-6)
    assert CLASS_MEANS["D"] == pytest.approx(1024e-6)
    assert class_bounds("B") == pytest.approx((32e-6, 128e-6))
    assert class_bounds("D") == pytest.approx((512e-6, 2048e-6))
    assert classify(20e-6) == "A" and classify(100e-6) == "B" and classify(300e-6) == "C"
    assert classify(1000e-6) == "D" and classify(1e-9) == "A"
    assert displacement_psd(0.1, 64e-6) == pytest.approx(64e-6)
    assert displacement_psd(1.0, 64e-6) == pytest.approx(64e-8)
    # Temporal PSD at constant speed: G_t(f) = G_d(f/v)/v.
    assert temporal_psd(2.0, 10.0, 64e-6) == pytest.approx(displacement_psd(0.2, 64e-6) / 10.0)


@pytest.fixture(scope="module")
def profile():
    return RoughnessProfile.generate(np.random.default_rng(1), 64e-6, period=512.0)


@pytest.mark.parametrize("derivative", [0, 1, 2])
def test_chirp_z_equals_direct_sum(profile, derivative):
    """The O(N log N) evaluation equals the direct harmonic sum for an arbitrary grid."""
    x0, dx, n = -123.4567, 0.0137, 3000
    fast = profile.evaluate_uniform(x0, dx, n, derivative)
    direct = profile.evaluate(x0 + dx * np.arange(n), derivative)
    assert np.max(np.abs(fast - direct)) <= 1e-9 * np.max(np.abs(direct))


def test_chirp_z_equals_ifft_when_grid_matches_period(profile):
    """When n dx = period the sum is an inverse FFT of the harmonic coefficients."""
    n = 1 << 14
    dx = profile.period / n
    idx = np.round(profile.spatial_frequencies * profile.period).astype(int)
    spec = np.zeros(n, complex)
    spec[idx] = profile.amplitudes * np.exp(1j * profile.phases)
    via_ifft = np.real(np.fft.ifft(spec) * n)
    assert np.allclose(profile.evaluate_uniform(0.0, dx, n), via_ifft, atol=1e-12)


def test_analytical_derivatives_are_consistent(profile):
    x = np.linspace(10, 15, 10001)
    r = profile.evaluate(x)
    r1 = profile.evaluate(x, 1)
    r2 = profile.evaluate(x, 2)
    fd1 = np.gradient(r, x)
    fd2 = np.gradient(r1, x)
    assert np.max(np.abs(fd1 - r1)[5:-5]) < 2e-3 * np.max(np.abs(r1))
    assert np.max(np.abs(fd2 - r2)[5:-5]) < 2e-3 * np.max(np.abs(r2))


@pytest.mark.parametrize("letter", ["A", "B", "C", "D"])
def test_generated_psd_follows_iso_and_class(letter):
    """Validation 6: Welch PSD follows G_d(n) in the band and the estimated class is the requested one."""
    gd = CLASS_MEANS[letter]
    prof = RoughnessProfile.generate(np.random.default_rng(7), gd, period=4096.0)
    dx = 0.02
    r = prof.evaluate_uniform(0.0, dx, int(4096 / dx))
    est = estimate_gd_n0(r, dx, band=(0.05, 5.0), nperseg=8192)
    assert est["class"] == letter
    assert est["gd_n0"] == pytest.approx(gd, rel=0.1)
    assert est["waviness_fit"] == pytest.approx(2.0, abs=0.1)
    sel = (est["n"] > 0.05) & (est["n"] < 5)
    ratio = est["psd"][sel] / displacement_psd(est["n"][sel], gd)
    assert np.median(ratio) == pytest.approx(1.0, abs=0.15)


def test_waviness_other_than_two():
    prof = RoughnessProfile.generate(np.random.default_rng(3), 64e-6, waviness=3.0, period=4096.0)
    r = prof.evaluate_uniform(0.0, 0.02, 204800)
    est = estimate_gd_n0(r, 0.02, band=(0.05, 5.0), waviness=3.0, nperseg=8192)
    assert est["waviness_fit"] == pytest.approx(3.0, abs=0.15)


def test_smoothing_is_a_sinc_filter():
    rng_a, rng_b = np.random.default_rng(5), np.random.default_rng(5)
    raw = RoughnessProfile.generate(rng_a, 64e-6, period=256.0)
    smooth = RoughnessProfile.generate(rng_b, 64e-6, period=256.0, smoothing_length=0.25)
    assert np.array_equal(raw.phases, smooth.phases)
    assert np.allclose(smooth.amplitudes, raw.amplitudes * np.sinc(raw.spatial_frequencies * 0.25))
    # The moving average of the raw profile equals the smoothed profile.
    dx = 0.005
    x = np.arange(0, 50, dx)
    win = int(round(0.25 / dx)) + 1
    r = raw.evaluate(x)
    ma = np.convolve(r, np.ones(win) / win, mode="same")
    core = slice(win, -win)
    assert np.max(np.abs(ma[core] - smooth.evaluate(x)[core])) < 0.03 * np.max(np.abs(r))


def test_profile_does_not_change_with_speed(typical):
    """Parametric studies: changing the speed keeps the same road (same seed)."""
    a = vbs.simulate(vbs.with_overrides(typical, {"crossing.speed": 10.0}), 4, return_internal=True)
    b = vbs.simulate(vbs.with_overrides(typical, {"crossing.speed": 20.0}), 4, return_internal=True)
    pa, pb = a.internal["roughness_profile"], b.internal["roughness_profile"]
    assert pa.period == pb.period and np.array_equal(pa.phases, pb.phases)
    assert a.metadata["roughness_period_m"] >= 2 * np.ptp(a.internal["trajectory"].positions)


def test_period_is_extended_for_long_travel(typical):
    cfg = vbs.with_overrides(typical, {"roughness.profile_period_m": 50.0})
    res = vbs.simulate(cfg, 0)
    travelled = 15.0 * (10.24 + 4.0 + 0.5)
    assert res.metadata["roughness_period_m"] >= 2 * travelled
    assert any("period extended" in w for w in res.warnings)
