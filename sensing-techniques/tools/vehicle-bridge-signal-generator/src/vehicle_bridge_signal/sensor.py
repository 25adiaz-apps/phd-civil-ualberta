"""Smartphone measurement chain.

Order (each stage can be switched off):

1. true acceleration at the sensor point, at fs_sim;
2. phone-mount filter (base-excited 1 DOF, absolute acceleration transmissibility);
3. sensor noise: white (noise density ND), pink (1/f), constant bias, random-walk drift;
4. Butterworth anti-alias filter (zero phase with ``sosfiltfilt`` or causal);
5. decimation to fs_out;
6. sampling jitter (resampling at slightly irregular instants);
7. quantisation and saturation.

Noise-density convention: ND is the one-sided amplitude spectral density in ug/sqrt(Hz). White
noise sampled at fs has variance ND^2 g^2 fs / 2, i.e. sigma = ND g sqrt(fs / 2), and a flat
one-sided PSD equal to (ND g)^2 [(m/s^2)^2/Hz].
"""
from __future__ import annotations

import numpy as np
from scipy import signal
from scipy.interpolate import make_interp_spline

from .vehicle import G

__all__ = ["UG", "white_noise_sigma", "white_noise", "pink_noise", "random_walk", "mount_sos", "antialias_sos",
           "apply_sos", "decimate", "jitter_resample", "quantize"]

UG = 1e-6 * G  # [m/s^2] one micro-g


def white_noise_sigma(density_ug: float, fs: float) -> float:
    """sigma = ND g sqrt(fs/2) [m/s^2]."""
    return density_ug * UG * np.sqrt(fs / 2.0)


def white_noise(rng: np.random.Generator, n: int, density_ug: float, fs: float) -> np.ndarray:
    return rng.standard_normal(n) * white_noise_sigma(density_ug, fs)


def pink_noise(rng: np.random.Generator, n: int, density_at_1hz_ug: float, fs: float) -> np.ndarray:
    """Gaussian 1/f noise with one-sided PSD (ND1 g)^2 (1 Hz / f) for f >= fs/n (zero mean).

    Built in the frequency domain: for an rfft of length n, E|X_k|^2 = S(f_k) fs n / 2 gives a
    one-sided PSD S(f).
    """
    f = np.fft.rfftfreq(n, 1.0 / fs)
    s = np.zeros_like(f)
    s[1:] = (density_at_1hz_ug * UG) ** 2 / f[1:]
    scale = np.sqrt(s * fs * n / 2.0)
    spec = (rng.standard_normal(f.size) + 1j * rng.standard_normal(f.size)) / np.sqrt(2.0) * scale
    if n % 2 == 0:
        spec[-1] = spec[-1].real * np.sqrt(2.0)  # the Nyquist bin is real
    spec[0] = 0.0
    return np.fft.irfft(spec, n)


def random_walk(rng: np.random.Generator, n: int, intensity: float, fs: float) -> np.ndarray:
    """Random walk starting at 0 with increments N(0, intensity^2 dt) [intensity in m/s^2/sqrt(s)]."""
    steps = rng.standard_normal(n) * intensity * np.sqrt(1.0 / fs)
    steps[0] = 0.0
    return np.cumsum(steps)


def mount_sos(frequency: float, damping_ratio: float, fs: float) -> np.ndarray:
    """Absolute-acceleration transmissibility of a base-excited 1 DOF holder.

    H(s) = (2 zeta w s + w^2) / (s^2 + 2 zeta w s + w^2), discretised with the bilinear transform
    pre-warped at the holder frequency.
    """
    if not 0 < frequency < 0.45 * fs:
        raise ValueError(f"Mount frequency {frequency} Hz must be below 0.45 fs_sim = {0.45 * fs} Hz")
    # Pre-warp so that the discrete resonance falls exactly at ``frequency``.
    wp = 2 * fs * np.tan(np.pi * frequency / fs)
    bz, az = signal.bilinear([2 * damping_ratio * wp, wp**2], [1.0, 2 * damping_ratio * wp, wp**2], fs=fs)
    return signal.tf2sos(bz, az)


def antialias_sos(order: int, cutoff_hz: float, fs: float) -> np.ndarray:
    return signal.butter(order, cutoff_hz, btype="low", fs=fs, output="sos")


def apply_sos(sos: np.ndarray, x: np.ndarray, zero_phase: bool, axis: int = -1) -> np.ndarray:
    """Apply a filter along ``axis`` (zero phase: ``sosfiltfilt`` with odd padding)."""
    if zero_phase:
        return signal.sosfiltfilt(sos, x, axis=axis)
    return signal.sosfilt(sos, x, axis=axis)


def decimate(x: np.ndarray, factor: int, start: int, n_out: int, axis: int = -1) -> np.ndarray:
    """Samples ``start, start + factor, ...`` (``n_out`` of them) along ``axis`` (no filtering here)."""
    idx = start + factor * np.arange(n_out)
    return np.take(x, idx, axis=axis)


def jitter_resample(x_sim: np.ndarray, t_sim: np.ndarray, t_nominal: np.ndarray, rng: np.random.Generator,
                    std: float, fs_out: float) -> tuple[np.ndarray, np.ndarray]:
    """Sample a band-limited fs_sim signal at t_nominal + delta, delta ~ N(0, std) truncated at 0.45/fs_out.

    Cubic-spline interpolation of the (already anti-aliased) signal. Returns the samples and delta.
    """
    delta = np.clip(rng.standard_normal(t_nominal.size) * std, -0.45 / fs_out, 0.45 / fs_out)
    lo = max(0, np.searchsorted(t_sim, t_nominal[0] - 1.0 / fs_out) - 4)
    hi = min(t_sim.size, np.searchsorted(t_sim, t_nominal[-1] + 1.0 / fs_out) + 4)
    spline = make_interp_spline(t_sim[lo:hi], x_sim[lo:hi], k=3)
    return spline(t_nominal + delta), delta


def quantize(x: np.ndarray, bits: int, range_g: float, include_gravity: bool) -> tuple[np.ndarray, int]:
    """Quantise with step 2 range / 2^bits and saturate to the signed code range.

    With ``include_gravity`` the sensor reads a + g (z axis up), so codes are computed for a + g and
    g is subtracted afterwards; saturation then happens at +range - g and -range - g.
    Returns the quantised signal and the number of saturated samples.
    """
    full = range_g * G
    step = 2.0 * full / 2**bits
    offset = G if include_gravity else 0.0
    codes = np.round((x + offset) / step)
    lo, hi = -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
    n_sat = int(np.count_nonzero((codes < lo) | (codes > hi)))
    return np.clip(codes, lo, hi) * step - offset, n_sat
