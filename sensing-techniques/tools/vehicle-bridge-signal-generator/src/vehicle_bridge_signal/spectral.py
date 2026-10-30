"""Spectral helpers with the CIV E 789 conventions (used for verification and plots only).

* Welch's method instead of the FFT of the whole record; Hann window; 50 % overlap;
  ``nperseg`` a power of two smaller than N. Frequency resolution df = fs / nperseg; zero padding
  does not improve it.
* Peak locations are read at the PSD bin of maximum value (no interpolation).
"""
from __future__ import annotations

import numpy as np
from scipy import signal

__all__ = ["default_nperseg", "welch_psd", "stft_spectrogram", "expected_frequencies", "peak_in_band"]


def default_nperseg(n: int, divisor: int = 4) -> int:
    """Largest power of two <= n / divisor (at least 16)."""
    return int(max(16, 1 << int(np.floor(np.log2(max(n // divisor, 16))))))


def welch_psd(x, fs: float, nperseg: int | None = None, axis: int = -1):
    """One-sided Welch PSD (Hann, 50 % overlap). Returns ``(f, P)``."""
    x = np.asarray(x, float)
    n = x.shape[axis]
    nperseg = default_nperseg(n) if nperseg is None else int(nperseg)
    if nperseg >= n:
        raise ValueError(f"nperseg ({nperseg}) must be smaller than the number of samples ({n})")
    return signal.welch(x, fs=fs, window="hann", nperseg=nperseg, noverlap=nperseg // 2, axis=axis)


def stft_spectrogram(x, fs: float, nperseg: int = 128):
    """Spectrogram (Hann, 50 % overlap): ``(f, t, Sxx)``."""
    return signal.spectrogram(np.asarray(x, float), fs=fs, window="hann", nperseg=nperseg, noverlap=nperseg // 2)


def expected_frequencies(metadata: dict) -> dict:
    """Frequencies expected in the vehicle response (Yang et al. 2004).

    * vehicle natural frequencies;
    * driving frequency v / (2 L);
    * bridge frequencies f_bn and the shifted pairs f_bn +/- n v / (2 L);
    * wheel rotation frequency and mean engine firing frequency, if active.
    """
    fd = float(metadata["driving_frequency"])
    fb = np.asarray(metadata["bridge_frequencies"], float)
    fb = fb[np.isfinite(fb)]
    n = np.arange(1, fb.size + 1)
    out = {
        "vehicle": [float(x) for x in np.atleast_1d(metadata["vehicle_frequencies"]) if np.isfinite(x)],
        "driving": fd,
        "bridge": fb.tolist(),
        "bridge_shifted_minus": (fb - n * fd).tolist(),
        "bridge_shifted_plus": (fb + n * fd).tolist(),
    }
    if metadata.get("wheel_enabled") and "wheel_frequency" in metadata:
        out["wheel"] = float(metadata["wheel_frequency"])
    if metadata.get("engine_enabled") and "engine_firing_frequency" in metadata:
        out["engine_firing"] = float(metadata["engine_firing_frequency"])
    return out


def peak_in_band(f, p, f_low: float, f_high: float) -> float:
    """Frequency of the largest PSD value inside [f_low, f_high]."""
    f, p = np.asarray(f), np.asarray(p)
    sel = (f >= f_low) & (f <= f_high)
    if not np.any(sel):
        raise ValueError("No PSD bins in the requested band")
    return float(f[sel][np.argmax(p[sel])])
