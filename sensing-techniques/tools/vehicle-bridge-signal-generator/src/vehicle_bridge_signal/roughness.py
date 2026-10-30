"""ISO 8608 road roughness by superposition of harmonics.

Displacement PSD (one-sided, spatial frequency n in cycles/m)::

    G_d(n) = G_d(n0) (n / n0)^(-w),   n0 = 0.1 cycles/m

Profile (equation 8 of Talebi-Kalaleh & Mei 2026)::

    r(x) = sum_i sqrt(2 G_d(n_i) dn) cos(2 pi n_i x + phi_i),   phi_i ~ U[0, 2 pi)

with n_i = i dn between n_min and n_max. The period of the profile is 1/dn. Its derivatives are
the exact derivatives of the same sum, so dr/dt = v r'(x) is analytical, never a finite difference.

Class table (n0 = 0.1 cycles/m convention, Agostinacchio et al. 2014, table 1): geometric means
A=16, B=64, C=256, D=1024, ... x 1e-6 m^3; limits A < 32, B 32-128, C 128-512, D 512-2048, ...
For log-uniform sampling the lower limit of class A is taken as 8e-6 m^3 [SUPUESTO], which keeps
16e-6 as its geometric mean.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["CLASS_MEANS", "class_bounds", "class_mean", "classify", "displacement_psd", "RoughnessProfile",
           "estimate_gd_n0", "temporal_psd"]

_LETTERS = "ABCDEFGH"
CLASS_MEANS = {c: 16e-6 * 4.0**i for i, c in enumerate(_LETTERS)}  # [m^3], n0 = 0.1 cycles/m


def class_bounds(letter: str) -> tuple[float, float]:
    """(lower, upper) limits of G_d(n0) [m^3] for an ISO 8608 class (n0 = 0.1 cycles/m)."""
    if letter not in CLASS_MEANS:
        raise ValueError(f"Unknown ISO 8608 class '{letter}' (use A-H)")
    mean = CLASS_MEANS[letter]
    return mean / 2.0, mean * 2.0


def class_mean(letter: str) -> float:
    return CLASS_MEANS[letter]


def classify(gd_n0: float) -> str:
    """ISO 8608 class of a value of G_d(n0) [m^3] (values below 8e-6 are class A)."""
    for letter in _LETTERS:
        if gd_n0 < class_bounds(letter)[1]:
            return letter
    return "H"


def displacement_psd(n, gd_n0: float, waviness: float = 2.0, n0: float = 0.1) -> np.ndarray:
    """G_d(n) [m^3] (one-sided, n in cycles/m)."""
    return gd_n0 * (np.asarray(n, float) / n0) ** (-waviness)


def temporal_psd(f, speed: float, gd_n0: float, waviness: float = 2.0, n0: float = 0.1) -> np.ndarray:
    """PSD in time at constant speed: G_t(f) = G_d(f / v) / v [m^2/Hz]."""
    return displacement_psd(np.asarray(f, float) / speed, gd_n0, waviness, n0) / speed


@dataclass(frozen=True)
class RoughnessProfile:
    """A realisation of a road profile (harmonic amplitudes and phases)."""

    spatial_frequencies: np.ndarray = field(repr=False)  # n_i [cycles/m]
    amplitudes: np.ndarray = field(repr=False)           # [m], including smoothing
    phases: np.ndarray = field(repr=False)               # [rad]
    period: float                                        # 1/dn [m]
    gd_n0: float
    waviness: float
    n0: float
    smoothing_length: float = 0.0

    @classmethod
    def generate(cls, rng: np.random.Generator, gd_n0: float, waviness: float = 2.0, n0: float = 0.1,
                 n_min: float = 0.01, n_max: float = 10.0, period: float = 2048.0,
                 smoothing_length: float = 0.0) -> "RoughnessProfile":
        """Draw random phases for the harmonics n_i = i/period, n_min <= n_i <= n_max."""
        if not (gd_n0 >= 0 and period > 0 and 0 < n_min < n_max):
            raise ValueError("Invalid roughness parameters")
        dn = 1.0 / period
        i_lo = int(np.ceil(n_min * period - 1e-9))
        i_hi = int(np.floor(n_max * period + 1e-9))
        if i_hi < i_lo:
            raise ValueError("The roughness band contains no harmonic; increase profile_period_m")
        n = np.arange(i_lo, i_hi + 1) * dn
        amp = np.sqrt(2.0 * displacement_psd(n, gd_n0, waviness, n0) * dn)
        if smoothing_length > 0:
            # Centred moving average of length l: multiplies each harmonic by sinc(n l) (exact).
            amp = amp * np.sinc(n * smoothing_length)
        phases = rng.uniform(0.0, 2.0 * np.pi, n.size)
        return cls(n, amp, phases, float(period), float(gd_n0), float(waviness), float(n0), float(smoothing_length))

    @property
    def n_harmonics(self) -> int:
        return int(self.spatial_frequencies.size)

    def _coefficients(self, derivative: int) -> np.ndarray:
        k = 2j * np.pi * self.spatial_frequencies
        return self.amplitudes * k**derivative * np.exp(1j * self.phases)

    def evaluate(self, x, derivative: int = 0, chunk: int = 2_000_000) -> np.ndarray:
        """Direct (vectorised) sum at arbitrary positions ``x``; ``derivative`` in {0, 1, 2} -> r, r', r''."""
        x = np.asarray(x, float)
        flat = x.ravel()
        coef = self._coefficients(derivative)
        out = np.empty(flat.size)
        step = max(1, chunk // max(1, coef.size))
        for s in range(0, flat.size, step):
            xs = flat[s:s + step]
            out[s:s + step] = np.real(np.exp(2j * np.pi * np.outer(xs, self.spatial_frequencies)) @ coef)
        return out.reshape(x.shape)

    def evaluate_uniform(self, x0: float, dx: float, n_points: int, derivative: int = 0) -> np.ndarray:
        """Same sum on the uniform grid x_j = x0 + j dx, in O(N log N) with a chirp-z transform.

        Writing n_i = (i0 + m)/P and alpha = 2 pi dx / P, the sum is
        Re[e^{i alpha i0 j} sum_m c_m e^{i alpha m j}] with c_m including the phase at x0. The
        Bluestein identity m j = (m^2 + j^2 - (j - m)^2) / 2 turns it into a convolution evaluated
        with FFTs. When dx * n_points equals the period this is exactly an inverse FFT; it is
        equivalent to :meth:`evaluate` to round-off for any dx.
        """
        if n_points <= 0:
            return np.zeros(0)
        coef = self._coefficients(derivative) * np.exp(2j * np.pi * self.spatial_frequencies * x0)
        m_count = coef.size
        i0 = int(round(self.spatial_frequencies[0] * self.period))
        frac = dx / self.period            # alpha / (2 pi)
        m = np.arange(m_count, dtype=np.float64)
        j = np.arange(n_points, dtype=np.float64)
        # Chirp phases pi*frac*k^2 reduced modulo 2 pi with exact integer squares.
        def chirp(k):
            return np.exp(1j * 2 * np.pi * np.mod(0.5 * frac * k * k, 1.0))
        g = coef * chirp(m)
        k_all = np.arange(-(m_count - 1), n_points, dtype=np.float64)
        h = np.conj(chirp(k_all))
        size = 1 << int(np.ceil(np.log2(m_count + n_points - 1)))
        conv = np.fft.ifft(np.fft.fft(g, size) * np.fft.fft(h, size))
        y = conv[m_count - 1:m_count - 1 + n_points]
        total = chirp(j) * y * np.exp(1j * 2 * np.pi * np.mod(i0 * frac * j, 1.0))
        return np.real(total)

    def psd(self, n) -> np.ndarray:
        """Target PSD G_d(n) of this profile (without smoothing)."""
        return displacement_psd(n, self.gd_n0, self.waviness, self.n0)


def estimate_gd_n0(profile: np.ndarray, dx: float, band: tuple[float, float] = (0.05, 5.0),
                   waviness: float = 2.0, n0: float = 0.1, nperseg: int | None = None) -> dict:
    """Estimate G_d(n0) and the ISO class of a sampled profile with Welch's method.

    Welch (Hann window, 50 % overlap). With w fixed, G_d(n0) is the mean of
    PSD(n) (n / n0)^w over the band (an unbiased linear average).
    """
    from scipy.signal import welch

    profile = np.asarray(profile, float)
    if nperseg is None:
        nperseg = 1 << int(np.floor(np.log2(max(16, profile.size // 8))))
    n, p = welch(profile, fs=1.0 / dx, window="hann", nperseg=nperseg, noverlap=nperseg // 2, detrend=False)
    sel = (n >= band[0]) & (n <= band[1])
    if not np.any(sel):
        raise ValueError("No Welch bins inside the requested band")
    gd = float(np.mean(p[sel] * (n[sel] / n0) ** waviness))
    # Least-squares slope in log-log (diagnostic of the waviness).
    slope = np.polyfit(np.log(n[sel]), np.log(p[sel]), 1)[0]
    return {"gd_n0": gd, "class": classify(gd), "waviness_fit": float(-slope), "n": n, "psd": p}
