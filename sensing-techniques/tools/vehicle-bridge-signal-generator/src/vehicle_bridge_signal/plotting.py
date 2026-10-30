"""Plots of simulated crossings (matplotlib). Every function returns the figure."""
from __future__ import annotations

import numpy as np

from .spectral import expected_frequencies, stft_spectrogram, welch_psd

__all__ = ["plot_components", "plot_component_spectra", "plot_spectrogram", "plot_bridge_modes", "plot_vehicle_frf",
           "plot_preview"]

COLORS = {"measured": "#222222", "bridge": "#1f77b4", "roughness": "#8c564b", "engine": "#d62728",
          "wheel": "#9467bd", "noise": "#7f7f7f"}


def _plt():
    import matplotlib.pyplot as plt
    return plt


def _arrays(obj) -> dict:
    """Common view of a SimulationResult or of a dataset sample dict."""
    if isinstance(obj, dict):
        t = obj["time"]
        return {"t": t, "fs": 1.0 / (t[1] - t[0]), "measured": obj["measured"], "components": obj["components"],
                "mask": np.asarray(obj["on_bridge_mask"], bool), "metadata": obj.get("metadata", {})}
    return {"t": obj.t, "fs": obj.fs, "measured": obj.measured, "components": obj.components, "mask": obj.on_bridge,
            "metadata": obj.metadata}


def _color(key: str) -> str:
    return COLORS["bridge"] if key.startswith("bridge") else COLORS.get(key, "#2ca02c")


def _shade(ax, t, mask):
    if mask.any():
        idx = np.flatnonzero(mask)
        ax.axvspan(t[idx[0]], t[idx[-1]], color="#ffd54f", alpha=0.25, lw=0)


def plot_components(result, keys=None, title: str | None = None, figsize=None):
    """Stacked plot: measured signal and every component (on-bridge window shaded)."""
    plt = _plt()
    a = _arrays(result)
    keys = list(a["components"]) if keys is None else list(keys)
    fig, axes = plt.subplots(len(keys) + 1, 1, sharex=True, figsize=figsize or (9, 1.25 * (len(keys) + 1) + 0.6))
    series = [("measured", a["measured"])] + [(k, a["components"][k]) for k in keys]
    for ax, (k, y) in zip(axes, series):
        ax.plot(a["t"], y, lw=0.7, color=COLORS["measured"] if k == "measured" else _color(k))
        _shade(ax, a["t"], a["mask"])
        rms = np.sqrt(np.mean(np.asarray(y, float) ** 2))
        ax.set_ylabel(k, rotation=0, ha="right", va="center", fontsize=8)
        ax.text(1.0, 0.5, f" rms {rms:.2e}", transform=ax.transAxes, fontsize=7, va="center")
        ax.tick_params(labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)
    axes[-1].set_xlabel("t [s]")
    fig.suptitle(title or "Measured signal and components [m/s²] (shaded: vehicle on the bridge)", fontsize=10)
    fig.tight_layout()
    return fig


def plot_component_spectra(result, keys=None, nperseg: int | None = None, show_expected: bool = True, fmax=None):
    """Welch PSD of the measured signal and each component, with the expected frequencies."""
    plt = _plt()
    a = _arrays(result)
    keys = [k for k in a["components"] if np.any(a["components"][k])] if keys is None else list(keys)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    f, p = welch_psd(a["measured"], a["fs"], nperseg)
    ax.semilogy(f, p, color=COLORS["measured"], lw=1.2, label="measured")
    for k in keys:
        f, p = welch_psd(a["components"][k], a["fs"], nperseg)
        ax.semilogy(f, np.maximum(p, 1e-30), lw=0.9, label=k, color=_color(k),
                    ls="-" if k in ("bridge_mode_1", "roughness", "engine", "noise", "wheel") else "--")
    if show_expected and a["metadata"]:
        ex = expected_frequencies(a["metadata"])
        for fv in ex["vehicle"]:
            ax.axvline(fv, color="#2ca02c", ls=":", lw=1)
        for fb in ex["bridge"]:
            ax.axvline(fb, color=COLORS["bridge"], ls=":", lw=1)
    ax.set_xlim(0, fmax or a["fs"] / 2)
    ax.set_xlabel("f [Hz]")
    ax.set_ylabel("PSD [(m/s²)²/Hz]")
    ax.set_title("Welch PSD (Hann, 50 % overlap). Dotted: vehicle (green) and bridge (blue) frequencies", fontsize=9)
    ax.legend(fontsize=7, ncol=3)
    fig.tight_layout()
    return fig


def plot_spectrogram(result, key: str = "measured", nperseg: int = 128, fmax: float | None = None):
    plt = _plt()
    a = _arrays(result)
    y = a["measured"] if key == "measured" else a["components"][key]
    f, t, s = stft_spectrogram(y, a["fs"], nperseg)
    fig, ax = plt.subplots(figsize=(9, 3.5))
    m = ax.pcolormesh(t + a["t"][0], f, 10 * np.log10(s + 1e-30), shading="auto", cmap="viridis")
    fig.colorbar(m, ax=ax, label="dB")
    ax.set_ylim(0, fmax or a["fs"] / 2)
    ax.set_xlabel("t [s]")
    ax.set_ylabel("f [Hz]")
    ax.set_title(f"STFT spectrogram of '{key}' (Hann, nperseg={nperseg})", fontsize=9)
    fig.tight_layout()
    return fig


def plot_bridge_modes(bridge, n_points: int = 300):
    plt = _plt()
    x = np.linspace(0, bridge.span, n_points)
    fig, ax = plt.subplots(figsize=(8, 3))
    for n, (phi, f) in enumerate(zip(bridge.shapes(x).T, bridge.frequencies), start=1):
        ax.plot(x, phi, label=f"mode {n}: {f:.2f} Hz")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("x [m]")
    ax.set_ylabel(r"$\phi_n(x)$")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_vehicle_frf(vehicle, f_max: float = 50.0, n: int = 2000):
    plt = _plt()
    f = np.linspace(0.05, f_max, n)
    h = vehicle.frf_contact_to_sensor(f)
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.semilogy(f, np.abs(h) / (2 * np.pi * f) ** 2, label="|body acc. / contact acc.|")
    for fv in vehicle.natural_frequencies():
        ax.axvline(fv, color="gray", ls=":")
    ax.axhline(1, color="k", lw=0.5)
    ax.set_xlabel("f [Hz]")
    ax.set_ylabel("transmissibility [-]")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_preview(results, titles=None):
    """Grid with the measured signal and the total bridge component of several samples."""
    plt = _plt()
    n = len(results)
    fig, axes = plt.subplots(n, 1, figsize=(9, 1.9 * n + 0.5), squeeze=False)
    for i, (ax, r) in enumerate(zip(axes[:, 0], results)):
        a = _arrays(r)
        bridge = np.sum([v for k, v in a["components"].items() if k.startswith("bridge")], axis=0)
        ax.plot(a["t"], a["measured"], lw=0.6, color=COLORS["measured"], label="measured")
        ax2 = ax.twinx()
        ax2.plot(a["t"], bridge, lw=0.9, color=COLORS["bridge"], label="bridge (right axis)")
        _shade(ax, a["t"], a["mask"])
        md = a["metadata"]
        label = titles[i] if titles else (f"L={md.get('span', np.nan):.1f} m, v={md.get('speed', np.nan):.1f} m/s, "
                                          f"f1={md.get('f1', np.nan):.2f} Hz")
        ax.set_title(label, fontsize=8)
        ax.tick_params(labelsize=7)
        ax2.tick_params(labelsize=7, colors=COLORS["bridge"])
    axes[-1, 0].set_xlabel("t [s]")
    fig.tight_layout()
    return fig
