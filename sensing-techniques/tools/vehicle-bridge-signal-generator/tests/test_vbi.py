"""Vehicle-bridge interaction: Yang et al. (2004) benchmark, coupled vs decoupled, spectral content."""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import vehicle_bridge_signal as vbs
from vehicle_bridge_signal.spectral import welch_psd

REF = Path(__file__).parent / "reference" / "yang2004_opensees_vbi_1000hz.npz"


def r2(y, yhat):
    return 1 - np.sum((y - yhat) ** 2) / np.sum((y - np.mean(y)) ** 2)


def _yang(interaction="coupled", **over):
    cfg = vbs.load_preset("yang2004_benchmark", {"simulation.interaction": interaction, **over})
    return vbs.simulate(cfg, 0, return_internal=True)


def _compare_with_reference(ref, interaction):
    res = _yang(interaction)
    it = res.internal
    bridge, t = it["bridge"], it["t_sim"]
    w_mid = it["response"].q[:, :, 0] @ bridge.shapes(bridge.span / 2)
    z = it["response"].z[:, 0, 0]
    tr, tv = ref[f"{interaction}_time"], ref[f"{interaction}_vehicle_time"]
    sel, selv = tr <= 4.0, tv <= 4.0
    r2_w = r2(ref[f"{interaction}_midspan_disp"][sel], np.interp(tr[sel], t, w_mid))
    r2_z = r2(ref[f"{interaction}_vehicle_disp"][selv], np.interp(tv[selv], t, z))
    return res, r2_w, r2_z


def test_yang2004_frequencies():
    res = _yang()
    assert res.metadata["f1"] == pytest.approx(2.08, abs=0.01)
    assert res.metadata["vehicle_frequencies"][0] == pytest.approx(3.25, abs=0.01)
    assert res.metadata["mass_ratio"] == pytest.approx(0.01)


@pytest.mark.parametrize("interaction", ["coupled", "decoupled"])
def test_yang2004_matches_opensees_vbi_reference(interaction):
    """Validation 3: midspan deflection and vehicle displacement vs OpenSees-VBI-Simulation, R^2 > 0.99.

    The reference was produced once with the professors' framework (commit stored in the file) by
    tests/reference/run_opensees_vbi_yang2004.py.
    """
    ref = np.load(REF)
    assert ref["coupled_bridge_freqs"][0] == pytest.approx(2.084, abs=1e-3)
    _, r2_w, r2_z = _compare_with_reference(ref, interaction)
    assert r2_w > 0.999 and r2_z > 0.999


@pytest.mark.opensees
def test_yang2004_against_live_opensees(tmp_path):
    """Optional: rerun the framework (needs openseespy and OVBI_DIR pointing to a clone of
    https://github.com/MTalebi/OpenSees-VBI-Simulation)."""
    pytest.importorskip("openseespy")
    ovbi = os.environ.get("OVBI_DIR")
    if not ovbi or not Path(ovbi, "opensees_vbi.py").is_file():
        pytest.skip("set OVBI_DIR to a local clone of OpenSees-VBI-Simulation")
    out = tmp_path / "ref.npz"
    script = Path(__file__).parent / "reference" / "run_opensees_vbi_yang2004.py"
    subprocess.run([sys.executable, str(script), ovbi, str(out), "1000"], check=True, capture_output=True)
    ref = np.load(out)
    for interaction in ("coupled", "decoupled"):
        _, r2_w, r2_z = _compare_with_reference(ref, interaction)
        assert r2_w > 0.99 and r2_z > 0.99


def _run_mass(mass, interaction):
    """Yang bridge with a vehicle of the given mass and the same f_v; returns vehicle disp/acc, mu."""
    k = mass * (2 * np.pi * 3.2487) ** 2
    res = _yang(interaction, **{"vehicle.sprung_mass.mass": mass, "vehicle.sprung_mass.stiffness": k})
    win = res.internal["window"]
    return res.internal["response"].z[win, 0, 0], res.measured, res.metadata["mass_ratio"]


def _gap(a, b):
    return np.linalg.norm(a - b) / np.linalg.norm(a)


def test_coupled_vs_decoupled_difference_grows_with_mass_ratio():
    """Validation 4: nearly identical displacements for mu ~ 1 %, difference growing with mu."""
    disp_gap, acc_gap, mus = [], [], []
    for mass in (1200.0, 6000.0, 12000.0, 24000.0):
        zc, ac, mu = _run_mass(mass, "coupled")
        zd, ad, _ = _run_mass(mass, "decoupled")
        disp_gap.append(_gap(zc, zd))
        acc_gap.append(_gap(ac, ad))
        mus.append(mu)
    assert mus == pytest.approx([0.01, 0.05, 0.1, 0.2])
    assert disp_gap[0] < 0.05
    assert all(b > a for a, b in zip(disp_gap, disp_gap[1:]))
    assert all(b > a for a, b in zip(acc_gap, acc_gap[1:]))


def test_coupled_decoupled_gap_equals_the_reference_framework_gap():
    """The difference between our coupled and decoupled solutions equals the difference between the
    two modes of OpenSees-VBI-Simulation (Yang benchmark, mu = 1 %): an independent check of the
    interaction terms."""
    ref = np.load(REF)
    sel = ref["coupled_time"] <= 4.0
    ours = {}
    for interaction in ("coupled", "decoupled"):
        res = _yang(interaction)
        t = res.internal["t_sim"]
        m = (t >= 0) & (t <= 4.0 + 1e-9)
        ours[interaction] = (res.internal["response"].z[m, 0, 0], res.internal["response"].zdd[m, 0, 0])
    ref_disp = _gap(ref["coupled_vehicle_disp"][sel], ref["decoupled_vehicle_disp"][sel])
    ref_acc = _gap(ref["coupled_vehicle_acc"][sel][1:], ref["decoupled_vehicle_acc"][sel][1:])
    assert _gap(*[ours[k][0] for k in ("coupled", "decoupled")]) == pytest.approx(ref_disp, rel=0.05)
    assert _gap(*[ours[k][1] for k in ("coupled", "decoupled")]) == pytest.approx(ref_acc, rel=0.05)


def test_vehicle_response_contains_the_theoretical_frequencies():
    """Validation 5 (Yang et al. 2004): with a smooth road, an undamped bridge and a light vehicle, the
    vehicle acceleration while on the bridge is a combination of exactly these frequencies:
    vehicle f_v (free vibration), n v/L (quasi-static, twice the driving frequency v/(2L)) and the
    shifted bridge frequencies f_bn +/- n v/(2L). A least-squares fit on that basis is ~exact."""
    res = _yang("decoupled", **{"vehicle.sprung_mass.mass": 12.0, "vehicle.sprung_mass.stiffness": 12.0 * (2 * np.pi * 3.2487) ** 2})
    m = res.metadata
    t, y = res.t, res.measured
    on = res.on_bridge.copy()
    on[np.flatnonzero(on)[[0, -1]]] = False    # skip the entry/exit samples (kinks of u_c)
    fd = m["driving_frequency"]                 # v / (2L)
    freqs = [m["vehicle_frequencies"][0]]
    for n, fb in enumerate(m["bridge_frequencies"], start=1):
        freqs += [2 * n * fd, fb - n * fd, fb + n * fd]
    basis = [np.ones(on.sum())]
    for f in freqs:
        basis += [np.cos(2 * np.pi * f * t[on]), np.sin(2 * np.pi * f * t[on])]
    a = np.array(basis).T
    coef, *_ = np.linalg.lstsq(a, y[on], rcond=None)
    assert r2(y[on], a @ coef) > 0.999
    # Removing the shifted bridge frequencies destroys the fit: they are really present.
    keep = [0, 1, 2] + [j for j in range(3, a.shape[1]) if (j - 3) // 2 % 3 == 0]
    coef2, *_ = np.linalg.lstsq(a[:, keep], y[on], rcond=None)
    assert r2(y[on], a[:, keep] @ coef2) < 0.9


def test_spectral_peaks_low_speed():
    """Welch spectrum (CIV E 789 conventions) of smooth-road crossings of the Yang benchmark.

    At 5 m/s the shifted bridge frequencies dominate: peaks within f_bn +/- (n v/(2L) + df) for modes 1
    and 2, and no separate vehicle peak (a slow entry hardly excites the vehicle's free vibration).
    At 10 m/s the vehicle frequency appears as well. The pair f_b1 +/- v/(2L) is separated by
    v/L = 1/T_crossing (the resolution limit of the on-bridge record): it shows as one peak.
    """
    from scipy.signal import find_peaks

    def peaks(speed):
        res = vbs.simulate(vbs.load_preset("yang2004_benchmark", {"crossing.speed": speed}), 0)
        f, p = welch_psd(res.measured, res.fs, nperseg=512)
        sel = (f > 0.3) & (f < 20)
        idx, _ = find_peaks(np.log10(p[sel]), prominence=0.3)
        return f[sel][idx], f[1] - f[0], res.metadata

    pk, df, m = peaks(5.0)
    fd, fb = m["driving_frequency"], m["bridge_frequencies"]
    for n in (1, 2):
        assert np.min(np.abs(pk - fb[n - 1])) <= n * fd + df
    assert np.min(np.abs(pk - m["vehicle_frequencies"][0])) > df
    pk, df, m = peaks(10.0)
    assert np.min(np.abs(pk - m["vehicle_frequencies"][0])) <= df
