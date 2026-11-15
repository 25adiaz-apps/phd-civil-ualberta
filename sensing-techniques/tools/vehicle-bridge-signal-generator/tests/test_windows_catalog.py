"""Windows defined by road lengths, quantities under the wheel, vehicle that does not load the bridge,
bridge catalog, padded/ragged layouts, derived (cropped) datasets and wheel lift-off resampling."""
import h5py
import numpy as np
import pytest

import vehicle_bridge_signal as vbs
from vehicle_bridge_signal.config import ConfigError
from vehicle_bridge_signal.sampling import plan_dataset, scenario_for

AUTO = {"simulation.n_samples_out": "auto", "crossing.approach_length_m": 20.0, "crossing.exit_length_m": 15.0,
        "dataset.layout": "ragged"}
FREE_BRIDGE = {"simulation.interaction": "decoupled", "simulation.vehicle_loads_bridge": False}
CATALOG = [
    {"name": "S1", "bridge.span": 12.0, "bridge.f1": 9.0, "bridge.mass_per_length": 5000.0,
     "bridge.damping_ratio": 0.02, "roughness.class": "A"},
    {"name": "S2", "bridge.span": 20.0, "bridge.f1": 5.0, "bridge.mass_per_length": 8000.0,
     "bridge.damping_ratio": 0.015, "roughness.class": "C"},
]
RANDOM = {**AUTO, **FREE_BRIDGE, "simulation.amplitude_mode": "physical", "engine.enabled": False,
          "ambient.enabled": True, "crossing.speed": {"dist": "uniform", "low": 10.0, "high": 20.0},
          "dataset.bridge_catalog": CATALOG, "dataset.n_bridges": None, "dataset.n_vehicles": None,
          "dataset.partitions": None, "dataset.store_profile": True, "sensor.quantization.enabled": False}


def _auto(**over):
    return vbs.load_preset("typical_car_smartphone", {**AUTO, **over})


# ---- window defined by road lengths ---------------------------------------------------------
@pytest.mark.parametrize("speed", [5.0, 13.7, 25.0])
def test_auto_window_covers_the_requested_road(speed):
    res = vbs.simulate(_auto(**{"crossing.speed": speed}), 3)
    m, x = res.metadata, res.profile["position"][0]
    assert x[0] == pytest.approx(-20.0, abs=1e-9)                 # first sample: 20 m before the bridge
    assert m["span"] + 15.0 - speed / res.fs < x[-1] <= m["span"] + 15.0 + 1e-9
    assert m["entry_time"] == pytest.approx(20.0 / speed)
    assert res.measured.size == m["n_samples_out"] == res.t.size
    assert np.allclose(np.diff(x), speed / res.fs)
    seg = res.segment
    assert np.all(np.diff(seg.astype(int)) >= 0) and set(np.unique(seg)) == {0, 1, 2}
    assert np.array_equal(seg == 1, res.on_bridge)
    assert np.all(x[seg == 0] < 0) and np.all(x[seg == 2] > m["span"])


def test_auto_window_validation():
    with pytest.raises(ConfigError, match="approach_length_m"):
        vbs.load_preset("typical_car_smartphone", {"simulation.n_samples_out": "auto", "dataset.layout": "ragged"})
    with pytest.raises(ConfigError, match="only used with"):
        vbs.load_preset("typical_car_smartphone", {"crossing.approach_length_m": 10.0, "crossing.exit_length_m": 10.0})
    with pytest.raises(ConfigError, match="padded or ragged"):
        vbs.load_preset("typical_car_smartphone", {**AUTO, "dataset.layout": "fixed"})
    with pytest.raises(ConfigError, match="pad_to_samples"):
        vbs.load_preset("typical_car_smartphone", {**AUTO, "dataset.layout": "padded"})
    with pytest.raises(ConfigError, match="n_samples_out: auto"):
        vbs.load_preset("typical_car_smartphone", {"dataset.layout": "ragged"})
    with pytest.raises(ConfigError, match="decoupled"):
        vbs.load_preset("typical_car_smartphone", {"simulation.vehicle_loads_bridge": False})


# ---- quantities under the wheel -------------------------------------------------------------
def test_profile_equals_its_definitions():
    """Road: the exact harmonic sum at the wheel positions (r and v^2 r''). Bridge: sum_n q_n phi_n(x)
    and sum_n q_n'' phi_n(x) at the output instants, zero off the bridge."""
    res = vbs.simulate(_auto(**{"ambient.enabled": True, "ambient.modal_accel_rms": 0.01}), 4, return_internal=True)
    it, p = res.internal, res.profile
    x, v = p["position"][0], res.metadata["speed"]
    prof = it["roughness_profile"]
    assert np.allclose(p["road_displacement"][0], prof.evaluate(x), rtol=0, atol=1e-12)
    ref = v * v * prof.evaluate(x, 2)
    assert np.max(np.abs(p["road_acceleration"][0] - ref)) < 1e-9 * np.max(np.abs(ref))
    bridge, oi = it["bridge"], it["out_index"]
    scales = np.array([it["scales"][n] for n in it["columns"]])
    q, qdd = it["response"].q[oi] @ scales, it["response"].qdd[oi] @ scales
    phi = bridge.shapes(x)
    assert np.allclose(p["bridge_displacement"][0], np.sum(phi * q, axis=1), rtol=0, atol=1e-15)
    assert np.allclose(p["bridge_acceleration"][0], np.sum(phi * qdd, axis=1), rtol=0, atol=1e-12)
    off = ~res.on_bridge
    assert np.all(p["bridge_displacement"][0][off] == 0) and np.all(p["bridge_acceleration"][0][off] == 0)
    assert np.any(p["bridge_displacement"][0][res.on_bridge] != 0)


# ---- vehicle that does not load the bridge --------------------------------------------------
def test_vehicle_that_does_not_load_the_bridge():
    base = {**FREE_BRIDGE, "engine.enabled": False}
    still = vbs.simulate(_auto(**base, **{"ambient.enabled": False}), 1)
    assert np.all(still.profile["bridge_displacement"] == 0)
    assert all(np.all(v == 0) for k, v in still.components.items() if k.startswith("bridge_"))
    assert still.metadata["vehicle_loads_bridge"] is False
    # With ambient vibration the bridge motion does not depend on the vehicle at all.
    amb = {**base, "ambient.enabled": True, "ambient.modal_accel_rms": 0.005}
    a = vbs.simulate(_auto(**amb), 2)
    b = vbs.simulate(_auto(**amb, **{"vehicle.quarter_car.body_mass": 300.0}), 2)
    assert np.array_equal(a.profile["bridge_displacement"], b.profile["bridge_displacement"])
    assert not np.array_equal(a.measured, b.measured)
    # The loaded (default) decoupled bridge does depend on the vehicle weight.
    loaded = vbs.simulate(_auto(**{"simulation.interaction": "decoupled", "engine.enabled": False,
                                   "ambient.enabled": False}), 1)
    assert np.max(np.abs(loaded.profile["bridge_displacement"])) > 0


# ---- bridge catalog -------------------------------------------------------------------------
def test_bridge_catalog_assigns_fixed_bridges():
    cfg = vbs.load_preset("smoke", RANDOM)
    plans = plan_dataset(cfg, 6, 2)
    assert sorted(p.bridge_id for p in plans) == [0, 0, 0, 1, 1, 1]
    for p in plans:
        sc, info = scenario_for(cfg, p, 2)
        entry = CATALOG[p.bridge_id]
        assert info["bridge_name"] == entry["name"]
        assert sc["bridge"]["span"] == entry["bridge.span"] and sc["bridge"]["f1"] == entry["bridge.f1"]
        assert sc["roughness"]["class"] == entry["roughness.class"]
        lo, hi = vbs.class_bounds(entry["roughness.class"])
        assert lo <= sc["roughness"]["gd_n0"] <= hi
        assert sc["dataset"]["bridge_catalog"] is None


def test_bridge_catalog_validation():
    bad = [({"dataset.n_bridges": 3}, "n_bridges"),
           ({"dataset.bridge_catalog": [{"dataset.n_samples": 3}]}, "global"),
           ({"dataset.bridge_catalog": [{"bridge.spam": 3.0}]}, "Did you mean"),
           ({"dataset.bridge_catalog": [{"bridge.span": -1.0}]}, "> 0"),
           ({"dataset.bridge_catalog": [CATALOG[0], CATALOG[0]]}, "unique"),
           ({"dataset.bridge_catalog": []}, "non-empty")]
    for over, msg in bad:
        with pytest.raises(ConfigError, match=msg):
            vbs.load_preset("smoke", {**RANDOM, **over})


def test_roughness_fixed_per_bridge_or_new_per_crossing():
    """scope bridge: every crossing of a bridge drives over the same road profile (in space);
    scope crossing: a new profile for every crossing, inside the class of the bridge."""
    def profiles(scope):
        cfg = vbs.load_preset("smoke", {**RANDOM, "roughness.scope": scope})
        out = []
        for p in [p for p in plan_dataset(cfg, 6, 2) if p.bridge_id == 1][:2]:
            sc, info = scenario_for(cfg, p, 2)
            res = vbs.simulate(sc, info["sample_seed"], return_internal=True)
            out.append((res.internal["roughness_profile"], res.profile["position"][0], sc["crossing"]["speed"]))
        return out
    (pa, xa, va), (pb, xb, vb) = profiles("bridge")
    assert va != vb
    x = np.linspace(-20, 30, 101)
    assert np.allclose(pa.evaluate(x), pb.evaluate(x), rtol=0, atol=1e-15)
    (pa, _, _), (pb, _, _) = profiles("crossing")
    assert not np.allclose(pa.evaluate(x), pb.evaluate(x))


# ---- layouts and derived datasets -----------------------------------------------------------
@pytest.fixture(scope="module")
def ragged_and_padded(tmp_path_factory):
    d = tmp_path_factory.mktemp("layouts")
    cfg = vbs.load_preset("smoke", RANDOM)
    rag = vbs.generate_dataset(cfg, 5, d / "rag.h5", seed=7, workers=2, progress=False)
    pad_cfg = vbs.with_overrides(cfg, {"dataset.layout": "padded", "dataset.pad_to_samples": 1200})
    pad = vbs.generate_dataset(pad_cfg, 5, d / "pad.h5", seed=7, workers=1, progress=False)
    return d, cfg, rag, pad


def test_ragged_and_padded_hold_the_same_crossings(ragged_and_padded):
    _, _, rag, pad = ragged_and_padded
    with vbs.load_dataset(rag) as r, vbs.load_dataset(pad) as p:
        assert r.layout == "ragged" and p.layout == "padded" and len(r) == len(p) == 5
        assert np.array_equal(r.lengths, p.lengths) and len(np.unique(r.lengths)) > 1
        assert r.file["offsets"][-1] == r.lengths.sum() and r.measured.shape == (r.lengths.sum(),)
        assert p.measured.shape == (5, 1200) and p.file["valid_mask"][()].sum(axis=1).tolist() == p.lengths.tolist()
        for i in range(5):
            n = int(p.lengths[i])
            assert np.all(p.measured[i, n:] == 0)
            for name in r.signal_names:
                assert np.array_equal(r.signal(name, i), p.signal(name, i)), name
            s = r.sample(i)
            assert s["time"].size == n and s["profile"]["position"].shape == (1, n)
            assert s["metadata"]["bridge_name"] in ("S1", "S2")
        assert "rag.h5" in repr(r) and "ragged" in repr(r)
        with pytest.raises(ValueError):
            r.stack_components()
        with pytest.raises(ValueError):
            r.time


def test_dataset_sample_equals_simulation_and_regenerates(ragged_and_padded):
    _, cfg, rag, _ = ragged_and_padded
    with vbs.load_dataset(rag) as ds:
        for i in (0, 3):
            res = vbs.regenerate_sample(ds, i)
            assert np.array_equal(ds.signal("measured", i), res.measured.astype(np.float32))
            assert np.array_equal(ds.signal("profile/road_displacement", i),
                                  res.profile["road_displacement"].astype(np.float32))


def test_padded_overflow_stops_with_a_clear_message(tmp_path):
    cfg = vbs.load_preset("smoke", {**RANDOM, "dataset.layout": "padded", "dataset.pad_to_samples": 100})
    with pytest.raises(RuntimeError, match="pad_to_samples"):
        vbs.generate_dataset(cfg, 1, tmp_path / "x.h5", progress=False)
    assert not (tmp_path / "x.h5").exists()


def test_derive_crop_keeps_the_same_crossings(ragged_and_padded):
    d, _, rag, _ = ragged_and_padded
    out = vbs.derive_dataset(rag, d / "crop.h5", approach_length_m=8.0, exit_length_m=5.0, layout="padded",
                             pad_to_samples=1200, progress=False)
    with vbs.load_dataset(rag) as src, vbs.load_dataset(out) as ds:
        meta, m0 = ds.metadata, src.metadata
        assert ds.layout == "padded" and ds.file.attrs["derived_from"] == "rag.h5"
        for i in range(len(ds)):
            j0, n = int(meta["crop_start_index"][i]), int(ds.lengths[i])
            v = float(meta["speed"][i])
            x = ds.signal("profile/position", i)[0]
            assert -8.0 <= x[0] < -8.0 + v / ds.fs + 1e-4
            assert float(meta["span"][i]) + 5.0 - v / ds.fs - 1e-4 < x[-1] <= float(meta["span"][i]) + 5.0 + 1e-4
            for name in src.signal_names:
                assert np.array_equal(ds.signal(name, i), src.signal(name, i)[..., j0:j0 + n]), name
            assert meta["entry_time"][i] == pytest.approx(m0["entry_time"][i] - j0 / ds.fs)
            assert meta["n_samples_out"][i] == n and meta["approach_length_m"][i] == 8.0
            assert meta["bridge_name"][i] == m0["bridge_name"][i]
    # Crop again (from the derived file) and back to ragged: offsets accumulate.
    out2 = vbs.derive_dataset(out, d / "crop2.h5", approach_length_m=4.0, layout="ragged", progress=False)
    with vbs.load_dataset(rag) as src, vbs.load_dataset(out2) as ds:
        for i in range(len(ds)):
            j0, n = int(ds.metadata["crop_start_index"][i]), int(ds.lengths[i])
            assert np.array_equal(ds.signal("measured", i), src.signal("measured", i)[j0:j0 + n])
    with pytest.raises(ValueError, match="before the bridge"):
        vbs.derive_dataset(rag, d / "bad.h5", approach_length_m=50.0, progress=False)
    with pytest.raises(ValueError, match="pad_to_samples"):
        vbs.derive_dataset(rag, d / "bad.h5", layout="padded", progress=False)
    assert not (d / "bad.h5").exists()


def test_cli_derive(ragged_and_padded):
    from vehicle_bridge_signal.cli import main
    d, _, rag, _ = ragged_and_padded
    assert main(["derive", str(rag), str(d / "cli.h5"), "--approach-length-m", "10", "--exit-length-m", "10",
                 "--quiet"]) == 0
    with h5py.File(d / "cli.h5") as f:
        assert f.attrs["layout"] == "ragged" and f.attrs["crop_exit_length_m"] == 10.0
    assert main(["info", str(d / "cli.h5")]) == 0


# ---- wheel lift-off resampling --------------------------------------------------------------
def test_wheel_lift_off_resample(tmp_path):
    """A rough road at high speed lifts the wheel often; with resample no stored crossing has
    lift-off, the redraws are recorded and every sample regenerates exactly."""
    over = {**RANDOM, "dataset.bridge_catalog": [{**CATALOG[1], "roughness.class": "C"}],
            "crossing.speed": {"dist": "uniform", "low": 20.0, "high": 25.0}}
    keep = vbs.load_preset("smoke", over)
    lifts = []
    for p in plan_dataset(keep, 8, 3):
        sc, info = scenario_for(keep, p, 3)
        lifts.append(vbs.simulate(sc, info["sample_seed"]).metadata["wheel_lift_off"])
    assert any(lifts)
    cfg = vbs.with_overrides(keep, {"dataset.wheel_lift_off": "resample"})
    out = vbs.generate_dataset(cfg, 8, tmp_path / "lift.h5", seed=3, workers=1, progress=False)
    with vbs.load_dataset(out) as ds:
        meta = ds.metadata
        assert not np.any(meta["wheel_lift_off"])
        assert np.any(meta["lift_off_resamples"] > 0)
        assert np.array_equal(meta["lift_off_resamples"] > 0, np.array(lifts))
        i = int(np.flatnonzero(meta["lift_off_resamples"] > 0)[0])
        assert np.array_equal(ds.signal("measured", i), vbs.regenerate_sample(ds, i).measured.astype(np.float32))
