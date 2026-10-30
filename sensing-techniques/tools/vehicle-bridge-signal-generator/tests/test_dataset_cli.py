"""Datasets (HDF5), reproducibility with any number of workers, group splits and the command line."""
import json
import subprocess
import sys

import h5py
import numpy as np
import pytest

import vehicle_bridge_signal as vbs
from vehicle_bridge_signal.sampling import plan_dataset, scenario_for

SMALL = {"dataset.n_bridges": 3, "dataset.n_vehicles": 3, "dataset.partitions": [0.5, 0.25, 0.25]}


@pytest.fixture(scope="module")
def small_dataset(tmp_path_factory):
    path = tmp_path_factory.mktemp("ds") / "small.h5"
    cfg = vbs.load_preset("smoke", SMALL)
    vbs.generate_dataset(cfg, 8, path, seed=5, workers=1, progress=False)
    return path, cfg


def test_simulate_is_bit_reproducible(typical):
    """Validation 10a: same configuration and seed -> identical arrays, bit for bit."""
    a, b = vbs.simulate(typical, 21), vbs.simulate(typical, 21)
    assert np.array_equal(a.measured, b.measured)
    assert all(np.array_equal(a.components[k], b.components[k]) for k in a.components)
    assert not np.array_equal(a.measured, vbs.simulate(typical, 22).measured)


def test_dataset_identical_with_one_and_several_workers(tmp_path, small_dataset):
    """Validation 10b: the same dataset with 1 and N processes, bit for bit (arrays and metadata)."""
    path1, cfg = small_dataset
    path3 = tmp_path / "w3.h5"
    vbs.generate_dataset(cfg, 8, path3, seed=5, workers=3, progress=False)
    with h5py.File(path1) as f1, h5py.File(path3) as f3:
        names = []
        f1.visititems(lambda n, o: names.append(n) if isinstance(o, h5py.Dataset) else None)
        assert len(names) > 50
        for n in names:
            a, b = f1[n][()], f3[n][()]
            assert a.dtype == b.dtype and a.shape == b.shape
            assert np.array_equal(a, b, equal_nan=True) if a.dtype.kind in "fiub" else np.array_equal(a, b), n


def test_dataset_layout_and_metadata(small_dataset):
    path, cfg = small_dataset
    with vbs.load_dataset(path) as ds:
        assert len(ds) == 8 and ds.fs == 200.0 and ds.schema_version == vbs.SCHEMA_VERSION
        assert ds.measured.dtype == np.float32 and ds.measured.shape == (8, 2048)
        assert ds.measured.chunks == (1, 2048)
        assert ds.component_keys == vbs.component_keys(3)
        assert ds.contact("total").shape == (8, 1, 2048)
        assert ds.file["on_bridge_mask"].dtype == np.uint8
        attrs = ds.attrs
        assert len(attrs["git_commit"]) >= 7 and attrs["base_seed"] == 5
        assert ds.config["dataset"]["n_bridges"] == 3
        meta = ds.metadata
        for key in ("bridge_id", "vehicle_id", "sample_seed", "mass_ratio", "roughness_class", "roughness_gd_n0",
                    "engine_rpm_mean", "bridge_frequencies", "vehicle_frequencies", "roughness_enabled",
                    "param.crossing.speed", "param.bridge.damping_ratio", "min_contact_force_ratio"):
            assert key in meta, key
        assert meta["bridge_frequencies"].shape == (8, 6)
        assert meta["param.bridge.damping_ratio"].shape == (8, 6)
        frame = ds.metadata_frame()
        assert len(frame) == 8 and "bridge_frequencies_1" in frame
        sample = ds.sample(3)
        total = np.sum([v for v in sample["components"].values()], axis=0)
        assert np.max(np.abs(sample["measured"] - total)) < 1e-6 * np.sqrt(np.mean(sample["measured"] ** 2))
        stack = ds.stack_components([0, 2])
        assert stack.shape == (2, len(ds.component_keys) - 1, 2048)
        assert np.all(np.diff(np.flatnonzero(sample["on_bridge_mask"])) == 1)   # one contiguous crossing


def test_dataset_sample_equals_fixed_simulation(small_dataset):
    """Sample i of a dataset is exactly simulate(scenario_i, seed_i) (cast to float32)."""
    path, cfg = small_dataset
    plan = plan_dataset(cfg, 8, 5)[6]
    scenario, _ = scenario_for(cfg, plan, 5)
    res = vbs.simulate(scenario, plan.sample_seed)
    with vbs.load_dataset(path) as ds:
        assert np.array_equal(ds.measured[6], res.measured.astype(np.float32))
        regen = vbs.regenerate_sample(ds, 6)
    assert np.array_equal(regen.measured, res.measured)


def test_group_split_has_no_leakage(small_dataset):
    path, _ = small_dataset
    with vbs.load_dataset(path) as ds:
        meta = ds.metadata
        split = vbs.group_split(ds)                       # uses the stored partitions
        assert split["discarded"].size == 0
        assert sorted(np.concatenate([split[k] for k in ("train", "val", "test")]).tolist()) == list(range(8))
        groups = vbs.group_split(ds, method="groups", seed=1)
    for s in (split, groups):
        for a in ("train", "val", "test"):
            for b in ("train", "val", "test"):
                if a < b:
                    assert not set(meta["bridge_id"][s[a]]) & set(meta["bridge_id"][s[b]])
                    assert not set(meta["vehicle_id"][s[a]]) & set(meta["vehicle_id"][s[b]])


def test_output_path_guards(tmp_path, small_dataset):
    path, cfg = small_dataset
    with pytest.raises(FileExistsError):
        vbs.generate_dataset(cfg, 2, path, progress=False)
    with pytest.raises(ValueError, match="tool folder"):
        vbs.generate_dataset(cfg, 2, vbs.config.TOOL_ROOT / "data" / "x.h5", progress=False)
    with pytest.raises(ValueError, match=".h5"):
        vbs.generate_dataset(cfg, 2, tmp_path / "x.csv", progress=False)
    with pytest.raises(ValueError):
        vbs.generate_dataset(cfg, 0, tmp_path / "x.h5", progress=False)


def test_failed_generation_leaves_no_file(tmp_path):
    """A sample that fails stops the run with its index and seed, and no (partial) file remains."""
    cfg = vbs.load_preset("smoke", {"bridge.span": 60.0, "crossing.speed": 4.0})   # never fits
    out = tmp_path / "bad.h5"
    with pytest.raises(RuntimeError, match="Sample 0"):
        vbs.generate_dataset(cfg, 3, out, progress=False)
    assert not out.exists() and not (tmp_path / "bad.h5.partial").exists()


def test_n_samples_from_config(tmp_path):
    cfg = vbs.load_preset("smoke", {**SMALL, "dataset.n_samples": 2})
    out = vbs.generate_dataset(cfg, None, tmp_path / "n.h5", progress=False)
    with vbs.load_dataset(out) as ds:
        assert len(ds) == 2


def _cli(*args, cwd=None):
    return subprocess.run([sys.executable, "-m", "vehicle_bridge_signal", *args], capture_output=True, text=True,
                          cwd=cwd, timeout=600)


def test_cli_presets_check_and_errors():
    out = _cli("presets")
    assert out.returncode == 0 and "default_dataset" in out.stdout and "smoke" in out.stdout
    out = _cli("check", "--config", "smoke")
    assert out.returncode == 0 and "crossing.speed" in out.stdout
    out = _cli("check", "--config", "smoke", "--set", "crossing.sped=3")
    assert out.returncode == 2 and "Did you mean" in out.stderr
    out = _cli("simulate", "--config", "smoke")
    assert out.returncode == 2 and "--sample" in out.stderr and "crossing.speed" in out.stderr


def test_cli_simulate_plot_and_save(tmp_path):
    out = _cli("simulate", "--config", "typical_car_smartphone", "--seed", "3", "--set", "crossing.speed=12",
               "--plot", str(tmp_path / "fig.png"), "--save", str(tmp_path / "res.npz"))
    assert out.returncode == 0, out.stderr
    assert (tmp_path / "fig.png").stat().st_size > 10_000 and (tmp_path / "fig_psd.png").is_file()
    data = np.load(tmp_path / "res.npz")
    assert json.loads(str(data["metadata_json"]))["speed"] == 12.0
    ref = vbs.simulate(vbs.load_preset("typical_car_smartphone", {"crossing.speed": 12.0}), 3)
    assert np.array_equal(data["measured"], ref.measured)
    out = _cli("simulate", "--config", "smoke", "--sample", "--seed", "4")
    assert out.returncode == 0 and "identity" in out.stdout


def test_cli_generate_preview_and_dataset(tmp_path):
    out = subprocess.run([sys.executable, "-m", "vehicle_bridge_signal.generate", "--config", "smoke", "--preview", "2",
                          "--plot", str(tmp_path / "prev.png")], capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stderr
    assert (tmp_path / "prev.png").is_file() and not list(tmp_path.glob("*.h5"))
    out = subprocess.run([sys.executable, "-m", "vehicle_bridge_signal.generate", "--config", "smoke", "--n-samples", "3",
                          "--out", str(tmp_path / "d.h5"), "--workers", "2", "--quiet", "--set", "dataset.n_bridges=3",
                          "--set", "dataset.n_vehicles=3"], capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stderr
    info = _cli("info", str(tmp_path / "d.h5"))
    assert info.returncode == 0 and "n=3" in info.stdout
