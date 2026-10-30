"""Configuration validation and random sampling."""
import numpy as np
import pytest

import vehicle_bridge_signal as vbs
from vehicle_bridge_signal.config import ConfigError, distribution_paths, parse_override, yaml_load
from vehicle_bridge_signal.roughness import class_bounds
from vehicle_bridge_signal.sampling import plan_dataset, scenario_for


@pytest.mark.parametrize("preset", [p["name"] for p in vbs.list_presets()])
def test_every_preset_loads_and_documents_itself(preset):
    cfg = vbs.load_preset(preset)
    assert len(cfg["description"]) > 40


@pytest.mark.parametrize("preset", ["typical_car_smartphone", "bridge_only_smooth", "rough_road", "high_speed",
                                    "engine_dominant", "low_noise_sensor", "half_car_typical", "yang2004_benchmark"])
def test_fixed_presets_simulate(preset):
    res = vbs.simulate(preset, 0)
    assert res.measured.shape == (2048,) and res.identity_error() < 1e-6


def test_unknown_key_suggests_the_right_one():
    with pytest.raises(ConfigError, match="Did you mean: crossing.speed"):
        vbs.load_config({"crossing": {"sped": 10.0}})
    with pytest.raises(ConfigError, match="Unknown configuration key"):
        vbs.load_config({"bridgee": {}})


@pytest.mark.parametrize("bad, match", [
    ({"crossing": {"speed": -3.0}}, "must be > 0"),
    ({"crossing": {"speed": {"dist": "uniform", "low": 10, "high": 5}}}, "low < high"),
    ({"crossing": {"speed": {"dist": "gauss", "mean": 1}}}, "unknown distribution"),
    ({"crossing": {"speed": {"dist": "normal", "mean": 10, "std": 2}}}, "truncation"),
    ({"roughness": {"class": "Z"}}, "must be one of"),
    ({"roughness": {"class": "B", "gd_n0": 500e-6}}, "outside the limits"),
    ({"simulation": {"fs_sim": 1000.0, "fs_out": 300.0}}, "integer multiple"),
    ({"simulation": {"fs_out": {"dist": "uniform", "low": 100, "high": 200}}}, "structural"),
    ({"simulation": {"newmark_beta": 0.1}}, "unconditional stability"),
    ({"bridge": {"damping_ratio": 1.2}}, "must be < 1"),
    ({"bridge": {"damping_ratio": [0.01, {"dist": "uniform", "low": 0.0, "high": 0.1}]}}, "inside a list"),
    ({"bridge": {"n_label_modes": 7}}, "cannot exceed"),
    ({"bridge": {"f1": 3.0, "flexural_rigidity": 1e10}}, "not both"),
    ({"sensor": {"jitter": {"std_s": 0.01}}}, "0.45/fs_out"),
    ({"crossing": {"speed": 60.0}}, "Nyquist"),
    ({"engine": {"rpm_mean": 9000.0, "n_cylinders": 8}}, "fs_sim"),
    ({"crossing": {"placement": "fixed"}}, "entry_time_s"),
    ({"dataset": {"partitions": [1.0]}}, "partitions"),
    ({"vehicle": {"quarter_car": 5}}, "is a section"),
    ({"engine": {"enabled": "yes"}}, "true or false"),
])
def test_invalid_configurations_are_rejected(bad, match):
    with pytest.raises(ConfigError, match=match):
        vbs.load_config(bad)


def test_extends_cycle_is_detected(tmp_path):
    (tmp_path / "a.yaml").write_text("extends: b.yaml\n", encoding="utf-8")
    (tmp_path / "b.yaml").write_text("extends: a.yaml\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="Circular"):
        vbs.load_config(tmp_path / "a.yaml")


def test_extends_and_overrides(tmp_path):
    (tmp_path / "child.yaml").write_text("extends: typical_car_smartphone\ncrossing:\n  speed: 21.0\n",
                                         encoding="utf-8")
    cfg = vbs.load_config(tmp_path / "child.yaml", ["engine.enabled=false", "bridge.span=40"])
    assert cfg["crossing"]["speed"] == 21.0 and cfg["engine"]["enabled"] is False and cfg["bridge"]["span"] == 40
    assert cfg["vehicle"]["quarter_car"]["body_mass"] == 400.0   # inherited
    assert cfg["name"] == "child"
    with pytest.raises(ConfigError):
        vbs.with_overrides(cfg, {"crossing.speeed": 3})
    with pytest.raises(ConfigError, match="key.path=value"):
        parse_override("crossing.speed")


def test_yaml_scientific_notation_is_numeric():
    assert yaml_load("3.3e9") == 3.3e9 and yaml_load("1e-4") == 1e-4 and yaml_load("A") == "A"
    cfg = vbs.load_config({"bridge": {"flexural_rigidity": yaml_load("3.3e9"), "f1": None}})
    assert cfg["bridge"]["flexural_rigidity"] == 3.3e9


def test_simulate_lists_only_the_random_parameters_it_needs():
    cfg = vbs.load_preset("typical_car_smartphone")
    assert distribution_paths(cfg, relevant_only=True) == []
    rnd = vbs.load_preset("smoke")
    with pytest.raises(ConfigError) as err:
        vbs.simulate(rnd, 0)
    msg = str(err.value)
    assert "crossing.speed" in msg and "vehicle.half_car" not in msg and "wheel.diameter_m" not in msg


def test_sample_scenario_is_fixed_and_reproducible():
    cfg, seed = vbs.sample_scenario("smoke", 11)
    assert distribution_paths(cfg, relevant_only=True) == []
    cfg2, seed2 = vbs.sample_scenario("smoke", 11)
    assert cfg == cfg2 and seed == seed2
    res = vbs.simulate(cfg, seed)
    assert res.identity_error() < 1e-6
    assert len(cfg["bridge"]["damping_ratio"]) == cfg["bridge"]["max_modes"]
    assert len(set(cfg["bridge"]["damping_ratio"])) == cfg["bridge"]["max_modes"]  # independent per mode


def test_sampled_values_respect_ranges_and_class_limits():
    plans = plan_dataset("smoke", 40, 3)
    for plan in plans:
        cfg, info = scenario_for("smoke", plan, 3)
        assert 5 <= cfg["crossing"]["speed"] <= 25 and 15 <= cfg["bridge"]["span"] <= 60
        lo, hi = class_bounds(cfg["roughness"]["class"])
        assert lo <= cfg["roughness"]["gd_n0"] <= hi
        qc = cfg["vehicle"]["quarter_car"]
        assert 250 <= qc["body_mass"] <= 500 and 8 <= qc["wheel_frequency"] <= 15


def test_speed_is_resampled_until_the_crossing_fits():
    over = {"bridge.span": 60.0, "crossing.speed": {"dist": "uniform", "low": 3.0, "high": 25.0}}
    cfg = vbs.load_preset("smoke", over)
    for plan in plan_dataset(cfg, 20, 0):
        sc, info = scenario_for(cfg, plan, 0)
        assert (60.0 / sc["crossing"]["speed"]) + 2.0 <= 10.24 + 1e-9
    assert any(scenario_for(cfg, p, 0)[1]["speed_resamples"] > 0 for p in plan_dataset(cfg, 20, 0))
    with pytest.raises(vbs.simulate.__globals__["CrossingDoesNotFitError"]):
        scenario_for(vbs.load_preset("smoke", {"bridge.span": 60.0, "crossing.speed": 4.0}), plan_dataset(cfg, 1)[0])


def test_infeasible_vehicles_are_resampled():
    over = {"vehicle.quarter_car.unsprung_mass": {"dist": "uniform", "low": 30.0, "high": 35.0},
            "vehicle.quarter_car.body_mass": {"dist": "uniform", "low": 480.0, "high": 500.0},
            "vehicle.quarter_car.wheel_frequency": {"dist": "uniform", "low": 8.0, "high": 10.0},
            "vehicle.quarter_car.body_frequency": {"dist": "uniform", "low": 1.0, "high": 2.0}}
    cfg = vbs.load_preset("smoke", over)
    infos = [scenario_for(cfg, p, 0)[1] for p in plan_dataset(cfg, 10, 0)]
    assert any(i["vehicle_resamples"] > 0 for i in infos)


def test_changing_one_range_does_not_change_other_draws():
    a = scenario_for("smoke", plan_dataset("smoke", 5, 2)[3], 2)[0]
    cfg_b = vbs.load_preset("smoke", {"engine.rpm_mean": {"dist": "uniform", "low": 1000.0, "high": 2000.0}})
    b = scenario_for(cfg_b, plan_dataset(cfg_b, 5, 2)[3], 2)[0]
    assert a["crossing"]["speed"] == b["crossing"]["speed"] and a["bridge"]["span"] == b["bridge"]["span"]
    assert a["engine"]["rpm_mean"] != b["engine"]["rpm_mean"]


def test_bridge_and_vehicle_pools_and_partitions():
    cfg = vbs.load_preset("default_dataset", {"dataset.n_bridges": 20, "dataset.n_vehicles": 10})
    plans = plan_dataset(cfg, 200, 1)
    bid = np.array([p.bridge_id for p in plans])
    vid = np.array([p.vehicle_id for p in plans])
    part = np.array([p.partition for p in plans])
    assert len(np.unique(bid)) == 20 and len(np.unique(vid)) == 10
    assert np.bincount(part).tolist() == [140, 30, 30]
    for g in range(3):   # no bridge or vehicle is shared between partitions
        for h in range(g + 1, 3):
            assert not set(bid[part == g]) & set(bid[part == h])
            assert not set(vid[part == g]) & set(vid[part == h])
    # The same bridge id always gets the same bridge parameters.
    sc = [scenario_for(cfg, p, 1)[0] for p in plans[:60]]
    for b in np.unique(bid[:60]):
        spans = {round(s["bridge"]["span"], 12) for s, p in zip(sc, plans[:60]) if p.bridge_id == b}
        assert len(spans) == 1
    with pytest.raises(ConfigError):
        plan_dataset(vbs.load_preset("default_dataset", {"dataset.n_bridges": 2}), 10)


def test_roughness_scope_bridge_shares_the_profile():
    cfg = vbs.load_preset("smoke", {"roughness.scope": "bridge"})
    plans = plan_dataset(cfg, 30, 0)
    by_bridge = {}
    for p in plans:
        sc = scenario_for(cfg, p, 0)[0]
        by_bridge.setdefault(p.bridge_id, set()).add((sc["roughness"]["seed"], sc["roughness"]["gd_n0"]))
    assert all(len(v) == 1 for v in by_bridge.values())
    assert len({next(iter(v)) for v in by_bridge.values()}) == len(by_bridge)
