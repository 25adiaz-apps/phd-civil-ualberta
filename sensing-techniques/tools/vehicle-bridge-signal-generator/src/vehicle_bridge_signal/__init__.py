"""vehicle_bridge_signal: physics-based drive-by (vehicle-bridge interaction) signal generator.

Main API
--------
simulate(config, seed)                         one crossing with fixed parameters -> SimulationResult
sample_scenario(config, seed)                  draw one fixed scenario from a random configuration
generate_dataset(config, n_samples, out_path, seed, workers)   HDF5 dataset with random parameters
load_dataset(path)                             lazy reader (VBIDataset)
group_split(dataset, ...)                      optional leak-free split by bridge and vehicle
load_config / load_preset / list_presets / with_overrides       configuration helpers
"""
from .bridge import Bridge
from .config import (ConfigError, config_to_yaml, default_config, list_presets, load_config, load_preset,
                     with_overrides)
from .dataset import SCHEMA_VERSION, VBIDataset, generate_dataset, group_split, load_dataset, regenerate_sample
from .roughness import RoughnessProfile, class_bounds, classify, displacement_psd
from .sampling import plan_dataset, sample_scenario, scenario_for
from .simulate import SimulationError, SimulationResult, component_keys, contact_keys, simulate
from .vehicle import VehicleModel, build_vehicle, half_car, quarter_car, sprung_mass

__version__ = "1.0.0"

__all__ = [
    "simulate", "sample_scenario", "generate_dataset", "load_dataset", "group_split", "regenerate_sample",
    "load_config", "load_preset", "list_presets", "with_overrides", "default_config", "config_to_yaml",
    "plan_dataset", "scenario_for", "SimulationResult", "SimulationError", "ConfigError", "VBIDataset",
    "SCHEMA_VERSION", "component_keys", "contact_keys", "Bridge", "VehicleModel", "build_vehicle", "sprung_mass",
    "quarter_car", "half_car", "RoughnessProfile", "class_bounds", "classify", "displacement_psd", "__version__",
]
