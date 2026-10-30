import warnings

import numpy as np
import pytest

import vehicle_bridge_signal as vbs


def quiet_overrides():
    """Overrides that switch off every disturbance and measurement effect."""
    return {
        "roughness.enabled": False, "engine.enabled": False, "wheel.enabled": False, "ambient.enabled": False,
        "mount.enabled": False, "sensor.white_noise.enabled": False, "sensor.quantization.enabled": False,
        "sensor.jitter.enabled": False, "sensor.anti_alias.enabled": False,
    }


@pytest.fixture
def typical():
    return vbs.load_preset("typical_car_smartphone")


@pytest.fixture
def quiet():
    return vbs.load_preset("typical_car_smartphone", quiet_overrides())


def rms(x):
    return float(np.sqrt(np.mean(np.asarray(x, float) ** 2)))


@pytest.fixture(autouse=True)
def _no_config_warnings():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", vbs.config.ConfigWarning)
        yield
