"""
Project Phase 3 starter code: modal features for k-means anomaly detection
==========================================================================

Course: Sensing Techniques and Data Analytics (CIV E 789), Fall 2026
Instructor: Mohammad Talebi-Kalaleh, Ph.D., University of Alberta

This file only loads the data and builds the feature matrices. The k-means
detector, the threshold, the evaluation and every figure are your work.

What it gives you
-----------------
load_modal_data(json_path, sensors)   frequencies and mode shapes at YOUR sensors
split_healthy(n_records, seed)        indices of the fit / calibration / test records
calculate_mac(phi_ref, phi_test)      Modal Assurance Criterion of two mode shapes
build_features(data, fit_idx)         six features per record: f1, f2, f3, MAC1, MAC2, MAC3

Minimal use
-----------
    from phase3_kmeans_starter import *

    GROUP = 1                                         # your group number
    data = load_modal_data('phase3_modal_data.json', SENSOR_GROUPS[GROUP])
    fit_idx, cal_idx, test_idx = split_healthy(200, seed=GROUP)
    X = build_features(data, fit_idx)

    X_fit  = X['intact'][fit_idx]                     # 100 x 6, fit the scaler and k-means
    X_cal  = X['intact'][cal_idx]                     #  50 x 6, set the threshold
    X_test = X['intact'][test_idx]                    #  50 x 6, healthy test records
    X_d1, X_d2 = X['damage1'], X['damage2']           #  50 x 6 each, test only

Requires numpy only. Run this file directly to print a summary for your group.
"""

import json

import numpy as np

# Sensor assignments from Phase 1, Table 2 (sensor S1 is index 0 in every mode shape list)
SENSOR_GROUPS = {
    1: [1, 2, 5, 7, 9, 13, 16, 19, 22],
    2: [2, 3, 6, 8, 10, 14, 17, 20, 23],
    3: [3, 4, 7, 9, 11, 15, 18, 21, 24],
    4: [1, 3, 8, 10, 12, 16, 19, 22, 24],
    5: [3, 4, 5, 7, 9, 13, 15, 18, 21],
    6: [2, 4, 6, 9, 12, 15, 18, 21, 24],
}

SCENARIOS = ('intact', 'damage1', 'damage2')
MODES = ('mode_1', 'mode_2', 'mode_3')
FEATURE_NAMES = ['f1 (Hz)', 'f2 (Hz)', 'f3 (Hz)', 'MAC1', 'MAC2', 'MAC3']


def load_modal_data(json_path, sensors):
    """Load natural frequencies and mode shapes at the given sensors.

    Parameters
    ----------
    json_path : str
        Path to phase3_modal_data.json.
    sensors : list of int
        Your nine sensor numbers, 1 to 24 (for example SENSOR_GROUPS[1]).

    Returns
    -------
    data : dict
        data[scenario]['freq']   array (n_records, 3), natural frequencies in Hz
        data[scenario]['shapes'] array (n_records, 3, n_sensors), mode shapes at your sensors
        for scenario in 'intact' (200 records), 'damage1' (50) and 'damage2' (50).
    """
    with open(json_path, 'r') as f:
        raw = json.load(f)['scenarios']

    cols = [s - 1 for s in sensors]                   # sensor S_k is at list index k - 1
    data = {}
    for scenario in SCENARIOS:
        block = raw[scenario]
        n = len(block['natural_frequencies']['mode_1'])
        freq = np.column_stack([block['natural_frequencies'][m] for m in MODES])
        shapes = np.array([[np.asarray(block['mode_shapes'][m][f'est{i + 1}'])[cols] for m in MODES]
                           for i in range(n)])
        data[scenario] = {'freq': freq, 'shapes': shapes}
    return data


def split_healthy(n_records, seed, n_fit=100, n_cal=50):
    """Shuffle the healthy record indices and split them into fit, calibration and test sets.

    Use your group number as the seed so that your split can be reproduced.
    With 200 healthy records the default sizes give 100 fit, 50 calibration and 50 test records.
    """
    order = np.random.default_rng(seed).permutation(n_records)
    return order[:n_fit], order[n_fit:n_fit + n_cal], order[n_fit + n_cal:]


def calculate_mac(phi_ref, phi_test):
    """Modal Assurance Criterion between two mode shape vectors (1 = same shape, 0 = orthogonal)."""
    return np.dot(phi_ref, phi_test) ** 2 / (np.dot(phi_ref, phi_ref) * np.dot(phi_test, phi_test))


def build_features(data, fit_idx):
    """Build the six features of every record: f1, f2, f3, MAC1, MAC2, MAC3.

    The MAC reference for each mode is the mean mode shape of the healthy FIT records only,
    so nothing from the calibration, test or damaged records enters the reference.

    Returns
    -------
    X : dict
        X[scenario] is an array (n_records, 6) with columns in the order of FEATURE_NAMES.
    """
    reference = data['intact']['shapes'][fit_idx].mean(axis=0)      # (3 modes, n_sensors)
    X = {}
    for scenario in SCENARIOS:
        shapes = data[scenario]['shapes']
        mac = np.array([[calculate_mac(reference[m], shapes[i, m]) for m in range(3)]
                        for i in range(len(shapes))])
        X[scenario] = np.column_stack([data[scenario]['freq'], mac])
    return X


if __name__ == '__main__':
    GROUP = 1                                         # replace with your group number
    data = load_modal_data('phase3_modal_data.json', SENSOR_GROUPS[GROUP])
    fit_idx, cal_idx, test_idx = split_healthy(len(data['intact']['freq']), seed=GROUP)
    X = build_features(data, fit_idx)

    print(f'Group {GROUP}, sensors {SENSOR_GROUPS[GROUP]}')
    print(f'Healthy records: {len(fit_idx)} fit, {len(cal_idx)} calibration, {len(test_idx)} test')
    for scenario in SCENARIOS:
        print(f'{scenario:8s} feature matrix {X[scenario].shape}')
    print('Feature means of the healthy fit records:')
    for name, value in zip(FEATURE_NAMES, X['intact'][fit_idx].mean(axis=0)):
        print(f'  {name:8s} {value:.4f}')
