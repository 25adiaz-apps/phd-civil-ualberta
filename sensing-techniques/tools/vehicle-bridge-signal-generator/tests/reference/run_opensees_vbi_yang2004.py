"""Run the Yang et al. (2004) benchmark with the OpenSees-VBI-Simulation framework.

Usage: python run_ovbi_yang2004.py <framework_dir> <out_npz> [sampling_rate]
"""
import subprocess
import sys
import time

import numpy as np

framework_dir, out_path = sys.argv[1], sys.argv[2]
fs = float(sys.argv[3]) if len(sys.argv) > 3 else 500.0
sys.path.insert(0, framework_dir)

from opensees_vbi import (  # noqa: E402
    initial_calcs, generate_vehicle_props_pool, road_roughness_generator,
    generate_random_traffic_nodal_forces, main_vehicle_bridge)

span, E, Iz, A, mbar, n_el, speed = 25.0, 27.5e9, 0.12, 2.0, 4800.0, 250, 10.0
road_window = [-10, span + 10]
run_time = span / speed + 1.5

np.random.seed(42)
rough = road_roughness_generator(0.0, 0.05, road_window)
out = {}
for mode in ("moving_mass", "moving_load"):
    bp, bric, ts, dt = initial_calcs(span, [0, span], Iz, A, E, 0.0, mbar, n_el, run_time, fs)
    traffic = generate_random_traffic_nodal_forces(0, n_el + 1, ts)
    vp, vr = generate_vehicle_props_pool(ts, [road_window[0], 0], [0.7, 1.3], [0.7, 1.3], [0.7, 1.3],
                                         base_vehicle_name="quarter_car_simple_Yang2004",
                                         num_gen_vehicle=1, base_speed=speed)
    t0 = time.time()
    ba, va, _, iters, _ = main_vehicle_bridge(bp, vp, traffic, rough, bric, vr, ts, dt,
                                              vehicle_model=mode, plot_mode_shapes=False,
                                              print_iterations=False)
    print(mode, "done in", round(time.time() - t0, 1), "s; bridge freqs", bp["natural_freqs"][:3])
    key = "coupled" if mode == "moving_mass" else "decoupled"
    out[f"{key}_time"] = ba["time"].flatten()
    out[f"{key}_midspan_disp"] = ba["disp"][n_el // 2, :]
    out[f"{key}_vehicle_time"] = va["vehicle_1"]["time"].flatten()
    out[f"{key}_vehicle_disp"] = va["vehicle_1"]["disp"][1, :]
    out[f"{key}_vehicle_acc"] = va["vehicle_1"]["acc"][1, :]
    out[f"{key}_bridge_freqs"] = np.asarray(bp["natural_freqs"])
    out[f"{key}_vehicle_freqs"] = np.asarray(va["vehicle_1"]["natural_freqs"])

commit = subprocess.run(["git", "-C", framework_dir, "rev-parse", "HEAD"], capture_output=True,
                        text=True).stdout.strip()
np.savez_compressed(out_path, framework_commit=commit, sampling_rate=fs, span=span, E=E, Iz=Iz,
                    mbar=mbar, n_elements=n_el, speed=speed, vehicle_mass=1200.0,
                    vehicle_stiffness=5e5, rear_axle_start=road_window[0], **out)
print("saved", out_path, "commit", commit)
