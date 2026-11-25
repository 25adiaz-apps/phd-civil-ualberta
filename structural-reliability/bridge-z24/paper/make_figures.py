# Figures and tables of the paper z24_early_damage_detection.tex
# Run from the paper folder:  python make_figures.py
# Uses the same data and criteria as machine-learning-project/z24_step_by_step.ipynb

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Input files (in the parent folder, bridge-z24)
FREQUENCIES_FILE = "../results/ems_natural_frequencies.csv"
ENVIRONMENT_FILE = "../cache/ems_inventory.csv"

# Output folder for the figures
FIGURES_FOLDER = "figures"
os.makedirs(FIGURES_FOLDER, exist_ok=True)

# Important dates
PREP_START = pd.Timestamp("1998-08-05")           # start of the installation of the pier lowering system

# Temperature sensors
PAVEMENT_SENSORS = ["TP1", "TP2"]
SLAB_SENSORS = ["TDT2", "TDT3", "TDS1", "TDS2"]

# Font sizes for a figure printed at the full page width
plt.rcParams["font.size"] = 8


# ---------------------------------------------------------------
# 1. Data: frequencies and temperatures of the healthy bridge
# ---------------------------------------------------------------
frequencies = pd.read_csv(FREQUENCIES_FILE, parse_dates=["start"])
temperatures = pd.read_csv(ENVIRONMENT_FILE, usecols=["code"] + PAVEMENT_SENSORS + SLAB_SENSORS)

# Error codes of the data logger (for example 3276.7) are replaced by NaN
for column in PAVEMENT_SENSORS + SLAB_SENSORS:
    temperatures.loc[temperatures[column].abs() >= 100, column] = np.nan

# Pavement temperature (T) and slab temperature (T_slab)
temperatures["T"] = temperatures[PAVEMENT_SENSORS].mean(axis=1)
temperatures["T_slab"] = temperatures[SLAB_SENSORS].mean(axis=1)

# One table with both, in time order
data = pd.merge(frequencies, temperatures[["code", "T", "T_slab"]], on="code", how="left")
data = data.sort_values("start").reset_index(drop=True)

# Healthy bridge: measurements before the damage preparation
healthy = data[data["start"] < PREP_START]
print(f"Healthy measurements: {len(healthy)}, "
      f"from {healthy['start'].min():%d.%m.%Y} to {healthy['start'].max():%d.%m.%Y}")


# ---------------------------------------------------------------
# 2. Figure: frequency of each mode vs pavement temperature
# ---------------------------------------------------------------
# One column wide: the three modes one below the other, sharing the temperature axis
fig, axes = plt.subplots(3, 1, figsize=(3.35, 5.0), sharex=True)

columns = ["f1", "f2", "f3"]
labels = ["(a)", "(b)", "(c)"]
for k in range(3):
    ax = axes[k]
    ax.plot(healthy["T"], healthy[columns[k]], ".", markersize=1.5, alpha=0.4)
    ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)     # 0 °C
    ax.set_ylabel(f"$f_{k + 1}$ (Hz)")
    ax.set_title(labels[k], loc="left")
    ax.grid(alpha=0.3)

# Only the bottom panel shows the temperature label
axes[2].set_xlabel("Pavement temperature (°C)")

fig.tight_layout()
fig.savefig(os.path.join(FIGURES_FOLDER, "frequency_vs_temperature.pdf"))
plt.close(fig)


# ---------------------------------------------------------------
# 3. Table: temperature range in each season (healthy bridge)
# ---------------------------------------------------------------
# Meteorological seasons: winter = Dec-Feb, spring = Mar-May, summer = Jun-Aug, autumn = Sep-Nov
def season_of(date):
    month = date.month
    if month in [12, 1, 2]:
        return "Winter"
    if month in [3, 4, 5]:
        return "Spring"
    if month in [6, 7, 8]:
        return "Summer"
    return "Autumn"


healthy = healthy.copy()
healthy["season"] = healthy["start"].apply(season_of)

rows = []
for season in ["Autumn", "Winter", "Spring", "Summer"]:
    part = healthy[healthy["season"] == season]
    below_zero = (part["T"] < 0).sum() / part["T"].notna().sum() * 100
    rows.append({
        "season": season,
        "from": f"{part['start'].min():%d.%m.%Y}",
        "to": f"{part['start'].max():%d.%m.%Y}",
        "n": len(part),
        "T_min": part["T"].min(),
        "T_mean": part["T"].mean(),
        "T_max": part["T"].max(),
        "Tslab_min": part["T_slab"].min(),
        "Tslab_mean": part["T_slab"].mean(),
        "Tslab_max": part["T_slab"].max(),
        "below_zero_%": below_zero,
    })

# Whole healthy period
below_zero = (healthy["T"] < 0).sum() / healthy["T"].notna().sum() * 100
rows.append({
    "season": "All",
    "from": f"{healthy['start'].min():%d.%m.%Y}",
    "to": f"{healthy['start'].max():%d.%m.%Y}",
    "n": len(healthy),
    "T_min": healthy["T"].min(),
    "T_mean": healthy["T"].mean(),
    "T_max": healthy["T"].max(),
    "Tslab_min": healthy["T_slab"].min(),
    "Tslab_mean": healthy["T_slab"].mean(),
    "Tslab_max": healthy["T_slab"].max(),
    "below_zero_%": below_zero,
})

table = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print(table.round(1).to_string(index=False))


# ---------------------------------------------------------------
# 4. Values quoted in the text: frequency change with temperature
# ---------------------------------------------------------------
for column in columns:
    near_zero = healthy[(healthy["T"] >= 0) & (healthy["T"] < 2)][column].median()
    warm = healthy[(healthy["T"] >= 28) & (healthy["T"] < 32)][column].median()
    cold = healthy[healthy["T"] < 0][column].quantile(0.99)
    print(f"{column}: median 0-2 °C = {near_zero:.2f} Hz, median 28-32 °C = {warm:.2f} Hz, "
          f"change = {(warm - near_zero) / near_zero * 100:.1f} %, "
          f"99th percentile below 0 °C = {cold:.2f} Hz ({(cold - near_zero) / near_zero * 100:.1f} %)")
