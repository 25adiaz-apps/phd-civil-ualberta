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


# ---------------------------------------------------------------
# 5. Figure: sketch of the bridge and position of the sensors
# ---------------------------------------------------------------
# Schematic, not to scale. Positions are approximate, read from the drawings of the
# EMS documentation (Figs. 27, 37, 38 and 39 of datos/Z24-EMS-readme.pdf and datos/C11-Appendices.pdf)
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon, Rectangle

# Geometry (m): x = 0 at the Utzenstorf abutment and x = 58 at the Koppigen abutment
SUPPORTS = [0.0, 14.0, 44.0, 58.0]   # abutment, pier, pier, abutment
DECK_START = -2.7                    # the deck goes 2.70 m beyond each abutment
DECK_END = 60.7
HALF_WIDTH = 4.3                     # deck width = 8.60 m
DECK_DEPTH = 1.1
PIER_HEIGHT = 4.5

# Cross-sections with temperature sensors: name -> approximate x position (m)
TEMPERATURE_SECTIONS = {"S1": 8.4, "S2": 29.0, "S3": 51.0}

# Deck accelerometers with data: channel -> (x, y, direction)
# y = 0 on the deck axis, y > 0 towards the north edge; "V" = vertical, "L" = longitudinal, "T" = transverse
DECK_ACCELEROMETERS = {
    "05": (6.3, 2.6, "V"),
    "10": (10.7, 2.6, "V"),
    "06": (9.7, 0.0, "L"),
    "16": (14.0, 0.9, "T"),
    "07": (48.1, 2.6, "V"),
    "12": (52.3, 2.6, "V"),
    "14": (55.4, 0.9, "T"),
}
NOT_USED_CHANNELS = ["10"]           # constant signal in 60 % of the hours, excluded from FDD

USED_COLOR = "tab:red"
NOT_USED_COLOR = "gray"
STRUCTURE_COLOR = "0.85"
plt.rcParams["font.size"] = 7


# Draws an out-of-plane accelerometer (circle with a dot) or a temperature sensor (filled circle)
def draw_out_of_plane(ax, x, y, color):
    ax.plot(x, y, "o", markersize=5, markerfacecolor="white", markeredgecolor=color, markeredgewidth=0.9)
    ax.plot(x, y, "o", markersize=1.5, color=color)


# Draws an in-plane accelerometer as a double arrow along its direction
def draw_in_plane(ax, x, y, dx, dy, color):
    ax.annotate("", xy=(x + dx, y + dy), xytext=(x - dx, y - dy),
                arrowprops=dict(arrowstyle="<|-|>", color=color, linewidth=0.9, mutation_scale=6))


fig, axes = plt.subplots(3, 1, figsize=(3.35, 4.9), gridspec_kw={"height_ratios": [1.0, 1.0, 1.35]})

# (a) Elevation: spans, piers and temperature cross-sections
ax = axes[0]
ground = -DECK_DEPTH - PIER_HEIGHT
ax.add_patch(Rectangle((DECK_START, -DECK_DEPTH), DECK_END - DECK_START, DECK_DEPTH,
                       facecolor=STRUCTURE_COLOR, edgecolor="black", linewidth=0.6))
for x in SUPPORTS:
    ax.add_patch(Rectangle((x - 0.5, ground), 1.0, PIER_HEIGHT, facecolor="0.6", edgecolor="black", linewidth=0.4))
ax.plot([DECK_START - 1, DECK_END + 1], [ground, ground], color="black", linewidth=0.6)

# Span lengths
for k in range(3):
    left = SUPPORTS[k]
    right = SUPPORTS[k + 1]
    ax.annotate("", xy=(left, 1.4), xytext=(right, 1.4),
                arrowprops=dict(arrowstyle="<->", linewidth=0.5, mutation_scale=5))
    ax.text((left + right) / 2, 1.8, f"{right - left:.0f} m", ha="center", va="bottom")

# Temperature cross-sections
for name, x in TEMPERATURE_SECTIONS.items():
    ax.plot([x, x], [0.6, -DECK_DEPTH - 0.6], color=USED_COLOR, linestyle="--", linewidth=0.8)
    ax.text(x, -DECK_DEPTH - 1.0, name, color=USED_COLOR, ha="center", va="top")

ax.text(DECK_START, ground - 0.4, "Utzenstorf", ha="left", va="top")
ax.text(DECK_END, ground - 0.4, "Koppigen", ha="right", va="top")
ax.set_xlim(DECK_START - 1.5, DECK_END + 1.5)
ax.set_ylim(ground - 2.2, 3.4)
ax.set_title("(a) Elevation", loc="left")
ax.axis("off")

# (b) Plan view of the deck: accelerometers
ax = axes[1]
ax.add_patch(Rectangle((DECK_START, -HALF_WIDTH), DECK_END - DECK_START, 2 * HALF_WIDTH,
                       facecolor=STRUCTURE_COLOR, edgecolor="black", linewidth=0.6))
for x in SUPPORTS:
    ax.plot([x, x], [-HALF_WIDTH, HALF_WIDTH], color="black", linestyle=":", linewidth=0.6)

for channel, (x, y, direction) in DECK_ACCELEROMETERS.items():
    if channel in NOT_USED_CHANNELS:
        color = NOT_USED_COLOR
    else:
        color = USED_COLOR

    if direction == "V":
        draw_out_of_plane(ax, x, y, color)
        ax.text(x, y + 1.1, channel, ha="center", va="bottom", color=color)
    elif direction == "L":
        draw_in_plane(ax, x, y, 3.0, 0.0, color)
        ax.text(x, y - 1.0, channel, ha="center", va="top", color=color)
    else:
        draw_in_plane(ax, x, y, 0.0, 1.6, color)
        ax.text(x + 1.0, y, channel, ha="left", va="center", color=color)

ax.text(DECK_END + 0.8, HALF_WIDTH, "North", ha="left", va="top")
ax.text(DECK_END + 0.8, -HALF_WIDTH, "South", ha="left", va="bottom")
ax.set_xlim(DECK_START - 1.5, DECK_END + 7.5)
ax.set_ylim(-HALF_WIDTH - 0.8, HALF_WIDTH + 0.8)
ax.set_title("(b) Plan view of the deck", loc="left")
ax.axis("off")

# (c) Cross-section: box girder, temperature sensors and pier accelerometer
ax = axes[2]
outline = [(-4.3, 0.0), (4.3, 0.0), (4.3, -0.2), (3.0, -0.4), (3.0, -1.1),
           (-3.0, -1.1), (-3.0, -0.4), (-4.3, -0.2)]
ax.add_patch(Polygon(outline, closed=True, facecolor=STRUCTURE_COLOR, edgecolor="black", linewidth=0.6))
ax.add_patch(Rectangle((-2.65, -0.9), 2.5, 0.65, facecolor="white", edgecolor="black", linewidth=0.4))   # south cell
ax.add_patch(Rectangle((0.15, -0.9), 2.5, 0.65, facecolor="white", edgecolor="black", linewidth=0.4))    # north cell
ax.add_patch(Rectangle((-4.3, 0.0), 1.0, 0.25, facecolor="0.7", edgecolor="black", linewidth=0.4))      # south footway
ax.add_patch(Rectangle((3.3, 0.0), 1.0, 0.25, facecolor="0.7", edgecolor="black", linewidth=0.4))       # north footway
ax.add_patch(Rectangle((-3.3, 0.0), 6.6, 0.08, facecolor="0.3", edgecolor="none"))                     # pavement
ax.add_patch(Rectangle((-1.6, -3.0), 3.2, 1.9, facecolor="0.6", edgecolor="black", linewidth=0.4))      # pier

# Temperature sensors used in this study
used_sensors = {"TP": (-0.3, 0.04), "TDT": (0.45, -0.05), "TDS": (0.45, -0.21)}
for name, (x, y) in used_sensors.items():
    ax.plot(x, y, "o", markersize=3.5, color=USED_COLOR)
ax.annotate("TP (pavement)", xy=(-0.3, 0.04), xytext=(-2.6, 0.75), color=USED_COLOR, ha="center",
            arrowprops=dict(arrowstyle="-", color=USED_COLOR, linewidth=0.5))
ax.annotate("TDT, TDS (upper slab)", xy=(0.45, -0.05), xytext=(2.2, 0.75), color=USED_COLOR, ha="center",
            arrowprops=dict(arrowstyle="-", color=USED_COLOR, linewidth=0.5))

# Temperature sensors not used: footways, webs and lower slab
not_used_sensors = [(-3.8, 0.12), (3.8, 0.12), (-2.82, -0.65), (0.0, -0.6), (2.82, -0.65), (0.45, -1.0)]
for (x, y) in not_used_sensors:
    ax.plot(x, y, "o", markersize=3.5, markerfacecolor="white", markeredgecolor=NOT_USED_COLOR, markeredgewidth=0.8)

# Pier accelerometer 03 (longitudinal, out of the drawing plane)
draw_out_of_plane(ax, 1.75, -1.6, USED_COLOR)
ax.text(2.1, -1.6, "03", ha="left", va="center", color=USED_COLOR)

ax.text(-4.3, -2.2, "South", ha="left", va="center")
ax.text(4.3, -2.2, "North", ha="right", va="center")
ax.set_xlim(-4.8, 4.8)
ax.set_ylim(-3.0, 1.1)
ax.set_aspect("equal")
ax.set_title("(c) Cross-section", loc="left")
ax.axis("off")

# Legend for the three panels
handles = [
    Line2D([], [], marker=r"$\odot$", linestyle="none", markersize=7, color=USED_COLOR,
           label="Accelerometer, out of plane"),
    Line2D([], [], marker=r"$\leftrightarrow$", linestyle="none", markersize=8, color=USED_COLOR,
           label="Accelerometer, in plane"),
    Line2D([], [], marker="o", linestyle="none", markersize=3.5, color=USED_COLOR, label="Temperature sensor"),
    Line2D([], [], marker="o", linestyle="none", markersize=3.5, markerfacecolor="white",
           markeredgecolor=NOT_USED_COLOR, label="Not used in this study"),
]
fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, handletextpad=0.3, columnspacing=1.0)

fig.tight_layout(rect=(0, 0.07, 1, 1))
fig.savefig(os.path.join(FIGURES_FOLDER, "bridge_sensors.pdf"))
plt.close(fig)
