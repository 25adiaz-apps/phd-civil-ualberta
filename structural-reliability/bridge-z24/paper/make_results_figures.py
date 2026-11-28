# Figures of the Results section of the paper z24_early_damage_detection.tex
# Run from the paper folder:  python make_results_figures.py
# Repeats the calculations of machine-learning-project/z24_step_by_step.ipynb (steps 3 to 14) with the
# same data, folds, models and seeds, and prints the values used in the text and tables of the paper.

import os

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

# Fixed seed, so that any random result is always the same
SEED = 42
np.random.seed(SEED)

# Input files (in the parent folder, bridge-z24) and cached grids of the notebook
FREQUENCIES_FILE = "../results/ems_natural_frequencies.csv"
ENVIRONMENT_FILE = "../cache/ems_inventory.csv"
MLP_FILE = "../machine-learning-project/step_by_step_cache/mlp_cross_validation.csv"
FOREST_FILE = "../machine-learning-project/step_by_step_cache/forest_cross_validation.csv"

# Output folder for the figures
FIGURES_FOLDER = "figures"
os.makedirs(FIGURES_FOLDER, exist_ok=True)

# Important dates
WINTER_TEST_START = pd.Timestamp("1997-12-15")    # healthy test block in winter (two weeks)
WINTER_TEST_END = pd.Timestamp("1997-12-29")      # day after the last day
TEST_START = pd.Timestamp("1998-07-15")           # healthy test block in summer
PREP_START = pd.Timestamp("1998-08-05")           # start of the installation of the pier lowering system
DAMAGE_START = pd.Timestamp("1998-08-11 13:15")   # first pier lowering

MODES = [1, 2, 3]
N_FOLDS = 5
WINDOWS = [6, 12, 24, 48, 72, 168]
PANEL_LABELS = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]

# Widths of the figures: one column and the full page
COLUMN_WIDTH = 3.35
PAGE_WIDTH = 7.0
plt.rcParams["font.size"] = 7
plt.rcParams["axes.titlesize"] = 7


# RMSE (Eq. rmse of the paper)
def rmse(residuals):
    return np.sqrt(np.mean(residuals ** 2))


# ---------------------------------------------------------------
# 1. Data, blocks and folds (steps 1 to 3 of the notebook)
# ---------------------------------------------------------------
PAVEMENT_SENSORS = ["TP1", "TP2"]
SLAB_SENSORS = ["TDT2", "TDT3", "TDS1", "TDS2"]
frequencies = pd.read_csv(FREQUENCIES_FILE, parse_dates=["start"])
temperatures = pd.read_csv(ENVIRONMENT_FILE, usecols=["code"] + PAVEMENT_SENSORS + SLAB_SENSORS)

# Error codes of the data logger are replaced by NaN
for column in PAVEMENT_SENSORS + SLAB_SENSORS:
    temperatures.loc[temperatures[column].abs() >= 100, column] = np.nan
temperatures["T"] = temperatures[PAVEMENT_SENSORS].mean(axis=1)
temperatures["T_slab"] = temperatures[SLAB_SENSORS].mean(axis=1)

data = pd.merge(frequencies, temperatures[["code", "T", "T_slab"]], on="code", how="left")
data = data.sort_values("start").reset_index(drop=True)


# Time block of each measurement
def get_block(date):
    if date >= WINTER_TEST_START and date < WINTER_TEST_END:
        return "Healthy test, winter"
    elif date < TEST_START:
        return "Training"
    elif date < PREP_START:
        return "Healthy test, summer"
    elif date < DAMAGE_START:
        return "Preparation"
    else:
        return "Damaged test"


data["block"] = data["start"].apply(get_block)
complete = (data["f1"].notna() & data["f2"].notna() & data["f3"].notna()
            & data["T"].notna() & data["T_slab"].notna())
data["valid"] = complete & (data["block"] != "Preparation")

# Means of the last w hours (Eq. moving-average), with all the measurements in time order
for w in WINDOWS:
    data[f"T_mean_{w}h"] = data.rolling(f"{w}h", on="start")["T"].mean()
    data[f"T_slab_mean_{w}h"] = data.rolling(f"{w}h", on="start")["T_slab"].mean()

# Below-zero parts
for prefix in ["T", "T_slab"]:
    data[f"{prefix}_below_zero"] = np.minimum(data[prefix], 0)
    for w in WINDOWS:
        data[f"{prefix}_mean_{w}h_below_zero"] = np.minimum(data[f"{prefix}_mean_{w}h"], 0)

# Freezing index (Eq. freezing-index), with all the measurements in time order
freezing_index = []
index_value = 0
previous_start = None
for _, row in data.iterrows():
    if previous_start is not None and not np.isnan(row["T"]):
        hours = (row["start"] - previous_start).total_seconds() / 3600
        index_value = max(0, index_value - row["T"] * hours)
    freezing_index.append(index_value)
    previous_start = row["start"]
data["sqrt_freezing_index"] = np.sqrt(freezing_index)

# Training measurements and weekly folds
training = data[(data["block"] == "Training") & data["valid"]].copy()
first_date = training["start"].min().normalize()
week = (training["start"] - first_date).dt.days // 7
training["fold"] = week % N_FOLDS + 1
below_zero = training["T"] < 0
coldest = training["T"] < -3
print(f"Training: {len(training)} measurements, {below_zero.sum()} below 0 °C, {coldest.sum()} below -3 °C")


# ---------------------------------------------------------------
# 2. Week-blocked cross-validation of the three model types
# ---------------------------------------------------------------
def cross_validate_linear(features, mode):
    column = f"f{mode}"
    predictions = np.zeros(len(training))
    for fold in range(1, N_FOLDS + 1):
        is_validation = (training["fold"] == fold).to_numpy()
        fit_part = training[~is_validation]
        model = LinearRegression()
        model.fit(fit_part[features], fit_part[column])
        predictions[is_validation] = model.predict(training[is_validation][features])
    return predictions


def cross_validate_mean(mode):
    column = f"f{mode}"
    predictions = np.zeros(len(training))
    for fold in range(1, N_FOLDS + 1):
        is_validation = (training["fold"] == fold).to_numpy()
        predictions[is_validation] = training[~is_validation][column].mean()
    return predictions


def cross_validate_mlp(features, mode, hidden_layers, alpha):
    column = f"f{mode}"
    predictions = np.zeros(len(training))
    for fold in range(1, N_FOLDS + 1):
        is_validation = (training["fold"] == fold).to_numpy()
        fit_part = training[~is_validation]
        validation_part = training[is_validation]
        scaler = StandardScaler()
        x_fit = scaler.fit_transform(fit_part[features])
        x_validation = scaler.transform(validation_part[features])
        f_mean = fit_part[column].mean()
        f_std = fit_part[column].std()
        model = MLPRegressor(hidden_layer_sizes=hidden_layers, alpha=alpha, early_stopping=True,
                             max_iter=2000, random_state=SEED)
        model.fit(x_fit, (fit_part[column] - f_mean) / f_std)
        predictions[is_validation] = model.predict(x_validation) * f_std + f_mean
    return predictions


def cross_validate_forest(features, mode, min_samples_leaf, max_features):
    column = f"f{mode}"
    predictions = np.zeros(len(training))
    for fold in range(1, N_FOLDS + 1):
        is_validation = (training["fold"] == fold).to_numpy()
        fit_part = training[~is_validation]
        model = RandomForestRegressor(n_estimators=200, min_samples_leaf=min_samples_leaf,
                                      max_features=max_features, random_state=SEED, n_jobs=-1)
        model.fit(fit_part[features], fit_part[column])
        predictions[is_validation] = model.predict(training[is_validation][features])
    return predictions


# Input variables of the linear regression: two slopes plus the history of the given windows
def linear_features(prefix, windows):
    features = [prefix, f"{prefix}_below_zero"]
    for w in windows:
        features.append(f"{prefix}_mean_{w}h")
        features.append(f"{prefix}_mean_{w}h_below_zero")
    return features


# Inputs of the neural network and of the random forest (set "Historia térmica")
FLEXIBLE_FEATURES = ["T"]
for w in WINDOWS:
    FLEXIBLE_FEATURES.append(f"T_mean_{w}h")


# ---------------------------------------------------------------
# 3. Two-slope model and residuals near 0 °C (steps 5 and 6)
# ---------------------------------------------------------------
print("\n--- Two-slope model")
for mode in MODES:
    training[f"res_mean_f{mode}"] = training[f"f{mode}"] - cross_validate_mean(mode)
    training[f"res_2s_f{mode}"] = training[f"f{mode}"] - cross_validate_linear(["T", "T_below_zero"], mode)
    training[f"res_slab_2s_f{mode}"] = training[f"f{mode}"] - cross_validate_linear(
        ["T_slab", "T_slab_below_zero"], mode)
    for name in ["mean", "2s", "slab_2s"]:
        residuals = training[f"res_{name}_f{mode}"]
        print(f"f{mode} {name}: RMSE {rmse(residuals):.4f}, below 0 {rmse(residuals[below_zero]):.4f}, "
              f"above 0 {rmse(residuals[~below_zero]):.4f}")
    model = LinearRegression().fit(training[["T", "T_below_zero"]], training[f"f{mode}"])
    print(f"f{mode} slopes: theta0 {model.intercept_:.4f}, above 0 {model.coef_[0]:.4f}, "
          f"below 0 {model.coef_[0] + model.coef_[1]:.4f}")

# Correlation between the residual and the mean of the last w hours, near 0 °C (from -2 to 3 °C)
near_pavement = (training["T"] >= -2) & (training["T"] <= 3)
near_slab = (training["T_slab"] >= -2) & (training["T_slab"] <= 3)
correlations = {}
for prefix, residual_name, near in [("T", "2s", near_pavement), ("T_slab", "slab_2s", near_slab)]:
    for mode in MODES:
        values = []
        for w in WINDOWS:
            part = training[near]
            values.append(part[f"res_{residual_name}_f{mode}"].corr(part[f"{prefix}_mean_{w}h"]))
        correlations[(prefix, mode)] = values
        print(f"Correlation {prefix} f{mode}: " + ", ".join(f"{v:.2f}" for v in values))

# Figure: residuals of the two-slope model against T (top) and, near 0 °C, against the 72-h mean (bottom)
fig, axes = plt.subplots(2, 3, figsize=(PAGE_WIDTH, 3.6))
for i in range(3):
    mode = MODES[i]
    residuals = training[f"res_2s_f{mode}"]

    # Top: against the pavement temperature, with the mean residual of each 1 °C bin
    ax = axes[0][i]
    ax.plot(training["T"], residuals, ".", markersize=1.2, alpha=0.35, color="tab:blue")
    bin_edges = np.arange(-7, 42, 1)
    centers = []
    means = []
    for k in range(len(bin_edges) - 1):
        in_bin = (training["T"] >= bin_edges[k]) & (training["T"] < bin_edges[k + 1])
        if in_bin.sum() >= 10:
            centers.append((bin_edges[k] + bin_edges[k + 1]) / 2)
            means.append(residuals[in_bin].mean())
    ax.plot(centers, means, "-", color="black", linewidth=1.0, label="Mean of each 1 °C bin")
    ax.axhline(0, color="0.3", linewidth=0.6)
    ax.axvline(0, color="gray", linestyle="--", linewidth=0.6)
    ax.set_xlabel("Pavement temperature, $T$ (°C)")
    ax.set_ylabel(f"Residual of $f_{mode}$ (Hz)")
    ax.set_title(PANEL_LABELS[i], loc="left")
    ax.grid(alpha=0.3)

    # Bottom: measurements near 0 °C against the mean of the last 72 h
    ax = axes[1][i]
    part = training[near_pavement]
    ax.plot(part["T_mean_72h"], part[f"res_2s_f{mode}"], ".", markersize=1.5, alpha=0.5, color="tab:blue")
    ax.axhline(0, color="0.3", linewidth=0.6)
    ax.axvline(0, color="gray", linestyle="--", linewidth=0.6)
    ax.set_xlabel(r"Mean of the last 72 h, $\bar{T}_{72}$ (°C)")
    ax.set_ylabel(f"Residual of $f_{mode}$ (Hz)")
    ax.set_title(PANEL_LABELS[3 + i], loc="left")
    ax.grid(alpha=0.3)
axes[0][0].legend(fontsize=6, loc="upper right")
fig.tight_layout(pad=0.3)
fig.savefig(os.path.join(FIGURES_FOLDER, "residuals_two_slope.pdf"))
plt.close(fig)

# Figure: correlation against the window length, pavement and slab
fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 2.1))
MODE_MARKERS = {1: "o", 2: "s", 3: "^"}
MODE_COLORS = {1: "tab:blue", 2: "tab:orange", 3: "tab:green"}
for mode in MODES:
    ax.plot(WINDOWS, correlations[("T", mode)], "-", marker=MODE_MARKERS[mode], color=MODE_COLORS[mode],
            markersize=4, label=f"$f_{mode}$, pavement")
    ax.plot(WINDOWS, correlations[("T_slab", mode)], "--", marker=MODE_MARKERS[mode], color=MODE_COLORS[mode],
            markersize=4, markerfacecolor="white", label=f"$f_{mode}$, slab")
ax.set_xscale("log")
ax.set_xticks(WINDOWS)
ax.set_xticklabels([str(w) for w in WINDOWS])
ax.minorticks_off()
ax.set_xlabel("Window of the mean, $w$ (h)")
ax.set_ylabel("Correlation with the residual")
ax.axhline(0, color="0.3", linewidth=0.6)
ax.grid(alpha=0.3)
ax.legend(fontsize=5.5, ncol=2, loc="lower left")
fig.tight_layout(pad=0.3)
fig.savefig(os.path.join(FIGURES_FOLDER, "history_correlation.pdf"))
plt.close(fig)


# ---------------------------------------------------------------
# 4. Linear regression with history and freezing index (step 7)
# ---------------------------------------------------------------
print("\n--- Linear regression with history")
OPTIONS = [("Two slopes", [])]
for w in WINDOWS:
    OPTIONS.append((f"{w} h", [w]))
OPTIONS.append(("All", WINDOWS))

linear_rmse = {}
for prefix in ["T", "T_slab"]:
    for option_label, windows in OPTIONS:
        for mode in MODES:
            residuals = training[f"f{mode}"] - cross_validate_linear(linear_features(prefix, windows), mode)
            linear_rmse[(prefix, option_label, mode)] = rmse(residuals)
    print(prefix + ": " + "; ".join(
        f"{label} " + "/".join(f"{linear_rmse[(prefix, label, m)]:.4f}" for m in MODES) for label, _ in OPTIONS))

# Best option of each mode with the pavement temperature (without the two-slope model)
best_linear = {}
for mode in MODES:
    best_label = None
    for option_label, windows in OPTIONS[1:]:
        if best_label is None or linear_rmse[("T", option_label, mode)] < linear_rmse[("T", best_label, mode)]:
            best_label = option_label
            best_windows = windows
    best_linear[mode] = (best_label, best_windows)
    training[f"res_linear_f{mode}"] = training[f"f{mode}"] - cross_validate_linear(
        linear_features("T", best_windows), mode)

# Freezing index added to the best option
for mode in MODES:
    features = linear_features("T", best_linear[mode][1]) + ["sqrt_freezing_index"]
    residuals = training[f"f{mode}"] - cross_validate_linear(features, mode)
    print(f"f{mode} best '{best_linear[mode][0]}' + sqrt(I): RMSE {rmse(residuals):.4f} "
          f"(without {rmse(training[f'res_linear_f{mode}']):.4f}), below 0 {rmse(residuals[below_zero]):.4f}, "
          f"mean below -3 {residuals[coldest].mean():+.4f} "
          f"(without {training.loc[coldest, f'res_linear_f{mode}'].mean():+.4f})")

# Figure: RMSE of the linear regression with each window, pavement and slab
fig, axes = plt.subplots(3, 1, figsize=(COLUMN_WIDTH, 4.3), sharex=True)
positions = np.arange(len(OPTIONS))
for i in range(3):
    mode = MODES[i]
    ax = axes[i]
    for prefix, style, marker, label in [("T", "-", "o", "Pavement"), ("T_slab", "--", "s", "Slab")]:
        values = [linear_rmse[(prefix, option_label, mode)] for option_label, _ in OPTIONS]
        ax.plot(positions, values, style, marker=marker, markersize=3.5, color="black" if prefix == "T" else "0.5",
                markerfacecolor="black" if prefix == "T" else "white", label=label)
    ax.set_ylabel(f"RMSE of $f_{mode}$ (Hz)")
    ax.set_title(PANEL_LABELS[i], loc="left")
    ax.grid(alpha=0.3)
axes[2].set_xticks(positions)
axes[2].set_xticklabels(["Two\nslopes"] + [f"+{w} h" for w in WINDOWS] + ["+all"])
axes[2].set_xlabel("History added to the two-slope model")
axes[0].legend(fontsize=6, loc="upper right")
fig.tight_layout(pad=0.3)
fig.savefig(os.path.join(FIGURES_FOLDER, "linear_history_rmse.pdf"))
plt.close(fig)


# ---------------------------------------------------------------
# 5. Neural network and random forest (steps 8 and 9)
# ---------------------------------------------------------------
print("\n--- Neural network and random forest")
HIDDEN_LAYERS = {"10": (10,), "30": (30,), "30 y 30": (30, 30), "50 y 50": (50, 50), "100 y 100": (100, 100),
                 "30, 30 y 30": (30, 30, 30), "50, 50 y 50": (50, 50, 50)}
mlp_table = pd.read_csv(MLP_FILE, dtype={"Neuronas por capa oculta": str})
forest_table = pd.read_csv(FOREST_FILE)

for mode in MODES:
    column = f"RMSE f{mode} (Hz)"
    best = mlp_table.loc[mlp_table[column].idxmin()]
    training[f"res_mlp_f{mode}"] = training[f"f{mode}"] - cross_validate_mlp(
        FLEXIBLE_FEATURES, mode, HIDDEN_LAYERS[best["Neuronas por capa oculta"]], best["alpha"])
    best = forest_table.loc[forest_table[column].idxmin()]
    training[f"res_forest_f{mode}"] = training[f"f{mode}"] - cross_validate_forest(
        FLEXIBLE_FEATURES, mode, int(best["Mínimo por hoja"]), int(best["Variables por división"]))

# Summary of the best model of each type (table of the paper)
for mode in MODES:
    for name in ["2s", "linear", "mlp", "forest"]:
        residuals = training[f"res_{name}_f{mode}"]
        print(f"f{mode} {name}: RMSE {rmse(residuals):.4f}, below 0 {rmse(residuals[below_zero]):.4f}, "
              f"above 0 {rmse(residuals[~below_zero]):.4f}, mean below -3 {residuals[coldest].mean():+.3f}")

# Figure: mean residual in each temperature bin, best model of each type
MODEL_STYLES = [("linear", "Linear regression", "-", "o", "tab:blue"),
                ("mlp", "Neural network", "--", "s", "tab:orange"),
                ("forest", "Random forest", ":", "^", "tab:green")]
bin_edges = np.array([-7, -5, -3, -2, -1, 0, 1, 2, 3, 5, 10, 15, 20, 25, 30, 35, 41])
fig, axes = plt.subplots(3, 1, figsize=(COLUMN_WIDTH, 4.3), sharex=True)
for i in range(3):
    mode = MODES[i]
    ax = axes[i]
    for name, label, style, marker, color in MODEL_STYLES:
        residuals = training[f"res_{name}_f{mode}"]
        centers = []
        means = []
        for k in range(len(bin_edges) - 1):
            in_bin = (training["T"] >= bin_edges[k]) & (training["T"] < bin_edges[k + 1])
            if in_bin.sum() >= 10:
                centers.append(training.loc[in_bin, "T"].mean())
                means.append(residuals[in_bin].mean())
        ax.plot(centers, means, style, marker=marker, markersize=3, color=color, label=label)
    ax.axhline(0, color="0.3", linewidth=0.6)
    ax.axvline(0, color="gray", linestyle="--", linewidth=0.6)
    ax.axvline(-3, color="gray", linestyle=":", linewidth=0.6)
    ax.set_ylabel(f"Mean residual\nof $f_{mode}$ (Hz)")
    ax.set_title(PANEL_LABELS[i], loc="left")
    ax.grid(alpha=0.3)
axes[2].set_xlabel("Pavement temperature, $T$ (°C)")
axes[0].legend(fontsize=6, loc="lower right")
fig.tight_layout(pad=0.3)
fig.savefig(os.path.join(FIGURES_FOLDER, "binned_residuals_models.pdf"))
plt.close(fig)


# ---------------------------------------------------------------
# 6. One-standard-error rule (step 10)
# ---------------------------------------------------------------
print("\n--- One-standard-error rule")


def count_mlp_parameters(n_inputs, hidden_layers):
    sizes = [n_inputs] + list(hidden_layers) + [1]
    total = 0
    for i in range(len(sizes) - 1):
        total = total + sizes[i] * sizes[i + 1] + sizes[i + 1]
    return total


# All the candidates: type order (1 linear, 2 network, 3 forest) and size inside the type
candidates = []
for option_label, windows in OPTIONS:
    candidate = {"label": f"Linear, {option_label}", "type": "Linear regression", "order": 1,
                 "size": len(linear_features("T", windows)) + 1, "windows": windows}
    for mode in MODES:
        candidate[f"rmse_f{mode}"] = linear_rmse[("T", option_label, mode)]
    candidates.append(candidate)
for _, table_row in mlp_table.iterrows():
    layers_label = table_row["Neuronas por capa oculta"]
    candidate = {"label": f"Network {layers_label}, alpha {table_row['alpha']:g}", "type": "Neural network",
                 "order": 2, "size": count_mlp_parameters(len(FLEXIBLE_FEATURES), HIDDEN_LAYERS[layers_label]),
                 "layers": HIDDEN_LAYERS[layers_label], "alpha": table_row["alpha"]}
    for mode in MODES:
        candidate[f"rmse_f{mode}"] = table_row[f"RMSE f{mode} (Hz)"]
    candidates.append(candidate)
for _, table_row in forest_table.iterrows():
    leaf = int(table_row["Mínimo por hoja"])
    candidate = {"label": f"Forest, leaf {leaf}, {int(table_row['Variables por división'])} variables",
                 "type": "Random forest", "order": 3, "size": -leaf, "leaf": leaf,
                 "max_features": int(table_row["Variables por división"])}
    for mode in MODES:
        candidate[f"rmse_f{mode}"] = table_row[f"RMSE f{mode} (Hz)"]
    candidates.append(candidate)
candidates = pd.DataFrame(candidates)


def predict_candidate(candidate, mode):
    if candidate["type"] == "Linear regression":
        return cross_validate_linear(linear_features("T", candidate["windows"]), mode)
    elif candidate["type"] == "Neural network":
        return cross_validate_mlp(FLEXIBLE_FEATURES, mode, candidate["layers"], candidate["alpha"])
    else:
        return cross_validate_forest(FLEXIBLE_FEATURES, mode, int(candidate["leaf"]), int(candidate["max_features"]))


chosen_models = {}
thresholds = {}
for mode in MODES:
    column = f"rmse_f{mode}"
    best = candidates.loc[candidates[column].idxmin()]
    residuals = training[f"f{mode}"] - predict_candidate(best, mode)
    fold_errors = []
    for fold in range(1, N_FOLDS + 1):
        fold_errors.append(rmse(residuals[training["fold"] == fold]))
    s_min = np.std(fold_errors, ddof=1)
    threshold = best[column] + s_min / np.sqrt(N_FOLDS)
    under_threshold = candidates[candidates[column] <= threshold].sort_values(["order", "size", column])
    chosen = under_threshold.iloc[0]
    chosen_models[mode] = chosen
    thresholds[mode] = threshold
    n_types = under_threshold["type"].value_counts().to_dict()
    print(f"f{mode}: best '{best['label']}' {best[column]:.4f}, s_min {s_min:.4f}, threshold {threshold:.4f}, "
          f"under threshold {len(under_threshold)} {n_types}; chosen '{chosen['label']}' {chosen[column]:.4f}")

# Figure: validation RMSE of every candidate, threshold and chosen model
TYPE_STYLES = [("Linear regression", "o", "tab:blue"), ("Neural network", "s", "tab:orange"),
               ("Random forest", "^", "tab:green")]
fig, axes = plt.subplots(3, 1, figsize=(COLUMN_WIDTH, 4.6), sharex=True)
for i in range(3):
    mode = MODES[i]
    column = f"rmse_f{mode}"
    ax = axes[i]
    # Candidates from the simplest (left) to the most complex (right), in the order of the rule
    ordered = candidates.sort_values(["order", "size", column]).reset_index()
    for type_label, marker, color in TYPE_STYLES:
        type_rows = ordered[ordered["type"] == type_label]
        ax.plot(type_rows.index, type_rows[column], marker, markersize=3, color=color, markerfacecolor="none",
                label=type_label)
    chosen_position = ordered.index[ordered["index"] == chosen_models[mode].name][0]
    ax.plot(chosen_position, chosen_models[mode][column], "*", markersize=10, color="tab:red", label="Chosen")
    ax.axhline(thresholds[mode], color="black", linestyle="--", linewidth=0.8, label="Threshold")
    ax.set_ylabel(f"RMSE of $f_{mode}$ (Hz)")
    ax.set_title(PANEL_LABELS[i], loc="left")
    ax.grid(alpha=0.3)
    # Zoom on the best candidates: the worst linear options are left out of the axis
    low = ordered[column].min()
    ax.set_ylim(low - 0.15 * (thresholds[mode] - low) - 0.001, low + 4 * (thresholds[mode] - low) + 0.004)
axes[2].set_xlabel("Candidate, from the simplest to the most complex")
handles, legend_labels = axes[0].get_legend_handles_labels()
fig.legend(handles, legend_labels, fontsize=6, ncol=3, loc="upper center", frameon=False)
fig.tight_layout(pad=0.3)
fig.subplots_adjust(top=0.89)
fig.savefig(os.path.join(FIGURES_FOLDER, "one_standard_error_rule.pdf"))
plt.close(fig)


# ---------------------------------------------------------------
# 7. Final models and residuals of every block (step 11)
# ---------------------------------------------------------------
print("\n--- Residuals of the chosen models")
test = data[data["valid"] & (data["block"] != "Training")].copy()

# f1: neural network with two layers of 30 and alpha = 0.01
scaler = StandardScaler()
x_training = scaler.fit_transform(training[FLEXIBLE_FEATURES])
f1_mean = training["f1"].mean()
f1_std = training["f1"].std()
network = MLPRegressor(hidden_layer_sizes=(30, 30), alpha=0.01, early_stopping=True, max_iter=2000,
                       random_state=SEED)
network.fit(x_training, (training["f1"] - f1_mean) / f1_std)
test["predicted_f1"] = network.predict(scaler.transform(test[FLEXIBLE_FEATURES])) * f1_std + f1_mean

# f2: linear regression with the one-week window
F2_FEATURES = linear_features("T", [168])
linear_model = LinearRegression().fit(training[F2_FEATURES], training["f2"])
test["predicted_f2"] = linear_model.predict(test[F2_FEATURES])

# f3: random forest with leaves of at least 20 and 7 variables per split
forest = RandomForestRegressor(n_estimators=200, min_samples_leaf=20, max_features=7, random_state=SEED, n_jobs=-1)
forest.fit(training[FLEXIBLE_FEATURES], training["f3"])
test["predicted_f3"] = forest.predict(test[FLEXIBLE_FEATURES])

for mode in MODES:
    test[f"residual_f{mode}"] = test[f"f{mode}"] - test[f"predicted_f{mode}"]
    training[f"residual_f{mode}"] = training[f"f{mode}"] - predict_candidate(chosen_models[mode], mode)

columns = ["start", "block", "T", "residual_f1", "residual_f2", "residual_f3"]
residuals_all = pd.concat([training[columns], test[columns]]).sort_values("start").reset_index(drop=True)

BLOCKS = ["Training", "Healthy test, winter", "Healthy test, summer", "Damaged test"]
for block in BLOCKS:
    block_rows = residuals_all[residuals_all["block"] == block]
    print(f"{block}: {len(block_rows)} measurements; " + "; ".join(
        f"f{m} mean {block_rows[f'residual_f{m}'].mean():+.4f} sd {block_rows[f'residual_f{m}'].std():.4f}"
        for m in MODES))

# Summer test without the identification errors of 15 and 16.07.1998
summer = residuals_all[residuals_all["block"] == "Healthy test, summer"]
summer_clean = summer[summer["start"] >= pd.Timestamp("1998-07-17")]
print("Healthy test, summer, from 17.07: " + "; ".join(
    f"f{m} sd {summer_clean[f'residual_f{m}'].std():.4f}" for m in MODES))


# Shaded background of the time blocks, with a hatch for the tests so that they are not told apart only by color
def shade_blocks(ax, start, end):
    shades = [(WINTER_TEST_START, WINTER_TEST_END, "tab:green", "...."),
              (TEST_START, PREP_START, "tab:green", "...."),
              (PREP_START, DAMAGE_START, "0.5", "////"),
              (DAMAGE_START, end, "tab:red", "xxxx")]
    for shade_start, shade_end, color, hatch in shades:
        if shade_end > start:
            ax.axvspan(max(shade_start, start), shade_end, facecolor=color, alpha=0.15, hatch=hatch,
                       edgecolor="0.6", linewidth=0)


SHADE_HANDLES = [Patch(facecolor="white", edgecolor="0.6", label="Training"),
                 Patch(facecolor="tab:green", alpha=0.15, hatch="....", edgecolor="0.6", label="Test, healthy"),
                 Patch(facecolor="0.5", alpha=0.15, hatch="////", edgecolor="0.6", label="Preparation (excluded)"),
                 Patch(facecolor="tab:red", alpha=0.15, hatch="xxxx", edgecolor="0.6", label="Test, damaged")]

# Figure: residuals of the chosen models from 01.07.1998
ZOOM_START = pd.Timestamp("1998-07-01")
DATA_END = residuals_all["start"].max() + pd.Timedelta(days=1)
zoom = residuals_all[residuals_all["start"] >= ZOOM_START]
fig, axes = plt.subplots(3, 1, figsize=(COLUMN_WIDTH, 4.2), sharex=True)
for i in range(3):
    mode = MODES[i]
    ax = axes[i]
    shade_blocks(ax, ZOOM_START, DATA_END)
    ax.plot(zoom["start"], zoom[f"residual_f{mode}"], ".", markersize=1.5, color="black")
    ax.axhline(0, color="0.3", linewidth=0.6)
    ax.set_ylabel(f"Residual of $f_{mode}$ (Hz)")
    ax.set_title(PANEL_LABELS[i], loc="left")
    ax.grid(alpha=0.3)
axes[2].set_xlim(ZOOM_START, DATA_END)
axes[2].xaxis.set_major_locator(mdates.DayLocator(bymonthday=[1, 15]))
axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%d.%m"))
fig.legend(handles=SHADE_HANDLES, loc="upper center", ncol=2, fontsize=6, frameon=False)
fig.align_ylabels(axes)
fig.tight_layout(pad=0.3)
fig.subplots_adjust(top=0.9, left=0.17)
fig.savefig(os.path.join(FIGURES_FOLDER, "test_residuals.pdf"))
plt.close(fig)


# ---------------------------------------------------------------
# 8. Mahalanobis distance, smoothing and limit (steps 12 and 13)
# ---------------------------------------------------------------
print("\n--- Detection")
RESIDUAL_COLUMNS = ["residual_f1", "residual_f2", "residual_f3"]
training_residuals = training[RESIDUAL_COLUMNS]
print("Standard deviations: " + ", ".join(f"{v:.3f}" for v in training_residuals.std()))
print("Correlations:\n" + training_residuals.corr().round(3).to_string())
for column in RESIDUAL_COLUMNS:
    values = training_residuals[column]
    beyond = (np.abs(values - values.mean()) > 3 * values.std()).mean() * 100
    print(f"{column}: {beyond:.2f} % beyond 3 standard deviations")

residual_mean = training_residuals.mean().to_numpy()
inverse_covariance = np.linalg.inv(np.cov(training_residuals.to_numpy(), rowvar=False))
distances = []
for _, row in residuals_all.iterrows():
    difference = row[RESIDUAL_COLUMNS].to_numpy(dtype=float) - residual_mean
    distances.append(difference @ inverse_covariance @ difference)
residuals_all["D2"] = distances
for block in BLOCKS:
    values = residuals_all.loc[residuals_all["block"] == block, "D2"]
    print(f"{block}: D2 mean {values.mean():.2f}, median {values.median():.2f}, max {values.max():.2f}")


# EWMA (Eq. ewma), measurement by measurement in time order, starting with Z_1 = D_1^2
def ewma(values, smoothing):
    smoothed = []
    previous = None
    for value in values:
        if previous is None:
            current = value
        else:
            current = smoothing * value + (1 - smoothing) * previous
        smoothed.append(current)
        previous = current
    return smoothed


# Number of alarm episodes: alarms less than 24 h apart are one episode
def count_episodes(alarm, starts):
    episodes = 0
    last_alarm = None
    for is_alarm, start in zip(alarm, starts):
        if is_alarm:
            if last_alarm is None or start - last_alarm > pd.Timedelta(hours=24):
                episodes = episodes + 1
            last_alarm = start
    return episodes


LAMBDAS = [1, 0.3, 0.1, 0.05]
PERCENTILES = [90, 95, 97.5, 99]
for smoothing in LAMBDAS:
    residuals_all[f"Z_{smoothing}"] = ewma(residuals_all["D2"], smoothing)

training_rows = residuals_all[residuals_all["block"] == "Training"]
healthy_level = training_rows["D2"].median()
reference_level = np.percentile(training_rows["D2"], 99)
print(f"Reference change: D2 from {healthy_level:.2f} to {reference_level:.1f}")
trade_off = []
for smoothing in LAMBDAS:
    for percentile in PERCENTILES:
        z_training = training_rows[f"Z_{smoothing}"]
        limit = np.percentile(z_training, percentile)
        episodes = count_episodes(z_training > limit, training_rows["start"])
        z = healthy_level
        hours = 0
        while z <= limit and hours < 1000:
            z = smoothing * reference_level + (1 - smoothing) * z
            hours = hours + 1
        if hours == 1000:
            hours = np.nan
        trade_off.append({"lambda": smoothing, "p": percentile, "limit": limit, "episodes": episodes,
                          "hours": hours})
trade_off = pd.DataFrame(trade_off)
print(trade_off.to_string(index=False))

# Figure: false-alarm episodes against detection hours for each lambda and percentile
fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 2.4))
LAMBDA_STYLES = {1: ("o", "tab:gray"), 0.3: ("s", "tab:blue"), 0.1: ("^", "tab:red"), 0.05: ("D", "tab:green")}
for smoothing in LAMBDAS:
    rows = trade_off[(trade_off["lambda"] == smoothing) & trade_off["hours"].notna()]
    marker, color = LAMBDA_STYLES[smoothing]
    label = r"$\lambda$ = 1 (no smoothing)" if smoothing == 1 else rf"$\lambda$ = {smoothing:g}"
    ax.plot(rows["hours"], rows["episodes"], "-", marker=marker, markersize=4, color=color, label=label)
    for _, row in rows.iterrows():
        ax.annotate(f"{row['p']:g}", (row["hours"], row["episodes"]), textcoords="offset points",
                    xytext=(3, 2), fontsize=5, color=color)
chosen_row = trade_off[(trade_off["lambda"] == 0.1) & (trade_off["p"] == 95)].iloc[0]
ax.plot(chosen_row["hours"], chosen_row["episodes"], "o", markersize=10, markerfacecolor="none",
        markeredgecolor="black", label="Chosen")
ax.set_xlabel("Hours to detect the reference change")
ax.set_ylabel("False-alarm episodes\nin training (246 days)")
ax.grid(alpha=0.3)
ax.legend(fontsize=5.5, loc="upper right")
fig.tight_layout(pad=0.3)
fig.savefig(os.path.join(FIGURES_FOLDER, "smoothing_limit_trade_off.pdf"))
plt.close(fig)


# ---------------------------------------------------------------
# 9. Evaluation with the test blocks (step 14)
# ---------------------------------------------------------------
residuals_all["Z"] = residuals_all["Z_0.1"]
training_rows = residuals_all[residuals_all["block"] == "Training"]
Z_LIMIT = np.percentile(training_rows["Z"], 95)
residuals_all["alarm"] = residuals_all["Z"] > Z_LIMIT
print(f"\nLimit Z_lim = {Z_LIMIT:.2f}")

for block in ["Healthy test, winter", "Healthy test, summer"]:
    block_rows = residuals_all[residuals_all["block"] == block]
    alarm_rows = block_rows[block_rows["alarm"]]
    print(f"{block}: {len(alarm_rows)} of {len(block_rows)} in alarm ({len(alarm_rows) / len(block_rows) * 100:.1f} %), "
          f"{count_episodes(block_rows['alarm'], block_rows['start'])} episodes, "
          f"days {sorted(set(alarm_rows['start'].dt.strftime('%d.%m')))}, max Z {block_rows['Z'].max():.1f}")

damage_rows = residuals_all[residuals_all["block"] == "Damaged test"]
first_alarm = damage_rows[damage_rows["alarm"]]["start"].min()
delay_hours = (first_alarm - DAMAGE_START).total_seconds() / 3600
print(f"First alarm {first_alarm}, delay {delay_hours:.2f} h, "
      f"{(damage_rows['start'] < first_alarm).sum()} measurements before, "
      f"{damage_rows['alarm'].mean() * 100:.1f} % in alarm, "
      f"{damage_rows[damage_rows['start'] >= first_alarm]['alarm'].mean() * 100:.1f} % from the first")

# Figure: smoothed index over the whole monitoring and from 01.07.1998
fig, axes = plt.subplots(2, 1, figsize=(COLUMN_WIDTH, 3.6))
for i, period_start in enumerate([residuals_all["start"].min(), ZOOM_START]):
    ax = axes[i]
    period_rows = residuals_all[residuals_all["start"] >= period_start]
    shade_blocks(ax, period_start, DATA_END)
    normal = period_rows[~period_rows["alarm"]]
    alarm = period_rows[period_rows["alarm"]]
    ax.plot(normal["start"], normal["Z"], ".", markersize=1.2, color="black", label="Below the limit")
    ax.plot(alarm["start"], alarm["Z"], "x", markersize=1.8, markeredgewidth=0.5, color="tab:red",
            label="Alarm")
    ax.axhline(Z_LIMIT, color="tab:red", linestyle="--", linewidth=0.8, label=f"Limit = {Z_LIMIT:.2f}")
    ax.set_yscale("log")
    ax.set_ylabel("Smoothed index, $Z$")
    ax.set_xlim(period_start, DATA_END)
    ax.set_title(PANEL_LABELS[i], loc="left")
    ax.grid(alpha=0.3)
axes[0].xaxis.set_major_locator(mdates.MonthLocator(bymonth=[11, 1, 3, 5, 7, 9]))
axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
axes[1].xaxis.set_major_locator(mdates.DayLocator(bymonthday=[1, 15]))
axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%d.%m"))
axes[1].legend(fontsize=5.5, loc="upper left", markerscale=3)
fig.legend(handles=SHADE_HANDLES, loc="upper center", ncol=2, fontsize=6, frameon=False)
fig.tight_layout(pad=0.3)
fig.subplots_adjust(top=0.88)
fig.savefig(os.path.join(FIGURES_FOLDER, "detection_index.pdf"))
plt.close(fig)

# Figure: ROC curves of Z and D2 with the test blocks
test_rows = residuals_all[residuals_all["block"] != "Training"]
is_damaged = (test_rows["block"] == "Damaged test").astype(int)
healthy_test = test_rows[is_damaged == 0]
damaged_test = test_rows[is_damaged == 1]
point = ((healthy_test["Z"] > Z_LIMIT).mean(), (damaged_test["Z"] > Z_LIMIT).mean())
print(f"Chosen limit: false-alarm rate {point[0] * 100:.1f} %, detection rate {point[1] * 100:.1f} %")

fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 2.4))
inset = ax.inset_axes([0.33, 0.4, 0.4, 0.45])      # zoom on the upper left corner
for column, label, style in [("Z", r"$Z$ (smoothed, $\lambda$ = 0.1)", "-"), ("D2", "$D^2$ (not smoothed)", "--")]:
    false_alarm_rate, detection_rate, _ = roc_curve(is_damaged, test_rows[column])
    auc = roc_auc_score(is_damaged, test_rows[column])
    print(f"{column}: AUC = {auc:.3f}")
    for target in [ax, inset]:
        target.plot(false_alarm_rate, detection_rate, style, color="black", linewidth=1,
                    label=f"{label}, AUC = {auc:.3f}")
for target in [ax, inset]:
    target.plot(point[0], point[1], "o", markersize=5, markerfacecolor="tab:red", markeredgecolor="black",
                label="Chosen limit")
inset.set_xlim(0, 0.2)
inset.set_ylim(0.8, 1.005)
inset.tick_params(labelsize=5)
inset.grid(alpha=0.3)
ax.indicate_inset_zoom(inset, edgecolor="0.4")
ax.plot([0, 1], [0, 1], ":", color="gray", linewidth=0.8)
ax.set_xlabel("False-alarm rate (healthy test measurements)")
ax.set_ylabel("Detection rate\n(damaged measurements)")
ax.grid(alpha=0.3)
ax.legend(fontsize=6, loc="lower right")
fig.tight_layout(pad=0.3)
fig.savefig(os.path.join(FIGURES_FOLDER, "roc_curve.pdf"))
plt.close(fig)
