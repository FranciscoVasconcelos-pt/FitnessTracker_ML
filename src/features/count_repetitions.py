import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from DataTransformation import LowPassFilter
from rep_counting import count_reps_in_set, get_rep_config, peaks_from_df
from scipy.signal import argrelextrema
from sklearn.metrics import mean_absolute_error

pd.options.mode.chained_assignment = None

DATA_INTERIM = Path(__file__).resolve().parents[2] / "data" / "interim"


# Plot settings
plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (20, 5)
plt.rcParams["figure.dpi"] = 100
plt.rcParams["lines.linewidth"] = 2


# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------
df = pd.read_pickle(DATA_INTERIM / "01_data_processed.pkl")
df = df[df["label"] != "rest"]

acc_r = df["acc_x"] ** 2 + df["acc_y"] ** 2 + df["acc_z"] ** 2
gyro_r = df["gyro_x"] ** 2 + df["gyro_y"] ** 2 + df["gyro_z"] ** 2
df["acc_r"] = np.sqrt(acc_r)
df["gyro_r"] = np.sqrt(gyro_r)

# --------------------------------------------------------------
# Split data
# --------------------------------------------------------------

bench_df = df[df["label"] == "bench"]
squat_df = df[df["label"] == "squat"]
row_df = df[df["label"] == "row"]
ohp_df = df[df["label"] == "ohp"]
dead_df = df[df["label"] == "dead"]
# --------------------------------------------------------------
# Visualize data to identify patterns
# --------------------------------------------------------------

plot_df = bench_df
plot_df[plot_df["set"] == plot_df["set"].unique()[0]]["acc_x"].plot()
plot_df[plot_df["set"] == plot_df["set"].unique()[0]]["acc_y"].plot()
plot_df[plot_df["set"] == plot_df["set"].unique()[0]]["acc_z"].plot()
plot_df[plot_df["set"] == plot_df["set"].unique()[0]]["acc_r"].plot()

plot_df[plot_df["set"] == plot_df["set"].unique()[0]]["gyro_x"].plot()
plot_df[plot_df["set"] == plot_df["set"].unique()[0]]["gyro_y"].plot()
plot_df[plot_df["set"] == plot_df["set"].unique()[0]]["gyro_z"].plot()
plot_df[plot_df["set"] == plot_df["set"].unique()[0]]["gyro_r"].plot()

# --------------------------------------------------------------
# Configure LowPassFilter
# --------------------------------------------------------------

fs = 1000 / 200
LowPass = LowPassFilter()


# --------------------------------------------------------------
# Apply and tweak LowPassFilter
# --------------------------------------------------------------
bench_set = bench_df[bench_df["set"] == bench_df["set"].unique()[0]]
squat_set = squat_df[squat_df["set"] == squat_df["set"].unique()[0]]
row_set = row_df[row_df["set"] == row_df["set"].unique()[0]]
ohp_set = ohp_df[ohp_df["set"] == ohp_df["set"].unique()[0]]
dead_set = dead_df[dead_df["set"] == dead_df["set"].unique()[0]]

bench_set["acc_r"].plot()
column = "acc_y"
LowPass.low_pass_filter(
    bench_set, col=column, sampling_frequency=fs, cutoff_frequency=0.4, order=10
)[column + "_lowpass"].plot()


# --------------------------------------------------------------
# Count repetitions (shared rep_counting module)
# --------------------------------------------------------------
def count_reps(dataset, exercise=None, show_plot=False):
    label = exercise or dataset["label"].iloc[0]
    reps = count_reps_in_set(dataset, label)

    if show_plot:
        cfg = get_rep_config(label)
        column = cfg["column"]
        data = LowPass.low_pass_filter(
            dataset,
            col=column,
            sampling_frequency=fs,
            cutoff_frequency=cfg["cutoff"],
            order=10,
        )
        peak_times = {ms for ms, _ in peaks_from_df(dataset, label)}
        peak_mask = [
            int(ts.timestamp() * 1000) in peak_times for ts in data.index
        ]
        peaks = data[peak_mask]

        fig, ax = plt.subplots()
        plt.plot(data[f"{column}_lowpass"])
        plt.plot(peaks[f"{column}_lowpass"], "o", color="red")
        ax.set_ylabel(f"{column}_lowpass")
        exercise_name = label.title()
        category = dataset["category"].iloc[0].title()
        plt.title(f"{category} {exercise_name}: {reps} Reps")
        plt.close(fig)

    return reps


count_reps(bench_set)
count_reps(squat_set)
count_reps(row_set)
count_reps(ohp_set)
count_reps(dead_set)
# --------------------------------------------------------------
# Create benchmark dataframe
# --------------------------------------------------------------

df["reps"] = df["category"].apply(lambda x: 5 if x == "heavy" else 10)
rep_df = df.groupby(["label", "category", "set"])["reps"].max().reset_index()
rep_df["reps_pred"] = 0

for s in df["set"].unique():
    subset = df[df["set"] == s]
    label = subset["label"].iloc[0]
    reps = count_reps_in_set(subset, label)
    rep_df.loc[rep_df["set"] == s, "reps_pred"] = reps

rep_df
# --------------------------------------------------------------
# Evaluate the results
# --------------------------------------------------------------

erro = mean_absolute_error(rep_df["reps"], rep_df["reps_pred"]).round(2)
summary = rep_df.groupby(["label", "category"])[["reps", "reps_pred"]].mean().round(2)

print(f"Mean Absolute Error (reps): {erro}")
print("\nAverage reps by exercise and category:")
print(summary.to_string())
