import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from IPython.display import display
import seaborn as sns

# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------
df = pd.read_pickle("../../data/interim/01_data_processed.pkl")

# --------------------------------------------------------------
# Plot single columns
# --------------------------------------------------------------
set_df = df[df["set"] == 1]
plt.plot(set_df["acc_y"])
plt.plot(set_df["acc_y"].reset_index(drop=True))

# --------------------------------------------------------------
# Plot all exercises
# --------------------------------------------------------------
for label in df["label"].unique():
    subset = df[df["label"] == label]
    fig, ax = plt.subplots()
    plt.plot(subset["acc_y"].reset_index(drop=True), label=label)
    plt.legend()
    plt.show()
for label in df["label"].unique():
    subset = df[df["label"] == label]
    fig, ax = plt.subplots()
    plt.plot(subset[:100]["acc_y"].reset_index(drop=True), label=label)
    plt.legend()
    plt.show()

# --------------------------------------------------------------
# Adjust plot settings
# --------------------------------------------------------------
mpl.style.use("seaborn-v0_8-deep")
mpl.rcParams["figure.figsize"] = (20, 5)
mpl.rcParams["figure.dpi"] = 100
# --------------------------------------------------------------
# Compare medium vs. heavy sets
# --------------------------------------------------------------
category_df = df.query(" label == 'squat' ").query("participant == 'A'").reset_index()

fig, ax = plt.subplots()
category_df.groupby(["category"])["acc_y"].plot()
ax.set_ylabel("acc_y")
ax.set_xlabel("samples")
plt.legend()
# --------------------------------------------------------------
# Compare participants
# --------------------------------------------------------------
participant_df = df.query(" label == 'bench' ").sort_values("participant").reset_index()
fig, ax = plt.subplots()
participant_df.groupby(["participant"])["acc_y"].plot()
ax.set_ylabel("acc_y")
ax.set_xlabel("samples")
plt.legend()
# --------------------------------------------------------------
# Plot multiple axis
# --------------------------------------------------------------
label = "squat"
participant = "A"
all_axis_df = (
    df.query(f"label ==  '{label}' ")
    .query(f"participant == '{participant}'")
    .reset_index()
)


fig, ax = plt.subplots()
all_axis_df[["acc_x", "acc_y", "acc_z"]].plot(ax=ax)
ax.set_ylabel("acc_y")
ax.set_xlabel("samples")
plt.legend()


# --------------------------------------------------------------
# Create a loop to plot all combinations per sensor
# --------------------------------------------------------------
labels = df["label"].unique()
participants = df["participant"].unique()

for label in labels:
    for participant in participants:
        all_axis_df = (
            df.query(f"label ==  '{label}' ")
            .query(f"participant == '{participant}'")
            .reset_index()
        )

        if len(all_axis_df) > 0:
            fig, ax = plt.subplots()
            all_axis_df[["acc_x", "acc_y", "acc_z"]].plot(ax=ax)
            ax.set_ylabel("acc_y")
            ax.set_xlabel("samples")
            plt.title(f"{label} ({participant})".title())
            plt.legend()
for label in labels:
    for participant in participants:
        all_axis_df = (
            df.query(f"label ==  '{label}' ")
            .query(f"participant == '{participant}'")
            .reset_index()
        )

        if len(all_axis_df) > 0:
            fig, ax = plt.subplots()
            all_axis_df[["gyro_x", "gyro_y", "gyro_z"]].plot(ax=ax)
            ax.set_ylabel("gyro_y")
            ax.set_xlabel("samples")
            plt.title(f"{label} ({participant})".title())
            plt.legend()


# --------------------------------------------------------------
# Combine plots in one figure
# --------------------------------------------------------------
label = "row"
participant = "A"
combined_plot_df = (
    df.query(f"label ==  '{label}' ")
    .query(f"participant == '{participant}'")
    .reset_index(drop=True)
)

# ai generated styling because why not
fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(20, 10))
colors = ["#e41a1c", "#377eb8", "#e9a60a"]

combined_plot_df[["acc_x", "acc_y", "acc_z"]].plot(ax=ax[0], color=colors, alpha=0.8)
ax[0].set_title(
    f"Activity: {label.upper()} | Participant: {participant}", fontsize=16, pad=20
)
ax[0].set_ylabel("Acceleration (g)", fontsize=12)
ax[0].legend(loc="upper right", ncol=3)

# gyroscope Data
combined_plot_df[["gyro_x", "gyro_y", "gyro_z"]].plot(ax=ax[1], color=colors, alpha=0.8)
ax[1].set_ylabel("Gyroscope (deg/s)", fontsize=12)
ax[1].set_xlabel("Samples", fontsize=12)
ax[1].legend(loc="upper right", ncol=3)
plt.tight_layout()
plt.subplots_adjust(hspace=0.15)  # Reduce space between subplots
plt.show()


# --------------------------------------------------------------
# Loop over all combinations and export for both sensors
# --------------------------------------------------------------

labels = df["label"].unique()
participants = df["participant"].unique()

for label in labels:
    for participant in participants:
        combined_plot_df = (
            df.query(f"label == '{label}'")
            .query(f"participant == '{participant}'")
            .reset_index(drop=True)
        )

        if len(combined_plot_df) > 0:
            fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(20, 10))
            fig.suptitle(
                f"Activity: {label.upper()} | Participant: {participant}",
                fontsize=18,
                y=0.98,
            )

            combined_plot_df[["acc_x", "acc_y", "acc_z"]].plot(
                ax=ax[0], color=colors, alpha=0.8
            )
            ax[0].set_ylabel("Acceleration (g)", fontsize=12)
            ax[0].legend(
                loc="upper center",
                bbox_to_anchor=(0.5, 1.15),
                ncol=3,
                fancybox=True,
                shadow=True,
            )
            combined_plot_df[["gyro_x", "gyro_y", "gyro_z"]].plot(
                ax=ax[1], color=colors, alpha=0.8
            )
            ax[1].set_ylabel("Gyroscope (deg/s)", fontsize=12)
            ax[1].set_xlabel("Samples", fontsize=12)
            ax[1].legend(
                loc="upper center",
                bbox_to_anchor=(0.5, 1.15),
                ncol=3,
                fancybox=True,
                shadow=True,
            )
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])

            plt.savefig(f"../../reports/figures/{label.title()} ({participant}).png")
            plt.show()
