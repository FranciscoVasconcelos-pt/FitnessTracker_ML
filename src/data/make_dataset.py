import pandas as pd
from glob import glob
import os

# --------------------------------------------------------------
# Read single CSV file
# --------------------------------------------------------------
single_file_acc = pd.read_csv(
    "../../data/raw/MetaMotion/MetaMotion/A-bench-heavy2-rpe8_MetaWear_2019-01-11T16.10.08.270_C42732BE255C_Accelerometer_12.500Hz_1.4.4.csv"
)

single_file_gyro = pd.read_csv(
    "../../data/raw/MetaMotion/MetaMotion/A-bench-heavy2-rpe8_MetaWear_2019-01-11T16.10.08.270_C42732BE255C_Gyroscope_25.000Hz_1.4.4.csv"
)
# --------------------------------------------------------------
# List all data in data/raw/MetaMotion
# --------------------------------------------------------------
files = glob(
    "../../data/raw/MetaMotion/MetaMotion/*.csv"
)  # todos os ficheiros csv dentro desta pasta, e acessiveis atraves de indice (como esta em baixo)
len(files)
# --------------------------------------------------------------
# Extract features from filename
# --------------------------------------------------------------
# files[0] acesso

f = files[0]
filename = os.path.basename(f)  # remove o caminho
participant = filename.split("-")[0]
label = filename.split("-")[1]
category = filename.split("-")[2].rstrip("123").rstrip("_MetaWear_2019")

df = pd.read_csv(f)

df["participant"] = participant
df["label"] = label
df["category"] = category

# --------------------------------------------------------------
# Read all files
# --------------------------------------------------------------

acc_df = pd.DataFrame()
gyro_df = pd.DataFrame()

acc_set = 1  # counter individual
gyro_set = 1

for f in files:
    filename = os.path.basename(f)  # remove o caminho
    participant = filename.split("-")[0]
    label = filename.split("-")[1]
    category = filename.split("-")[2].rstrip("123").rstrip("_MetaWear_2019")

    df = pd.read_csv(f)

    df["participant"] = participant
    df["label"] = label
    df["category"] = category

    if "Accelerometer" in f:
        df["set"] = acc_set
        acc_set += 1
        acc_df = pd.concat([acc_df, df])  # concat junta o df ao acc_df e constroi o df

    if "Gyroscope" in f:
        df["set"] = gyro_set
        gyro_set += 1
        gyro_df = pd.concat([gyro_df, df])

# --------------------------------------------------------------
# Working with datetimes
# --------------------------------------------------------------
acc_df.index = pd.to_datetime(acc_df["epoch (ms)"], unit="ms")
gyro_df.index = pd.to_datetime(gyro_df["epoch (ms)"], unit="ms")

acc_df.drop(["epoch (ms)", "time (01:00)"], axis=1)
gyro_df.drop(["epoch (ms)", "time (01:00)"], axis=1)
# --------------------------------------------------------------
# Turn into function
# --------------------------------------------------------------
files = glob("../../data/raw/MetaMotion/MetaMotion/*.csv")


def data_from_files(files):
    acc_df = pd.DataFrame()
    gyro_df = pd.DataFrame()

    acc_set = 1  # counter individual
    gyro_set = 1

    for f in files:
        filename = os.path.basename(f)  # remove o caminho
        participant = filename.split("-")[0]
        label = filename.split("-")[1]
        category = filename.split("-")[2].rstrip("123").rstrip("_MetaWear_2019")

        df = pd.read_csv(f)

        df["participant"] = participant
        df["label"] = label
        df["category"] = category

        if "Accelerometer" in f:
            df["set"] = acc_set
            acc_set += 1
            acc_df = pd.concat(
                [acc_df, df]
            )  # concat junta o df ao acc_df e constroi o df

        if "Gyroscope" in f:
            df["set"] = gyro_set
            gyro_set += 1
            gyro_df = pd.concat([gyro_df, df])

    acc_df.index = pd.to_datetime(acc_df["epoch (ms)"], unit="ms")
    gyro_df.index = pd.to_datetime(gyro_df["epoch (ms)"], unit="ms")

    acc_df.drop(
        ["epoch (ms)", "time (01:00)", "elapsed (s)"], axis=1, inplace=True
    )  # inplace apaga as colunas do DF diretamente
    gyro_df.drop(["epoch (ms)", "time (01:00)", "elapsed (s)"], axis=1, inplace=True)

    return acc_df, gyro_df


acc_df, gyro_df = data_from_files(files)


# --------------------------------------------------------------
# Merging datasets
# --------------------------------------------------------------

data_merged = pd.concat([acc_df.iloc[:, :3], gyro_df], axis=1)
data_merged.columns = [
    "acc_x",
    "acc_y",
    "acc_z",
    "gyro_x",
    "gyro_y",
    "gyro_z",
    "label",
    "category",
    "participant",
    "set",
]

# --------------------------------------------------------------
# Resample data (frequency conversion)
# --------------------------------------------------------------

# Accelerometer:    12.500HZ
# Gyroscope:        25.000Hz

sampling = {
    "acc_x": "mean",
    "acc_y": "mean",
    "acc_z": "mean",
    "gyro_x": "mean",
    "gyro_y": "mean",
    "gyro_z": "mean",
    "label": "last",
    "category": "last",
    "participant": "last",
    "set": "last",
}

# Reamostragem do df principal
data_merged[:1000].resample(rule="200ms").apply(sampling)

days = [g for n, g in data_merged.groupby(pd.Grouper(freq="D"))]

data_resampled = pd.concat(
    [df.resample(rule="200ms").apply(sampling).dropna() for df in days]
)


# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------
