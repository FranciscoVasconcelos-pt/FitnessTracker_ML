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


# --------------------------------------------------------------
# Turn into function
# --------------------------------------------------------------


# --------------------------------------------------------------
# Merging datasets
# --------------------------------------------------------------


# --------------------------------------------------------------
# Resample data (frequency conversion)
# --------------------------------------------------------------

# Accelerometer:    12.500HZ
# Gyroscope:        25.000Hz


# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------
