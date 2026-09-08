import pandas as pd
from glob import glob
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw" / "MetaMotion" / "MetaMotion"
DATA_INTERIM = ROOT / "data" / "interim"

# --------------------------------------------------------------
# Read single CSV file
# --------------------------------------------------------------
single_file_acc = pd.read_csv(
    DATA_RAW / "A-bench-heavy2-rpe8_MetaWear_2019-01-11T16.10.08.270_C42732BE255C_Accelerometer_12.500Hz_1.4.4.csv"
)

single_file_gyro = pd.read_csv(
    DATA_RAW / "A-bench-heavy2-rpe8_MetaWear_2019-01-11T16.10.08.270_C42732BE255C_Gyroscope_25.000Hz_1.4.4.csv"
)
# --------------------------------------------------------------
# List all data in data/raw/MetaMotion
# --------------------------------------------------------------
files = glob(
    str(DATA_RAW / "*.csv")
)  # todos os ficheiros csv dentro desta pasta, e acessiveis atraves de indice (como esta em baixo)len(files)
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
# Turn into function
# --------------------------------------------------------------
files = glob(str(DATA_RAW / "*.csv"))


def parse_filename(filename):
    basename = os.path.basename(filename)
    participant = basename.split("-")[0]
    label = basename.split("-")[1]
    category = basename.split("-")[2].rstrip("123").rstrip("_MetaWear_2019")
    return participant, label, category


def session_key(filename):
    # chave unica para emparelhar acc e gyro da mesma gravacao
    basename = os.path.basename(filename)
    for sensor in ("Accelerometer", "Gyroscope"):
        if sensor in basename:
            return basename.split(f"_{sensor}")[0]
    raise ValueError(f"Sensor desconhecido em: {filename}")


def dedupe_sensor_files(files):
    # Algumas gravacoes existem duplicadas no dataset (ex: versao 1.4.4 e 1.4.41
    # no nome do ficheiro), mas sao a mesma sessao. Esta funcao mantem apenas
    # um ficheiro por gravacao, preferindo a versao 1.4.4.
    by_session = {}
    for filepath in sorted(files):
        key = session_key(filepath)
        if key not in by_session or "1.4.41" in by_session[key]:
            if "1.4.41" not in filepath or key not in by_session:
                by_session[key] = filepath
    return by_session


def load_sensor_file(filepath):
    participant, label, category = parse_filename(filepath)
    df = pd.read_csv(filepath)
    df.index = pd.to_datetime(df["epoch (ms)"], unit="ms")

    if "Accelerometer" in filepath:
        df = df.rename(
            columns={
                "x-axis (g)": "acc_x",
                "y-axis (g)": "acc_y",
                "z-axis (g)": "acc_z",
            }
        )
        value_cols = ["acc_x", "acc_y", "acc_z"]
    else:
        df = df.rename(
            columns={
                "x-axis (deg/s)": "gyro_x",
                "y-axis (deg/s)": "gyro_y",
                "z-axis (deg/s)": "gyro_z",
            }
        )
        value_cols = ["gyro_x", "gyro_y", "gyro_z"]

    df = df[value_cols]
    df["participant"] = participant
    df["label"] = label
    df["category"] = category
    return df


def resample_sensor(df, value_cols, meta_cols, rule="200ms"):
    sampling = {col: "mean" for col in value_cols}
    sampling.update({col: "last" for col in meta_cols})
    return df.resample(rule).agg(sampling).dropna(subset=value_cols)


def merge_and_resample_session(acc_df, gyro_df, set_id):
    acc_df = acc_df.copy()
    gyro_df = gyro_df.copy()
    acc_df["set"] = set_id
    gyro_df["set"] = set_id

    meta_cols = ["participant", "label", "category", "set"]

    acc_resampled = resample_sensor(acc_df, ["acc_x", "acc_y", "acc_z"], meta_cols)
    gyro_resampled = resample_sensor(
        gyro_df, ["gyro_x", "gyro_y", "gyro_z"], meta_cols
    )

    # join temporal: alinha acc e gyro pelo indice de tempo, nao por posicao de linha
    data_merged = acc_resampled[
        ["acc_x", "acc_y", "acc_z", "participant", "label", "category", "set"]
    ].join(gyro_resampled[["gyro_x", "gyro_y", "gyro_z"]], how="inner")

    return data_merged


def data_from_files(files):
    acc_files = dedupe_sensor_files([f for f in files if "Accelerometer" in f])
    gyro_files = dedupe_sensor_files([f for f in files if "Gyroscope" in f])

    common_sessions = sorted(set(acc_files.keys()) & set(gyro_files.keys()))
    merged_sessions = []

    for set_id, session in enumerate(common_sessions, start=1):
        acc_df = load_sensor_file(acc_files[session])
        gyro_df = load_sensor_file(gyro_files[session])
        merged_sessions.append(merge_and_resample_session(acc_df, gyro_df, set_id))

    data_resampled = pd.concat(merged_sessions)
    data_resampled["set"] = data_resampled["set"].astype(int)
    return data_resampled


# --------------------------------------------------------------
# Read all files
# --------------------------------------------------------------
data_resampled = data_from_files(files)

# --------------------------------------------------------------
# Merging datasets
# --------------------------------------------------------------
# O merge acc + gyro e feito dentro de data_from_files(), gravacao a gravacao.
# Cada par acc/gyro e reamostrado para 200ms e depois unido pelo timestamp.

# --------------------------------------------------------------
# Resample data (frequency conversion)
# --------------------------------------------------------------

# Accelerometer:    12.500HZ
# Gyroscope:        25.000Hz
# Reamostragem final para 5Hz (200ms) ja aplicada por gravacao em merge_and_resample_session

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

# Reamostragem do df principal (exemplo com subset)
data_resampled[:1000].resample(rule="200ms").apply(sampling)

# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------

DATA_INTERIM.mkdir(parents=True, exist_ok=True)
data_resampled.to_pickle(
    DATA_INTERIM / "01_data_processed.pkl"
)  # to pickle guarda o df serializado, ao abrir o ficheiro vai estar direito e como foi salvo (especialmente bom para datas)