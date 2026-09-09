"""Build training-time features from a live phone sensor buffer."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

APP_DIR = Path(__file__).resolve().parent
FEATURES_DIR = APP_DIR.parent / "features"
sys.path.insert(0, str(FEATURES_DIR))

from DataTransformation import LowPassFilter  # noqa: E402
from FrequencyAbstraction import FourierTransformation  
from TemporalAbstraction import NumericalAbstraction  
GRAVITY_MS2 = 9.80665
PREDICTOR_COLUMNS = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
FS = 5.0
CUTOFF = 1.3
TEMP_WS = 5
FREQ_WS = 10
MIN_RESAMPLED_ROWS = max(FREQ_WS + 2, 22)  # filtfilt padlen ~18


def _impute(df):
    for col in PREDICTOR_COLUMNS:
        if col in df.columns:
            df[col] = (
                df[col].interpolate(limit_direction="both").bfill().ffill().fillna(0)
            )
    return df


def _apply_lowpass(df, fs=FS, cutoff=CUTOFF):
    lowpass = LowPassFilter()
    out = df.copy()
    for col in PREDICTOR_COLUMNS:
        out = lowpass.low_pass_filter(out, col, fs, cutoff, order=5)
        out[col] = out[col + "_lowpass"]
        del out[col + "_lowpass"]
    return out


def _apply_pca(df, pca, col_means, col_mins, col_maxs):
    """Match DataTransformation.PrincipalComponentAnalysis normalization."""
    out = df.copy()
    span = (col_maxs - col_mins).replace(0, 1e-9)
    norm = (out[PREDICTOR_COLUMNS] - col_means) / span
    transformed = pca.transform(norm)
    for i in range(transformed.shape[1]):
        out[f"pca_{i + 1}"] = transformed[:, i]
    return out


def _add_magnitude(df):
    out = df.copy()
    out["acc_r"] = np.sqrt(out["acc_x"] ** 2 + out["acc_y"] ** 2 + out["acc_z"] ** 2)
    out["gyro_r"] = np.sqrt(
        out["gyro_x"] ** 2 + out["gyro_y"] ** 2 + out["gyro_z"] ** 2
    )
    return out


def _apply_temporal(df, temp_ws=TEMP_WS):
    out = df.copy()
    num_abs = NumericalAbstraction()
    feature_cols = PREDICTOR_COLUMNS + ["acc_r", "gyro_r"]
    for col in feature_cols:
        out = num_abs.abstract_numerical(out, [col], temp_ws, "mean")
        out = num_abs.abstract_numerical(out, [col], temp_ws, "std")
    for col in feature_cols:
        out = num_abs.abstract_numerical(out, [col], temp_ws, "mean")
        out = num_abs.abstract_numerical(out, [col], temp_ws, "std")
    return out


def _apply_frequency(df, freq_ws=FREQ_WS, fs=FS):
    """Match build_features: acc_y first, then all predictor columns per set."""
    out = df.reset_index()
    time_col = out.columns[0]
    freq_abs = FourierTransformation()
    feature_cols = PREDICTOR_COLUMNS + ["acc_r", "gyro_r"]
    out = freq_abs.abstract_frequency(out, ["acc_y"], freq_ws, int(fs))
    out = freq_abs.abstract_frequency(out, feature_cols, freq_ws, int(fs))
    return out.set_index(time_col, drop=True)


def readings_to_resampled_df(readings):
    """Convert phone JSON readings to a 5 Hz dataframe (acc in g)."""
    rows = []
    for reading in readings:
        if hasattr(reading, "t"):
            t = reading.t
            acc_x, acc_y, acc_z = reading.acc_x, reading.acc_y, reading.acc_z
            gyro_x = reading.gyro_x or 0.0
            gyro_y = reading.gyro_y or 0.0
            gyro_z = reading.gyro_z or 0.0
        else:
            t = reading["t"]
            acc_x, acc_y, acc_z = reading["acc_x"], reading["acc_y"], reading["acc_z"]
            gyro_x = reading.get("gyro_x") or 0.0
            gyro_y = reading.get("gyro_y") or 0.0
            gyro_z = reading.get("gyro_z") or 0.0

        rows.append(
            {
                "acc_x": acc_x / GRAVITY_MS2,
                "acc_y": acc_y / GRAVITY_MS2,
                "acc_z": acc_z / GRAVITY_MS2,
                "gyro_x": gyro_x,
                "gyro_y": gyro_y,
                "gyro_z": gyro_z,
                "t": t,
            }
        )

    df = pd.DataFrame(rows)
    df.index = pd.to_datetime(df["t"], unit="ms")
    df = df.drop(columns=["t"])
    df = _impute(df)
    resampled = df.resample("200ms").mean().dropna(subset=["acc_x", "acc_y", "acc_z"])
    return resampled


def build_feature_row(df_resampled, live_artifact):
    if len(df_resampled) < MIN_RESAMPLED_ROWS:
        raise ValueError(
            f"Need at least {MIN_RESAMPLED_ROWS} resampled rows, got {len(df_resampled)}"
        )

    df = _apply_lowpass(df_resampled, live_artifact["fs"], live_artifact["cutoff"])
    df = _apply_pca(
        df,
        live_artifact["pca"],
        live_artifact["col_means"],
        live_artifact["col_mins"],
        live_artifact["col_maxs"],
    )
    df = _add_magnitude(df)
    df = _apply_temporal(df, live_artifact["temp_ws"])
    df = _apply_frequency(df, live_artifact["freq_ws"], live_artifact["fs"])
    df = df.dropna()
    if df.empty:
        raise ValueError("Feature pipeline returned no complete rows")

    row = df.iloc[-1].copy()
    cluster_cols = ["acc_x", "acc_y", "acc_z"]
    cluster_input = pd.DataFrame([row[cluster_cols].tolist()], columns=cluster_cols)
    row["cluster"] = live_artifact["kmeans"].predict(cluster_input)[0]
    return row


def fit_live_artifact(processed_pkl, features_pkl):
    """Fit PCA + KMeans from training pickles (run after build_features + save_model)."""
    df = pd.read_pickle(processed_pkl)
    for col in PREDICTOR_COLUMNS:
        df[col] = df.groupby("set")[col].transform(
            lambda series: series.interpolate(limit_direction="both")
        )
        df[col] = df[col].bfill().ffill()

    df_lowpass = _apply_lowpass(df)
    col_means = df_lowpass[PREDICTOR_COLUMNS].mean()
    col_mins = df_lowpass[PREDICTOR_COLUMNS].min()
    col_maxs = df_lowpass[PREDICTOR_COLUMNS].max()
    span = (col_maxs - col_mins).replace(0, 1e-9)
    norm = (df_lowpass[PREDICTOR_COLUMNS] - col_means) / span
    pca = PCA(n_components=3).fit(norm)

    features_df = pd.read_pickle(features_pkl)
    kmeans = KMeans(n_clusters=5, n_init=20, random_state=0)
    kmeans.fit(features_df[["acc_x", "acc_y", "acc_z"]])

    return {
        "pca": pca,
        "col_means": col_means,
        "col_mins": col_mins,
        "col_maxs": col_maxs,
        "kmeans": kmeans,
        "fs": FS,
        "cutoff": CUTOFF,
        "temp_ws": TEMP_WS,
        "freq_ws": FREQ_WS,
    }
