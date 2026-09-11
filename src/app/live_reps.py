"""Live rep counting — same approach as count_repetitions.py (lowpass + peaks)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import argrelextrema

FEATURES_DIR = Path(__file__).resolve().parent.parent / "features"
sys.path.insert(0, str(FEATURES_DIR))

from DataTransformation import LowPassFilter  # noqa: E402

FS = 5.0
ORDER = 10
MIN_ROWS = 25

DEFAULTS = {
    "column": "acc_r",
    "cutoff": 0.4,
    "peak_order": 7,
    "min_gap_samples": 11,
    "prominence": 0.14,
    "min_peak_gap_ms": 2200,
}

REP_CONFIG = {
    "bench": {"cutoff": 0.4, "min_gap_samples": 11, "prominence": 0.14},
    "squat": {"cutoff": 0.38, "min_gap_samples": 11, "prominence": 0.14},
    "row": {"cutoff": 0.50, "min_gap_samples": 12, "prominence": 0.15},
    # OHP: slower reps, more noise while resting — stricter gaps & prominence
    "ohp": {
        "cutoff": 0.35,
        "peak_order": 9,
        "min_gap_samples": 15,
        "prominence": 0.20,
        "min_peak_gap_ms": 3000,
    },
    "dead": {"cutoff": 0.4, "min_gap_samples": 12, "prominence": 0.15},
}


def get_rep_config(exercise):
    cfg = dict(DEFAULTS)
    if exercise in REP_CONFIG:
        cfg.update(REP_CONFIG[exercise])
    return cfg


def _add_magnitude(df):
    out = df.copy()
    out["acc_r"] = np.sqrt(out["acc_x"] ** 2 + out["acc_y"] ** 2 + out["acc_z"] ** 2)
    return out


def _filter_peak_indices(signal, indexes, min_gap_samples):
    if len(indexes) == 0:
        return indexes

    kept = [int(indexes[0])]
    for idx in indexes[1:]:
        idx = int(idx)
        if idx - kept[-1] >= min_gap_samples:
            kept.append(idx)
        elif signal[idx] > signal[kept[-1]]:
            kept[-1] = idx
    return np.array(kept, dtype=int)


def _filter_by_prominence(signal, indexes, min_prominence):
    if len(indexes) == 0:
        return indexes

    kept = []
    for idx in indexes:
        idx = int(idx)
        left = max(0, idx - 12)
        right = min(len(signal), idx + 13)
        window = signal[left:right]
        baseline = float(np.min(window))
        peak = float(signal[idx])
        span = float(np.max(window) - baseline)
        if span <= 0:
            continue
        if (peak - baseline) / span >= min_prominence:
            kept.append(idx)
    return np.array(kept, dtype=int)


def find_peak_times_ms(df_resampled, exercise, after_ms=0):
    """Return peak timestamps (ms) from a 5 Hz dataframe."""
    if exercise == "rest" or df_resampled is None or len(df_resampled) < MIN_ROWS:
        return []

    cfg = get_rep_config(exercise)
    column = cfg["column"]
    data = _add_magnitude(df_resampled)

    if after_ms > 0:
        data = data[data.index >= pd.to_datetime(after_ms, unit="ms")]
        if len(data) < MIN_ROWS:
            return []

    lowpass = LowPassFilter()
    try:
        filtered = lowpass.low_pass_filter(
            data,
            col=column,
            sampling_frequency=FS,
            cutoff_frequency=cfg["cutoff"],
            order=ORDER,
        )
    except ValueError:
        return []

    signal = filtered[f"{column}_lowpass"].values
    if len(signal) < ORDER * 3:
        return []

    indexes = argrelextrema(signal, np.greater, order=cfg["peak_order"])[0]
    indexes = _filter_peak_indices(signal, indexes, cfg["min_gap_samples"])
    indexes = _filter_by_prominence(signal, indexes, cfg["prominence"])

    index_times = data.index[indexes]
    return [int(ts.timestamp() * 1000) for ts in index_times]


def count_reps_in_df(df_resampled, exercise):
    return len(find_peak_times_ms(df_resampled, exercise))
