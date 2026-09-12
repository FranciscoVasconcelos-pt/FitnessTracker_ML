"""Shared rep counting — batch sets and live incremental peaks."""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.signal import argrelextrema

from DataTransformation import LowPassFilter

FS = 5.0
ORDER = 10
MIN_ROWS = 25
PEAK_WINDOW = 12

DEFAULTS = {
    "column": "acc_r",
    "cutoff": 0.4,
    "peak_order": 7,
    "min_gap_samples": 11,
    "prominence": 0.14,
    "min_peak_gap_ms": 2200,
    "min_peak_delta": 0.10,
    "min_amplitude_ratio": 0.40,
    "movement_std_threshold": 0.035,
    "movement_window_samples": 5,
}

REP_CONFIG = {
    "bench": {"cutoff": 0.4, "min_gap_samples": 11, "prominence": 0.14},
    "squat": {"cutoff": 0.38, "min_gap_samples": 11, "prominence": 0.14},
    "row": {"cutoff": 0.50, "min_gap_samples": 12, "prominence": 0.15},
    "ohp": {
        "cutoff": 0.35,
        "peak_order": 9,
        "min_gap_samples": 15,
        "prominence": 0.20,
        "min_peak_gap_ms": 3000,
        "min_peak_delta": 0.12,
        "min_amplitude_ratio": 0.45,
    },
    "dead": {"cutoff": 0.4, "min_gap_samples": 12, "prominence": 0.15},
}


@dataclass
class RepCounterState:
    counted_peak_times: set = field(default_factory=set)
    session_max_amplitude: float = 0.0
    rep_counting_started_ms: int = 0
    movement_detected: bool = False


def get_rep_config(exercise):
    cfg = dict(DEFAULTS)
    if exercise in REP_CONFIG:
        cfg.update(REP_CONFIG[exercise])
    return cfg


def add_acc_magnitude(df):
    out = df.copy()
    out["acc_r"] = np.sqrt(out["acc_x"] ** 2 + out["acc_y"] ** 2 + out["acc_z"] ** 2)
    return out


def _local_peak_metrics(signal, idx):
    idx = int(idx)
    left = max(0, idx - PEAK_WINDOW)
    right = min(len(signal), idx + PEAK_WINDOW + 1)
    window = signal[left:right]
    baseline = float(np.min(window))
    peak = float(signal[idx])
    span = float(np.max(window) - baseline)
    delta = peak - baseline
    score = (delta / span) if span > 0 else 0.0
    return delta, score


def filter_peak_indices(signal, indexes, min_gap_samples):
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


def filter_by_prominence(signal, indexes, min_prominence):
    if len(indexes) == 0:
        return indexes

    kept = []
    for idx in indexes:
        _, score = _local_peak_metrics(signal, idx)
        if score >= min_prominence:
            kept.append(int(idx))
    return np.array(kept, dtype=int)


def find_peak_indices(signal, cfg):
    if len(signal) < ORDER * 3:
        return np.array([], dtype=int)

    indexes = argrelextrema(signal, np.greater, order=cfg["peak_order"])[0]
    indexes = filter_peak_indices(signal, indexes, cfg["min_gap_samples"])
    return filter_by_prominence(signal, indexes, cfg["prominence"])


def prepare_filtered_signal(df_resampled, exercise):
    """Return (filtered_signal, datetime_index) at 5 Hz, or (None, None)."""
    if exercise == "rest" or df_resampled is None or len(df_resampled) < MIN_ROWS:
        return None, None

    cfg = get_rep_config(exercise)
    column = cfg["column"]
    data = add_acc_magnitude(df_resampled)

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
        return None, None

    signal = filtered[f"{column}_lowpass"].values
    if len(signal) < ORDER * 3:
        return None, None

    return signal, data.index


def peaks_from_df(df_resampled, exercise, after_ms=0):
    """Return [(peak_ms, amplitude_delta), ...] from a 5 Hz dataframe."""
    if exercise == "rest" or df_resampled is None or len(df_resampled) < MIN_ROWS:
        return []

    data = add_acc_magnitude(df_resampled)
    if after_ms > 0:
        data = data[data.index >= pd.to_datetime(after_ms, unit="ms")]
        if len(data) < MIN_ROWS:
            return []

    signal, index = prepare_filtered_signal(data, exercise)
    if signal is None:
        return []

    cfg = get_rep_config(exercise)
    peak_indices = find_peak_indices(signal, cfg)

    peaks = []
    for idx in peak_indices:
        delta, _ = _local_peak_metrics(signal, idx)
        peak_ms = int(index[idx].timestamp() * 1000)
        peaks.append((peak_ms, delta))
    return peaks


def find_peak_times_ms(df_resampled, exercise, after_ms=0):
    """Return peak timestamps (ms) — backward-compatible helper."""
    return [ms for ms, _ in peaks_from_df(df_resampled, exercise, after_ms=after_ms)]


def count_reps_in_set(df_resampled, exercise):
    """Batch rep count for a full resampled set."""
    return len(peaks_from_df(df_resampled, exercise))


def check_movement_detected(df_resampled, exercise):
    """True when recent acc activity exceeds threshold (≈1 s at 5 Hz)."""
    if df_resampled is None or len(df_resampled) < 3:
        return False

    cfg = get_rep_config(exercise)
    window = int(cfg["movement_window_samples"])
    data = add_acc_magnitude(df_resampled)
    tail = data.iloc[-window:]
    if len(tail) < 3:
        return False

    acc_r = tail["acc_r"].values
    return float(np.std(acc_r)) >= cfg["movement_std_threshold"]


def should_accept_peak(peak_ms, amplitude, state, cfg):
    if peak_ms < state.rep_counting_started_ms:
        return False
    if peak_ms in state.counted_peak_times:
        return False
    if amplitude < cfg["min_peak_delta"]:
        return False
    if state.session_max_amplitude > 0:
        min_amp = cfg["min_amplitude_ratio"] * state.session_max_amplitude
        if amplitude < min_amp:
            return False
    if state.counted_peak_times:
        last_peak = max(state.counted_peak_times)
        if peak_ms - last_peak < cfg["min_peak_gap_ms"]:
            return False
    return True


def register_peaks(peaks, state, exercise):
    """Register new peaks into state; return updated rep count."""
    cfg = get_rep_config(exercise)
    for peak_ms, amplitude in sorted(peaks):
        if not should_accept_peak(peak_ms, amplitude, state, cfg):
            continue
        state.counted_peak_times.add(peak_ms)
        state.session_max_amplitude = max(state.session_max_amplitude, amplitude)
    return len(state.counted_peak_times)
