"""Load model + live artifact and predict exercise from phone buffers."""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from live_features import build_feature_row, readings_to_resampled_df
from live_reps import find_peak_times_ms, get_rep_config

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models" / "exercise_classifier.pkl"
LIVE_ARTIFACT_PATH = ROOT / "models" / "live_artifact.pkl"

EXERCISE_NAMES = {
    "bench": "Bench Press",
    "squat": "Squat",
    "row": "Row",
    "ohp": "Overhead Press",
    "dead": "Deadlift",
    "rest": "Rest",
}

IDLE_AFTER_LAST_REP_MS = 4500
IDLE_ACTIVITY_STD = 0.022
IDLE_ACTIVITY_RANGE = 0.055


class LivePredictor:
    def __init__(self):
        self.model_artifact = joblib.load(MODEL_PATH)
        self.live_artifact = joblib.load(LIVE_ARTIFACT_PATH)
        self.buffer = []
        self.max_buffer_ms = 6000
        self.min_buffer_span_ms = 4400
        self.min_resampled_rows = 22
        self.rep_buffer = []
        self.rep_count = 0
        self.rep_target = 5
        self.rep_counting_started_ms = 0
        self.counted_peak_times = set()
        self.rep_counting_active = False
        self.rep_set_complete = False
        self.locked_rep_exercise = None

    def add_readings(self, readings):
        self.buffer.extend(readings)
        if self.rep_counting_active and not self.rep_set_complete:
            self.rep_buffer.extend(readings)
        self._trim_buffer()
        self._trim_rep_buffer()

    def _trim_rep_buffer(self, max_ms=60000):
        if not self.rep_buffer:
            return
        latest = max(r.t if hasattr(r, "t") else r["t"] for r in self.rep_buffer)
        self.rep_buffer = [
            r
            for r in self.rep_buffer
            if (r.t if hasattr(r, "t") else r["t"]) >= latest - max_ms
        ]

    def _buffer_span_ms(self):
        if len(self.buffer) < 2:
            return 0
        times = [r.t if hasattr(r, "t") else r["t"] for r in self.buffer]
        return max(times) - min(times)

    def _warmup_response(self, message, **extra):
        span = self._buffer_span_ms()
        progress = min(100, int(span / self.min_buffer_span_ms * 100))
        payload = {
            "status": "warming_up",
            "message": message,
            "warmup_progress": progress,
            "samples": len(self.buffer),
            "buffer_span_ms": int(span),
        }
        payload.update(self._update_reps())
        payload.update(extra)
        return payload

    def _trim_buffer(self):
        if not self.buffer:
            return
        latest = max(r.t if hasattr(r, "t") else r["t"] for r in self.buffer)
        self.buffer = [
            r
            for r in self.buffer
            if (r.t if hasattr(r, "t") else r["t"]) >= latest - self.max_buffer_ms
        ]

    def _latest_reading_time(self):
        if not self.rep_buffer:
            return max(
                (r.t if hasattr(r, "t") else r["t"] for r in self.buffer),
                default=0,
            )
        return max(r.t if hasattr(r, "t") else r["t"] for r in self.rep_buffer)

    def _rep_readings_for_count(self):
        if self.rep_counting_started_ms <= 0:
            return []
        return [
            r
            for r in self.rep_buffer
            if (r.t if hasattr(r, "t") else r["t"]) >= self.rep_counting_started_ms
        ]

    def _min_peak_gap_ms(self):
        if not self.locked_rep_exercise:
            return 2200
        return get_rep_config(self.locked_rep_exercise)["min_peak_gap_ms"]

    def _register_new_peaks(self, peak_times_ms):
        min_gap = self._min_peak_gap_ms()
        for peak_ms in sorted(peak_times_ms):
            if peak_ms < self.rep_counting_started_ms:
                continue
            if peak_ms in self.counted_peak_times:
                continue
            if self.counted_peak_times:
                last_peak = max(self.counted_peak_times)
                if peak_ms - last_peak < min_gap:
                    continue
            self.counted_peak_times.add(peak_ms)

        self.rep_count = len(self.counted_peak_times)

    def _is_resting_after_set(self, rep_df):
        """True when movement has been low for several seconds after the last rep."""
        if self.rep_count == 0 or not self.counted_peak_times:
            return False

        last_peak_ms = max(self.counted_peak_times)
        latest_ms = int(rep_df.index[-1].timestamp() * 1000)
        if latest_ms - last_peak_ms < IDLE_AFTER_LAST_REP_MS:
            return False

        tail = rep_df.iloc[-10:]
        acc_r = np.sqrt(tail["acc_x"] ** 2 + tail["acc_y"] ** 2 + tail["acc_z"] ** 2)
        return (
            float(acc_r.std()) < IDLE_ACTIVITY_STD
            and float(acc_r.max() - acc_r.min()) < IDLE_ACTIVITY_RANGE
        )

    def _peak_search_start_ms(self):
        if not self.counted_peak_times:
            return self.rep_counting_started_ms
        return max(self.rep_counting_started_ms, max(self.counted_peak_times) - 800)

    def _rep_status(self):
        locked = self.locked_rep_exercise
        counting = bool(
            self.rep_counting_active
            and not self.rep_set_complete
            and locked
            and locked != "rest"
        )
        return {
            "reps": self.rep_count,
            "rep_target": self.rep_target,
            "counting": counting,
            "set_complete": self.rep_set_complete,
            "locked_exercise": locked,
            "locked_exercise_name": EXERCISE_NAMES.get(locked, locked) if locked else None,
        }

    def predict(self):
        span = self._buffer_span_ms()
        if span < self.min_buffer_span_ms:
            return self._warmup_response(
                "A aquecer… move o telemóvel durante o exercício."
            )

        df = readings_to_resampled_df(self.buffer)
        if len(df) < self.min_resampled_rows:
            return self._warmup_response(
                "A construir janela de features…",
                resampled_rows=len(df),
            )

        try:
            row = build_feature_row(df, self.live_artifact)
        except ValueError as exc:
            return self._warmup_response(str(exc), resampled_rows=len(df))

        feature_names = self.model_artifact["features"]
        values = [float(row.get(name, 0.0)) for name in feature_names]
        X = np.ascontiguousarray([values])

        model = self.model_artifact["model"]
        label = model.predict(X)[0]
        probabilities = model.predict_proba(X)[0]
        confidence = float(max(probabilities))

        result = {
            "status": "ok",
            "exercise": label,
            "exercise_name": EXERCISE_NAMES.get(label, label),
            "confidence": round(confidence, 3),
            "resampled_rows": len(df),
            "buffer_samples": len(self.buffer),
        }
        result.update(self._update_reps())
        return result

    def _update_reps(self):
        if not self.rep_counting_active or not self.locked_rep_exercise:
            return self._rep_status()

        if self.rep_set_complete:
            return self._rep_status()

        rep_readings = self._rep_readings_for_count()
        if len(rep_readings) < 15:
            return self._rep_status()

        rep_df = readings_to_resampled_df(rep_readings)
        if self._is_resting_after_set(rep_df):
            self.rep_set_complete = True
            return self._rep_status()

        search_from = self._peak_search_start_ms()
        peak_times = find_peak_times_ms(
            rep_df, self.locked_rep_exercise, after_ms=search_from
        )
        self._register_new_peaks(peak_times)

        if self.rep_count >= self.rep_target and self._is_resting_after_set(rep_df):
            self.rep_set_complete = True

        return self._rep_status()

    def lock_exercise(self, exercise):
        if exercise and exercise != "rest":
            self.locked_rep_exercise = exercise

    def set_rep_target(self, target):
        self.rep_target = max(1, int(target))

    def reset_reps(self, exercise=None):
        if exercise and exercise != "rest":
            self.locked_rep_exercise = exercise
        self.rep_count = 0
        self.counted_peak_times = set()
        self.rep_counting_started_ms = self._latest_reading_time()
        self.rep_set_complete = False
        self.rep_buffer = []
        self.rep_counting_active = bool(
            self.locked_rep_exercise and self.locked_rep_exercise != "rest"
        )

    def end_set(self):
        self.rep_set_complete = True
        self.rep_counting_active = False

    def clear(self):
        self.buffer = []
        self.rep_buffer = []
        self.rep_count = 0
        self.counted_peak_times = set()
        self.rep_counting_started_ms = 0
        self.rep_counting_active = False
        self.rep_set_complete = False
        self.locked_rep_exercise = None
