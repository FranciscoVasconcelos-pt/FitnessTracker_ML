"""Load model + live artifact and predict exercise from phone buffers."""

import joblib
import numpy as np
from pathlib import Path

from live_features import build_feature_row, readings_to_resampled_df

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


class LivePredictor:
    def __init__(self):
        self.model_artifact = joblib.load(MODEL_PATH)
        self.live_artifact = joblib.load(LIVE_ARTIFACT_PATH)
        self.buffer = []
        self.max_buffer_ms = 6000
        self.min_buffer_span_ms = 4400  # 22 rows at 5 Hz resampling
        self.min_resampled_rows = 22

    def add_readings(self, readings):
        self.buffer.extend(readings)
        self._trim_buffer()

    def _buffer_span_ms(self):
        if len(self.buffer) < 2:
            return 0
        times = [r.t if hasattr(r, "t") else r["t"] for r in self.buffer]
        return max(times) - min(times)

    def _warmup_response(self, message, **extra):
        span = self._buffer_span_ms()
        progress = min(100, int(span / self.min_buffer_span_ms * 100))
        return {
            "status": "warming_up",
            "message": message,
            "warmup_progress": progress,
            "samples": len(self.buffer),
            "buffer_span_ms": int(span),
            **extra,
        }

    def _trim_buffer(self):
        if not self.buffer:
            return
        latest = max(r.t if hasattr(r, "t") else r["t"] for r in self.buffer)
        self.buffer = [
            r
            for r in self.buffer
            if (r.t if hasattr(r, "t") else r["t"]) >= latest - self.max_buffer_ms
        ]

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

        return {
            "status": "ok",
            "exercise": label,
            "exercise_name": EXERCISE_NAMES.get(label, label),
            "confidence": round(confidence, 3),
            "resampled_rows": len(df),
            "buffer_samples": len(self.buffer),
        }

    def clear(self):
        self.buffer = []
