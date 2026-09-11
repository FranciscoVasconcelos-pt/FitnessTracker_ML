"""Smoke test live rep counting on a saved phone CSV."""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "app"))
from live_features import readings_to_resampled_df  # noqa: E402
from live_reps import count_reps_in_df  # noqa: E402


def load_csv(path):
    df = pd.read_csv(path)
    return [
        {
            "t": float(row["epoch_ms"]),
            "acc_x": float(row["acc_x"]),
            "acc_y": float(row["acc_y"]),
            "acc_z": float(row["acc_z"]),
            "gyro_x": float(row.get("gyro_x") or 0),
            "gyro_y": float(row.get("gyro_y") or 0),
            "gyro_z": float(row.get("gyro_z") or 0),
        }
        for _, row in df.iterrows()
    ]


def main():
    phone = ROOT / "data" / "raw" / "phone"
    for pattern, label, expected in [
        ("*squat-heavy*.csv", "squat", 5),
        ("*bench-heavy*.csv", "bench", 5),
        ("*dead-heavy*.csv", "dead", 5),
    ]:
        csv = sorted(phone.glob(pattern), key=lambda p: p.stat().st_size, reverse=True)[0]
        readings = load_csv(csv)
        resampled = readings_to_resampled_df(readings)
        reps = count_reps_in_df(resampled, label)
        print(f"{csv.name}: {reps} reps (expected ~{expected})")


if __name__ == "__main__":
    main()
