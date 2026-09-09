"""Quick smoke test for live prediction on a saved phone CSV."""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "app"))
from predict_live import LivePredictor  # noqa: E402


def main():
    csv = next((ROOT / "data/raw/phone").glob("*ohp*.csv"))
    df = pd.read_csv(csv)
    print("CSV:", csv.name, "label:", df["label"].iloc[0], "rows:", len(df))

    readings = [
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

    predictor = LivePredictor()
    predictor.add_readings(readings)
    result = predictor.predict()
    print("Full set:", result)

    predictor.clear()
    chunk = max(1, len(readings) // 10)
    stream_result = None
    for i in range(0, len(readings), chunk):
        predictor.add_readings(readings[i : i + chunk])
        stream_result = predictor.predict()
    print("Streamed:", stream_result)


if __name__ == "__main__":
    main()
