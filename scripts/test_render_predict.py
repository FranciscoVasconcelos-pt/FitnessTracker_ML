"""Replay a phone CSV against local and Render /predict for comparison."""

import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "app"))
from predict_live import LivePredictor  # noqa: E402

RENDER_URL = "https://ml-fitness-tracker.onrender.com"


def load_readings(csv_path):
    df = pd.read_csv(csv_path)
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


def stream_local(readings, chunks=40):
    predictor = LivePredictor()
    chunk_size = max(1, len(readings) // chunks)
    results = []
    for i in range(0, len(readings), chunk_size):
        predictor.add_readings(readings[i : i + chunk_size])
        results.append(predictor.predict())
    return results


def post_json(path, payload=None, timeout=120):
    data = None if payload is None else json.dumps(payload).encode()
    headers = {"Content-Type": "application/json"} if payload else {}
    req = urllib.request.Request(
        f"{RENDER_URL}{path}", data=data, headers=headers, method="POST"
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def stream_render(readings, chunks=20):
    post_json("/predict/reset")
    chunk_size = max(1, len(readings) // chunks)
    results = []
    for i in range(0, len(readings), chunk_size):
        batch = readings[i : i + chunk_size]
        results.append(post_json("/predict", {"readings": batch}))
    return results


def summarize(label, results, expected):
    ok = [r for r in results if r.get("status") == "ok"]
    print(f"\n{label} (expected ~{expected})")
    print(f"  batches: {len(results)}, ok: {len(ok)}, warmup: {len(results) - len(ok)}")
    if ok:
        last = ok[-1]
        print(
            f"  last ok: exercise={last.get('exercise')} "
            f"conf={last.get('confidence')} name={last.get('exercise_name')}"
        )
        print(f"  last 5 ok exercises: {[r.get('exercise') for r in ok[-5:]]}")


def main():
    phone = ROOT / "data" / "raw" / "phone"
    for pattern, expected in [
        ("*bench-heavy*.csv", "bench"),
        ("*squat-heavy*.csv", "squat"),
        ("*rest-heavy*.csv", "rest"),
    ]:
        csv = sorted(phone.glob(pattern), key=lambda p: p.stat().st_size, reverse=True)[0]
        readings = load_readings(csv)
        print(f"CSV: {csv.name} ({len(readings)} samples)")

        local_results = stream_local(readings)
        summarize("LOCAL", local_results, expected)

        try:
            render_results = stream_render(readings)
            summarize("RENDER", render_results, expected)
        except urllib.error.URLError as exc:
            print(f"RENDER failed: {exc}")


if __name__ == "__main__":
    main()
