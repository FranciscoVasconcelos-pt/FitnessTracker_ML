"""Ingest phone recordings from data/raw/phone/ into the pipeline format.

After recording sets on the phone, run this instead of make_dataset.py,
then continue with remove_outliers → build_features → save_model.
This feeds the same pipeline so exercise_classifier.pkl can be retrained on iPhone data.

Phone acc is in m/s² (DeviceMotion); we convert to g to stay close to MetaMotion units.
Gyro stays in deg/s like the original dataset.
"""

import re
from glob import glob
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PHONE_RAW = ROOT / "data" / "raw" / "phone"
DATA_INTERIM = ROOT / "data" / "interim"
GRAVITY_MS2 = 9.80665


def parse_filename(filename):
    """Parse user-bench-heavy-20260101_120000.csv style names."""
    stem = Path(filename).stem
    match = re.match(r"^(.+)-([^-]+)-([^-]+)-(\d{8}_\d{6})$", stem)
    if match:
        return match.group(1), match.group(2), match.group(3)
    return "user", "unknown", "heavy"


def load_phone_csv(filepath):
    df = pd.read_csv(filepath)
    participant = df["participant"].iloc[0] if "participant" in df.columns else parse_filename(filepath)[0]
    label = df["label"].iloc[0] if "label" in df.columns else parse_filename(filepath)[1]
    category = df["category"].iloc[0] if "category" in df.columns else parse_filename(filepath)[2]

    df.index = pd.to_datetime(df["epoch_ms"], unit="ms")

    # m/s² → g (MetaMotion CSVs use g for acceleration)
    for col in ("acc_x", "acc_y", "acc_z"):
        df[col] = df[col] / GRAVITY_MS2

    value_cols = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
    meta_cols = ["participant", "label", "category", "set"]
    sampling = {col: "mean" for col in value_cols}
    sampling.update({col: "last" for col in meta_cols[:-1]})

    df["participant"] = participant
    df["label"] = label
    df["category"] = category

    return df[value_cols + ["participant", "label", "category"]]


def main():
    files = sorted(glob(str(PHONE_RAW / "*.csv")))
    if not files:
        print(f"No CSV files found in {PHONE_RAW}")
        print("Record sets on the phone first (POST /record), then re-run.")
        return

    sessions = []
    for set_id, filepath in enumerate(files, start=1):
        df = load_phone_csv(filepath)
        df["set"] = set_id
        resampled = df.resample("200ms").agg(
            {
                "acc_x": "mean",
                "acc_y": "mean",
                "acc_z": "mean",
                "gyro_x": "mean",
                "gyro_y": "mean",
                "gyro_z": "mean",
                "participant": "last",
                "label": "last",
                "category": "last",
                "set": "last",
            }
        ).dropna(subset=["acc_x", "acc_y", "acc_z"])
        sessions.append(resampled)
        print(f"  set {set_id}: {Path(filepath).name} → {len(resampled)} rows @ 5 Hz")

    data = pd.concat(sessions)
    data["set"] = data["set"].astype(int)

    DATA_INTERIM.mkdir(parents=True, exist_ok=True)
    out_path = DATA_INTERIM / "01_data_processed_phone.pkl"
    data.to_pickle(out_path)

    print(f"\nSaved {len(data)} rows, {data['set'].nunique()} sets → {out_path}")
    print("Next: point remove_outliers.py at 01_data_processed_phone.pkl, or copy/rename")
    print("      to 01_data_processed.pkl, then run the usual pipeline + save_model.py")


if __name__ == "__main__":
    main()
