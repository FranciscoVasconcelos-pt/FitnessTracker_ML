"""Quick diagnosis after accuracy drop."""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MODEL = ROOT / "models" / "exercise_classifier.pkl"

EXERCISE_NAMES = {
    "bench": "Bench Press",
    "squat": "Squat",
    "row": "Row",
    "ohp": "Overhead Press",
    "dead": "Deadlift",
    "rest": "Rest",
}


def main():
    artifact = joblib.load(MODEL)
    df = pd.read_pickle(DATA / "interim" / "03_data_features.pkl")
    test_sets = artifact["test_sets"]
    test_df = df[df["set"].isin(test_sets)].copy()

    model = artifact["model"]
    features = artifact["features"]
    X = np.ascontiguousarray(test_df[features].to_numpy())
    preds = model.predict(X)

    test_df = test_df.assign(pred=preds, correct=test_df["label"].values == preds)

    print(f"Overall holdout accuracy: {test_df['correct'].mean():.1%}")
    print(f"Holdout sets ({len(test_sets)}): {test_sets}\n")

    print("Accuracy by exercise (holdout):")
    for label, group in test_df.groupby("label"):
        print(f"  {EXERCISE_NAMES.get(label, label):18} {group['correct'].mean():.1%}  ({group['set'].nunique()} sets)")

    print("\nWorst holdout sets (by row accuracy):")
    by_set = test_df.groupby(["set", "label"]).agg(
        rows=("label", "size"),
        accuracy=("correct", "mean"),
    ).reset_index()
    by_set = by_set.sort_values("accuracy")
    for _, row in by_set.head(10).iterrows():
        name = EXERCISE_NAMES.get(row["label"], row["label"])
        print(f"  set {int(row['set']):2d}  {name:18}  {row['accuracy']:.0%}  ({int(row['rows'])} rows)")

    print("\nConfusion (holdout rows):")
    confusion = pd.crosstab(
        test_df["label"].map(lambda x: EXERCISE_NAMES.get(x, x)),
        pd.Series(preds, index=test_df.index).map(lambda x: EXERCISE_NAMES.get(x, x)),
    )
    print(confusion.to_string())

    print("\nCSV files — duration & samples:")
    phone = DATA / "raw" / "phone"
    rows = []
    for csv in sorted(phone.glob("*.csv")):
        d = pd.read_csv(csv)
        t0, t1 = d["epoch_ms"].iloc[0], d["epoch_ms"].iloc[-1]
        duration_s = (t1 - t0) / 1000
        rows.append(
            {
                "file": csv.name,
                "label": d["label"].iloc[0],
                "samples": len(d),
                "duration_s": round(duration_s, 1),
            }
        )
    summary = pd.DataFrame(rows)
    short = summary[summary["duration_s"] < 15]
    if not short.empty:
        print("\n⚠ Very short recordings (<15 s) — may be low quality:")
        print(short.to_string(index=False))

    evening = summary[summary["file"].str.contains("_213")]
    if not evening.empty:
        print(f"\nNew evening batch (_213*): {len(evening)} files")
        print(evening.groupby("label")["file"].count().to_string())


if __name__ == "__main__":
    main()
