import joblib
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_INTERIM = ROOT / "data" / "interim"
MODEL_PATH = ROOT / "models" / "exercise_classifier.pkl"

EXERCISE_NAMES = {
    "bench": "Bench Press",
    "squat": "Squat",
    "row": "Row",
    "ohp": "Overhead Press",
    "dead": "Deadlift",
    "rest": "Rest",
}


def load_model(model_path=MODEL_PATH):
    return joblib.load(model_path)


def predict_exercises(df, artifact):
    model = artifact["model"]
    features = artifact["features"]
    X = np.ascontiguousarray(df[features].to_numpy())
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)
    prob_df = pd.DataFrame(probabilities, columns=model.classes_, index=df.index)
    return predictions, prob_df


def format_exercise(label):
    return EXERCISE_NAMES.get(label, label)


def main():
    artifact = load_model()
    df = pd.read_pickle(DATA_INTERIM / "03_data_features.pkl")

    test_df = df[df["participant"] == artifact["test_participant"]]
    predictions, _ = predict_exercises(test_df, artifact)

    results = test_df[["label", "participant", "set"]].copy()
    results["timestamp"] = results.index.strftime("%Y-%m-%d %H:%M:%S.%f").str[:-3]
    results["actual_exercise"] = results["label"].map(format_exercise)
    results["predicted_exercise"] = pd.Series(predictions, index=results.index).map(
        format_exercise
    )
    results["correct"] = np.where(
        results["label"] == predictions, "Yes", "No"
    )

    results = results[
        [
            "timestamp",
            "participant",
            "set",
            "actual_exercise",
            "predicted_exercise",
            "correct",
        ]
    ]

    participant = artifact["test_participant"]
    accuracy = (results["correct"] == "Yes").mean()
    print(f"Overall accuracy (participant {participant}): {accuracy * 100:.1f}%\n")

    print("Accuracy by exercise:")
    accuracy_by_exercise = (
        test_df.assign(correct=test_df["label"].values == predictions)
        .groupby("label")["correct"]
        .mean()
    )
    for label, score in accuracy_by_exercise.items():
        print(f"  - {format_exercise(label)}: {score * 100:.1f}%")
    print()

    print(f"All predictions (participant {participant}):")
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.max_colwidth", None)
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
