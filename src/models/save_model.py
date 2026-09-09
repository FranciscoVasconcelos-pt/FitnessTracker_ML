"""Treina e guarda o Random Forest sem repetir o grid search completo do train_model.py."""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

ROOT = Path(__file__).resolve().parents[2]
DATA_INTERIM = ROOT / "data" / "interim"
MODELS_DIR = ROOT / "models"


def build_feature_set_4(df_train):
    basic_features = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
    square_features = ["acc_r", "gyro_r"]
    time_features = [f for f in df_train.columns if "_temp_" in f]
    freq_features = [f for f in df_train.columns if ("_freq" in f) or ("_pse" in f)]
    cluster_features = ["cluster"]
    feature_set_2 = list(set(basic_features + square_features))
    feature_set_3 = list(set(feature_set_2 + time_features))
    return list(set(feature_set_3 + freq_features + cluster_features))


def split_train_test(df):
    """MetaMotion: train B-E, test A. Phone / single user: hold out ~20% of sets."""
    participant_df = df.drop(["category"], axis=1)
    test_sets = None

    if participant_df["participant"].eq("A").any():
        train_mask = participant_df["participant"] != "A"
        test_mask = participant_df["participant"] == "A"
        test_participant = "A"
    else:
        unique_sets = participant_df["set"].drop_duplicates()
        n_test = max(1, int(round(len(unique_sets) * 0.2)))
        test_sets = (
            unique_sets.sample(n=n_test, random_state=0).sort_values().tolist()
        )
        train_mask = ~participant_df["set"].isin(test_sets)
        test_mask = participant_df["set"].isin(test_sets)
        test_participant = str(participant_df["participant"].iloc[0])

    feature_df = participant_df.drop(["label", "participant", "set"], axis=1)
    X_train = feature_df.loc[train_mask]
    Y_train = participant_df.loc[train_mask, "label"]
    X_test = feature_df.loc[test_mask]
    Y_test = participant_df.loc[test_mask, "label"]
    return X_train, Y_train, X_test, Y_test, test_participant, test_sets


def main():
    df = pd.read_pickle(DATA_INTERIM / "03_data_features.pkl")
    df_train = df.drop(["participant", "category", "set"], axis=1)
    feature_set_4 = build_feature_set_4(df_train)

    X_train, Y_train, X_test, Y_test, test_participant, test_sets = split_train_test(
        df
    )

    X_train_fit = np.ascontiguousarray(X_train[feature_set_4].to_numpy())
    X_test_fit = np.ascontiguousarray(X_test[feature_set_4].to_numpy())

    tuned_parameters = [
        {
            "min_samples_leaf": [2, 10, 50, 100, 200],
            "n_estimators": [10, 50, 100],
            "criterion": ["gini", "entropy"],
        }
    ]
    rf_search = GridSearchCV(
        RandomForestClassifier(), tuned_parameters, cv=5, scoring="accuracy"
    )
    rf_search.fit(X_train_fit, Y_train)

    test_accuracy = rf_search.score(X_test_fit, Y_test)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "exercise_classifier.pkl"
    artifact = {
        "model": rf_search.best_estimator_,
        "features": feature_set_4,
        "test_participant": test_participant,
        "test_accuracy": test_accuracy,
    }
    if test_sets is not None:
        artifact["test_sets"] = test_sets

    joblib.dump(artifact, model_path)

    import sys

    sys.path.insert(0, str(ROOT / "src" / "app"))
    from live_features import fit_live_artifact

    live_artifact = fit_live_artifact(
        DATA_INTERIM / "02_outliers_removed_chauvenet.pkl",
        DATA_INTERIM / "03_data_features.pkl",
    )
    live_path = MODELS_DIR / "live_artifact.pkl"
    joblib.dump(live_artifact, live_path)
    print(f"Live artifact guardado em {live_path}")

    if test_sets:
        print(
            f"Modelo guardado em {model_path} "
            f"(test accuracy: {test_accuracy:.3f}, holdout sets: {test_sets})"
        )
    else:
        print(f"Modelo guardado em {model_path} (test accuracy: {test_accuracy:.3f})")


if __name__ == "__main__":
    main()
