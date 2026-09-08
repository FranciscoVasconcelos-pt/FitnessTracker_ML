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


def main():
    df = pd.read_pickle(DATA_INTERIM / "03_data_features.pkl")
    df_train = df.drop(["participant", "category", "set"], axis=1)
    feature_set_4 = build_feature_set_4(df_train)

    participant_df = df.drop(["set", "category"], axis=1)
    X_train = participant_df[participant_df["participant"] != "A"].drop(
        ["label", "participant"], axis=1
    )
    Y_train = participant_df[participant_df["participant"] != "A"]["label"]
    X_test = participant_df[participant_df["participant"] == "A"].drop(
        ["label", "participant"], axis=1
    )
    Y_test = participant_df[participant_df["participant"] == "A"]["label"]

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
    joblib.dump(
        {
            "model": rf_search.best_estimator_,
            "features": feature_set_4,
            "test_participant": "A",
            "test_accuracy": test_accuracy,
        },
        model_path,
    )
    print(f"Modelo guardado em {model_path} (test accuracy: {test_accuracy:.3f})")


if __name__ == "__main__":
    main()
