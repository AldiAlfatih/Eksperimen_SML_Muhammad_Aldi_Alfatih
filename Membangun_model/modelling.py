"""Kriteria 2 Basic - MLflow autolog only.

File ini sengaja hanya menggunakan mlflow.sklearn.autolog() dan tidak memakai
manual logging. File tuning manual dibuat terpisah pada modelling_tuning.py.
"""
from __future__ import annotations

from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

EXPERIMENT_NAME = "Breast Cancer Prediction - Basic Autolog"
TARGET_COLUMN = "target"
DATA_DIR = Path("breast_cancer_preprocessing")


def load_data():
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    X_train = train_df.drop(columns=[TARGET_COLUMN])
    y_train = train_df[TARGET_COLUMN].astype(int)
    X_test = test_df.drop(columns=[TARGET_COLUMN])
    y_test = test_df[TARGET_COLUMN].astype(int)
    return X_train, y_train, X_test, y_test


def main() -> None:
    mlflow.set_experiment(EXPERIMENT_NAME)
    mlflow.sklearn.autolog(log_input_examples=True, log_model_signatures=True)

    X_train, y_train, X_test, y_test = load_data()
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        random_state=42,
        class_weight="balanced",
    )

    with mlflow.start_run(run_name="basic_random_forest_autolog"):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("Precision:", precision_score(y_test, y_pred))
        print("Recall:", recall_score(y_test, y_pred))
        print("F1 Score:", f1_score(y_test, y_pred))


if __name__ == "__main__":
    main()
