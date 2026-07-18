"""Kriteria 2 Skilled/Advanced - hyperparameter tuning with manual logging.

File ini sengaja tidak menggunakan autolog. Semua parameter, metric, model, dan
artifact dicatat secara manual agar berbeda dari modelling.py.
"""
from __future__ import annotations

from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    RocCurveDisplay,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import GridSearchCV

EXPERIMENT_NAME = "Breast Cancer Prediction - Skilled Manual Logging"
TARGET_COLUMN = "target"
DATA_DIR = Path("breast_cancer_preprocessing")
ARTIFACT_DIR = Path("training_artifacts")
PIP_REQUIREMENTS = ["mlflow==2.19.0", "pandas", "numpy", "scikit-learn==1.5.2"]


def load_data():
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    X_train = train_df.drop(columns=[TARGET_COLUMN])
    y_train = train_df[TARGET_COLUMN].astype(int)
    X_test = test_df.drop(columns=[TARGET_COLUMN])
    y_test = test_df[TARGET_COLUMN].astype(int)
    return X_train, y_train, X_test, y_test


def make_artifacts(model, X_test, y_test, y_pred, y_score, model_name: str) -> list[Path]:
    ARTIFACT_DIR.mkdir(exist_ok=True)
    paths: list[Path] = []

    report_path = ARTIFACT_DIR / f"classification_report_{model_name}.json"
    report_path.write_text(json.dumps(classification_report(y_test, y_pred, output_dict=True), indent=2))
    paths.append(report_path)

    cm_path = ARTIFACT_DIR / f"confusion_matrix_{model_name}.png"
    ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred)).plot()
    plt.title(f"Confusion Matrix - {model_name}")
    plt.tight_layout()
    plt.savefig(cm_path)
    plt.close()
    paths.append(cm_path)

    roc_path = ARTIFACT_DIR / f"roc_curve_{model_name}.png"
    RocCurveDisplay.from_predictions(y_test, y_score)
    plt.title(f"ROC Curve - {model_name}")
    plt.tight_layout()
    plt.savefig(roc_path)
    plt.close()
    paths.append(roc_path)

    if hasattr(model, "feature_importances_"):
        fi_path = ARTIFACT_DIR / f"feature_importance_{model_name}.csv"
        pd.DataFrame({"feature": X_test.columns, "importance": model.feature_importances_}).sort_values(
            "importance", ascending=False
        ).to_csv(fi_path, index=False)
        paths.append(fi_path)

    return paths


def train_and_log(model_name: str, estimator, param_grid: dict, X_train, y_train, X_test, y_test) -> str:
    grid = GridSearchCV(estimator, param_grid, cv=3, scoring="f1", n_jobs=1)
    grid.fit(X_train, y_train)
    model = grid.best_estimator_
    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test)[:, 1]

    metrics = {
        "test_accuracy": accuracy_score(y_test, y_pred),
        "test_precision": precision_score(y_test, y_pred),
        "test_recall": recall_score(y_test, y_pred),
        "test_f1_score": f1_score(y_test, y_pred),
        "test_roc_auc": roc_auc_score(y_test, y_score),
        "best_cv_f1": grid.best_score_,
    }
    artifact_paths = make_artifacts(model, X_test, y_test, y_pred, y_score, model_name)

    with mlflow.start_run(run_name=model_name) as run:
        mlflow.log_param("model_name", model_name)
        mlflow.log_params(grid.best_params_)
        for key, value in metrics.items():
            mlflow.log_metric(key, value)
        for artifact_path in artifact_paths:
            mlflow.log_artifact(str(artifact_path), artifact_path="evaluation_artifacts")
        mlflow.sklearn.log_model(model, artifact_path="model", pip_requirements=PIP_REQUIREMENTS)
        Path("run_id.txt").write_text(run.info.run_id)
        print(f"{model_name} run_id={run.info.run_id}")
        print(json.dumps(metrics, indent=2))
        return run.info.run_id


def main() -> None:
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment(EXPERIMENT_NAME)
    X_train, y_train, X_test, y_test = load_data()

    train_and_log(
        "RandomForest_tuned_manual_logging",
        RandomForestClassifier(random_state=42, class_weight="balanced"),
        {"n_estimators": [30, 50], "max_depth": [4, 8], "min_samples_split": [2, 5]},
        X_train,
        y_train,
        X_test,
        y_test,
    )

    train_and_log(
        "LogisticRegression_tuned_manual_logging",
        LogisticRegression(random_state=42, max_iter=1000),
        {"C": [0.1, 1.0, 10.0], "solver": ["liblinear"]},
        X_train,
        y_train,
        X_test,
        y_test,
    )


if __name__ == "__main__":
    main()
