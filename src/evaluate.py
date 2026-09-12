"""
evaluate.py
-----------
Weeks 6-7: evaluation utilities — accuracy / precision / recall / F1,
confusion matrix plots, and classification reports — reusable from
notebooks or scripts, plus a standalone runner that evaluates every
saved model in models/ and writes results/comparison.csv.

Usage:
    python src/evaluate.py
"""

import os
import json
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")  # headless-safe backend
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)

PROCESSED_DIR = os.path.join("data", "processed")
MODEL_DIR = "models"
RESULTS_DIR = "results"
METRICS_DIR = os.path.join(RESULTS_DIR, "metrics")
CM_DIR = os.path.join(RESULTS_DIR, "confusion_matrices")


def load_processed_data():
    X_train = pd.read_csv(os.path.join(PROCESSED_DIR, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(PROCESSED_DIR, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(PROCESSED_DIR, "y_train.csv")).squeeze("columns")
    y_test = pd.read_csv(os.path.join(PROCESSED_DIR, "y_test.csv")).squeeze("columns")
    return X_train, X_test, y_train, y_test


def compute_metrics(y_true, y_pred, y_proba=None) -> dict:
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_score": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }
    if y_proba is not None:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
        except ValueError:
            pass
    return metrics


def plot_confusion_matrix(y_true, y_pred, title: str, save_path: str) -> None:
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()


def save_metrics_report(y_true, y_pred, metrics: dict, model_name: str) -> None:
    os.makedirs(METRICS_DIR, exist_ok=True)
    report_path = os.path.join(METRICS_DIR, f"{model_name}_metrics.txt")
    with open(report_path, "w") as f:
        f.write(f"Metrics for {model_name}\n")
        f.write("=" * 40 + "\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v:.4f}\n")
        f.write("\nClassification Report\n")
        f.write("-" * 40 + "\n")
        f.write(classification_report(y_true, y_pred, zero_division=0))
    print(f"Saved metrics report to {report_path}")


def evaluate_model(model, model_name: str, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    y_proba = None
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]

    metrics = compute_metrics(y_test, y_pred, y_proba)
    save_metrics_report(y_test, y_pred, metrics, model_name)
    plot_confusion_matrix(
        y_test, y_pred,
        title=f"Confusion Matrix - {model_name}",
        save_path=os.path.join(CM_DIR, f"{model_name}_cm.png"),
    )
    return metrics


def main():
    _, X_test, _, y_test = load_processed_data()

    model_files = {
        "random_forest": os.path.join(MODEL_DIR, "random_forest.pkl"),
        "xgboost": os.path.join(MODEL_DIR, "xgboost.pkl"),
        "tuned_random_forest": os.path.join(MODEL_DIR, "tuned_random_forest.pkl"),
        "tuned_xgboost": os.path.join(MODEL_DIR, "tuned_xgboost.pkl"),
    }

    comparison_rows = []
    for name, path in model_files.items():
        if not os.path.exists(path):
            continue
        model = joblib.load(path)
        metrics = evaluate_model(model, name, X_test, y_test)
        row = {"Model": name}
        row.update({k.replace("_", " ").title(): round(v, 4) for k, v in metrics.items()})
        comparison_rows.append(row)

    if comparison_rows:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        comparison_df = pd.DataFrame(comparison_rows)
        comparison_path = os.path.join(RESULTS_DIR, "comparison.csv")
        comparison_df.to_csv(comparison_path, index=False)
        print(f"\nSaved model comparison to {comparison_path}")
        print(comparison_df.to_string(index=False))
    else:
        print("No trained models found in models/. Run the train_*.py scripts first.")


if __name__ == "__main__":
    main()
