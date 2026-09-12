"""
tune.py
-------
Weeks 6-7: hyperparameter tuning for Random Forest and XGBoost via
GridSearchCV, optimizing for weighted F1 (a better target than raw
accuracy on an imbalanced fraud dataset). Saves the best estimators to
models/tuned_random_forest.pkl and models/tuned_xgboost.pkl.

Usage:
    python src/tune.py                 # tunes both models
    python src/tune.py --model rf      # tunes Random Forest only
    python src/tune.py --model xgb     # tunes XGBoost only
"""

import os
import argparse
import pandas as pd
import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

PROCESSED_DIR = os.path.join("data", "processed")
MODEL_DIR = "models"
RANDOM_STATE = 42

RF_PARAM_GRID = {
    "n_estimators": [100, 200],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5],
}

XGB_PARAM_GRID = {
    "n_estimators": [100, 200],
    "learning_rate": [0.05, 0.1],
    "max_depth": [3, 5, 7],
}


def load_processed_data():
    X_train = pd.read_csv(os.path.join(PROCESSED_DIR, "X_train.csv"))
    y_train = pd.read_csv(os.path.join(PROCESSED_DIR, "y_train.csv")).squeeze("columns")
    return X_train, y_train


def tune_random_forest(X_train, y_train) -> RandomForestClassifier:
    rf = RandomForestClassifier(
        class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
    )
    grid_search = GridSearchCV(
        rf, RF_PARAM_GRID, cv=5, scoring="f1_weighted", n_jobs=-1, verbose=1
    )
    grid_search.fit(X_train, y_train)
    print("Random Forest best params:", grid_search.best_params_)
    print(f"Random Forest best CV F1: {grid_search.best_score_:.4f}")
    return grid_search.best_estimator_


def tune_xgboost(X_train, y_train) -> XGBClassifier:
    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    scale_pos_weight = neg / pos if pos > 0 else 1.0

    xgb = XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    grid_search = GridSearchCV(
        xgb, XGB_PARAM_GRID, cv=5, scoring="f1_weighted", n_jobs=-1, verbose=1
    )
    grid_search.fit(X_train, y_train)
    print("XGBoost best params:", grid_search.best_params_)
    print(f"XGBoost best CV F1: {grid_search.best_score_:.4f}")
    return grid_search.best_estimator_


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["rf", "xgb", "both"], default="both")
    args = parser.parse_args()

    X_train, y_train = load_processed_data()
    os.makedirs(MODEL_DIR, exist_ok=True)

    if args.model in ("rf", "both"):
        best_rf = tune_random_forest(X_train, y_train)
        joblib.dump(best_rf, os.path.join(MODEL_DIR, "tuned_random_forest.pkl"))
        print(f"Saved tuned Random Forest to {MODEL_DIR}/tuned_random_forest.pkl")

    if args.model in ("xgb", "both"):
        best_xgb = tune_xgboost(X_train, y_train)
        joblib.dump(best_xgb, os.path.join(MODEL_DIR, "tuned_xgboost.pkl"))
        print(f"Saved tuned XGBoost to {MODEL_DIR}/tuned_xgboost.pkl")


if __name__ == "__main__":
    main()
