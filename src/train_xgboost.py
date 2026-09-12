"""
train_xgboost.py
-----------------
Week 5: trains an XGBoost classifier on the processed train/test split
and saves the fitted model to models/xgboost.pkl.

Usage:
    python src/train_xgboost.py
"""

import os
import pandas as pd
import joblib
from xgboost import XGBClassifier

PROCESSED_DIR = os.path.join("data", "processed")
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "xgboost.pkl")

RANDOM_STATE = 42


def load_processed_data():
    X_train = pd.read_csv(os.path.join(PROCESSED_DIR, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(PROCESSED_DIR, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(PROCESSED_DIR, "y_train.csv")).squeeze("columns")
    y_test = pd.read_csv(os.path.join(PROCESSED_DIR, "y_test.csv")).squeeze("columns")
    return X_train, X_test, y_train, y_test


def train_xgboost(X_train, y_train, **params) -> XGBClassifier:
    # scale_pos_weight compensates for class imbalance (neg/pos ratio)
    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    scale_pos_weight = neg / pos if pos > 0 else 1.0

    default_params = dict(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    default_params.update(params)
    model = XGBClassifier(**default_params)
    model.fit(X_train, y_train)
    return model


def main():
    X_train, X_test, y_train, y_test = load_processed_data()

    model = train_xgboost(X_train, y_train)
    y_pred = model.predict(X_test)

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"XGBoost trained and saved to {MODEL_PATH}")

    from sklearn.metrics import accuracy_score
    print(f"Test accuracy (untuned): {accuracy_score(y_test, y_pred):.4f}")


if __name__ == "__main__":
    main()
