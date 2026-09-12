"""
train_random_forest.py
-----------------------
Week 5: trains a Random Forest classifier on the processed train/test split
and saves the fitted model to models/random_forest.pkl.

Usage:
    python src/train_random_forest.py
"""

import os
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

PROCESSED_DIR = os.path.join("data", "processed")
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "random_forest.pkl")

RANDOM_STATE = 42


def load_processed_data():
    X_train = pd.read_csv(os.path.join(PROCESSED_DIR, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(PROCESSED_DIR, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(PROCESSED_DIR, "y_train.csv")).squeeze("columns")
    y_test = pd.read_csv(os.path.join(PROCESSED_DIR, "y_test.csv")).squeeze("columns")
    return X_train, X_test, y_train, y_test


def train_random_forest(X_train, y_train, **params) -> RandomForestClassifier:
    default_params = dict(
        n_estimators=100,
        max_depth=None,
        class_weight="balanced",  # dataset is heavily imbalanced (~0.17% fraud)
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    default_params.update(params)
    model = RandomForestClassifier(**default_params)
    model.fit(X_train, y_train)
    return model


def main():
    X_train, X_test, y_train, y_test = load_processed_data()

    model = train_random_forest(X_train, y_train)
    y_pred = model.predict(X_test)

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Random Forest trained and saved to {MODEL_PATH}")

    from sklearn.metrics import accuracy_score
    print(f"Test accuracy (untuned): {accuracy_score(y_test, y_pred):.4f}")


if __name__ == "__main__":
    main()
