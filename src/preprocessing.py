"""
preprocessing.py
-----------------
Loads the raw Credit Card Fraud dataset (Kaggle: creditcard.csv), does a
stratified train/test split (BEFORE any resampling, to avoid data leakage —
same principle used in Week 4), scales Time/Amount, and writes the
processed splits to data/processed/.

Usage:
    python src/preprocessing.py
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

RAW_PATH = os.path.join("data", "raw", "creditcard.csv")
PROCESSED_DIR = os.path.join("data", "processed")

TARGET_COL = "Class"
TEST_SIZE = 0.2
RANDOM_STATE = 42


def load_raw_data(path: str = RAW_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Could not find {path}. Download the Kaggle Credit Card Fraud "
            f"dataset and place it at data/raw/creditcard.csv "
            f"(or point RAW_PATH at your own dataset)."
        )
    return pd.read_csv(path)


def scale_features(df: pd.DataFrame) -> pd.DataFrame:
    """Scale Time and Amount (V1-V28 are already PCA-scaled in the Kaggle set)."""
    df = df.copy()
    scaler = StandardScaler()
    cols_to_scale = [c for c in ["Time", "Amount"] if c in df.columns]
    if cols_to_scale:
        df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
    return df


def split_and_save(df: pd.DataFrame, target_col: str = TARGET_COL) -> None:
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,  # keep the (tiny) fraud rate proportional in both splits
    )

    X_train.to_csv(os.path.join(PROCESSED_DIR, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(PROCESSED_DIR, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(PROCESSED_DIR, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(PROCESSED_DIR, "y_test.csv"), index=False)

    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    print(f"Train fraud rate: {y_train.mean():.4%}")
    print(f"Test fraud rate:  {y_test.mean():.4%}")
    print(f"Saved processed splits to {PROCESSED_DIR}/")


def main():
    df = load_raw_data()
    df = scale_features(df)
    split_and_save(df)


if __name__ == "__main__":
    main()
