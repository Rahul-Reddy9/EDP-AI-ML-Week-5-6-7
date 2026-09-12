"""
Generates a small SYNTHETIC stand-in dataset shaped like the Kaggle
Credit Card Fraud dataset (Time, V1-V28, Amount, Class) so the pipeline
can be run and tested end-to-end before you drop in the real
data/raw/creditcard.csv from Kaggle.
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
n_normal = 5000
n_fraud = 40  # keep it heavily imbalanced like the real dataset (~0.17%)

def make_rows(n, fraud):
    data = {}
    data["Time"] = rng.uniform(0, 172792, n)
    for i in range(1, 29):
        base = rng.normal(0, 1, n)
        if fraud:
            base += rng.normal(0.8, 0.5, n)  # shift fraud class slightly
        data[f"V{i}"] = base
    data["Amount"] = rng.gamma(2, 80, n) if not fraud else rng.gamma(2, 200, n)
    data["Class"] = 1 if fraud else 0
    return pd.DataFrame(data)

df = pd.concat([make_rows(n_normal, False), make_rows(n_fraud, True)], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df.to_csv("data/raw/creditcard.csv", index=False)
print(df.shape, df["Class"].value_counts().to_dict())
