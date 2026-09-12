# Week 5 Report — Random Forest & XGBoost

## Objective
Train two classifiers — Random Forest and XGBoost — on the processed
Credit Card Fraud dataset (continuing from Week 4's imbalance handling).

## Work done
- Implemented `RandomForestClassifier` (`src/train_random_forest.py`,
  `notebooks/02_random_forest.ipynb`) with `class_weight="balanced"` to
  account for the ~0.17% fraud rate.
- Implemented `XGBClassifier` (`src/train_xgboost.py`,
  `notebooks/03_xgboost.ipynb`) with `scale_pos_weight` set from the
  train split's class ratio.
- Trained both models on the stratified train/test split produced in
  `src/preprocessing.py`.
- Saved trained models to `models/random_forest.pkl` and
  `models/xgboost.pkl`.

## Results (untuned, on the sample run)
| Model | Test Accuracy |
|---|---|
| Random Forest | 0.9921 |
| XGBoost | 0.9940 |

*(These numbers came from the synthetic placeholder dataset used to test
the pipeline — replace `data/raw/creditcard.csv` with the real Kaggle
dataset and re-run to get your actual figures.)*

## Next steps (Weeks 6–7)
- Evaluate both models properly with precision/recall/F1 and confusion
  matrices (accuracy alone is misleading on this imbalanced dataset).
- Tune hyperparameters with `GridSearchCV`.
- Compare tuned vs. untuned models and pick the best.
