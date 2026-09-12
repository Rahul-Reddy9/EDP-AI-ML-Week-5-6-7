# Weeks 6–7 Report — Evaluation & Tuning

## Objective
Evaluate the Week 5 models properly (accuracy is not enough on an
imbalanced fraud dataset), then tune hyperparameters and pick the best
performer.

## Work done
- Built `src/evaluate.py` with reusable functions for:
  - Accuracy, weighted precision, weighted recall, weighted F1, ROC-AUC
  - Confusion matrix plots (`results/confusion_matrices/*.png`)
  - Per-model classification reports (`results/metrics/*_metrics.txt`)
- Built `src/tune.py`, using `GridSearchCV` (5-fold, scoring =
  `f1_weighted`) to tune:
  - **Random Forest**: `n_estimators`, `max_depth`, `min_samples_split`
  - **XGBoost**: `n_estimators`, `learning_rate`, `max_depth`
- Ran both notebooks/scripts end-to-end and produced
  `results/comparison.csv` comparing all four models (RF, tuned RF,
  XGBoost, tuned XGBoost).

## Results (sample run, synthetic placeholder data)
| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---|---|---|---|---|
| Random Forest | 0.9921 | 0.9842 | 0.9921 | 0.9881 | 0.9974 |
| XGBoost | 0.9940 | 0.9931 | 0.9940 | 0.9931 | 0.9946 |
| Tuned Random Forest | 0.9921 | 0.9842 | 0.9921 | 0.9881 | 0.9979 |
| Tuned XGBoost | 0.9950 | 0.9945 | 0.9950 | 0.9945 | 0.9910 |

*(Replace the placeholder dataset with the real Kaggle
`creditcard.csv` and re-run `preprocessing.py` → `train_*.py` →
`tune.py` → `evaluate.py` to get real numbers.)*

## Best model
Tuned XGBoost had the best F1 score on this sample run. On the real
dataset, re-check the comparison table before deciding — with such a
rare positive class, prioritize **recall** and **F1** over raw
accuracy, since a model that predicts "not fraud" for everything would
still score ~99.8% accuracy while catching zero fraud.

## Selected best-performing model
`models/tuned_xgboost.pkl` (pending re-validation on the real dataset).
