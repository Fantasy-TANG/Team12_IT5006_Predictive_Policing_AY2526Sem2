# Random Forest — Arrest Prediction

## Overview
Binary classification model predicting whether a crime incident results in an arrest,
trained on the Chicago Crime Dataset (2015–2024), tested on 2025 held-out data.

## Files
| File | Description |
|------|-------------|
| `train_random_forest.py` | Full training pipeline: load → train → evaluate → save |
| `param_search.py` | 3-phase hyperparameter search (12 experiments, 20% sample) |
| `feature_engineering_.ipynb` | Feature engineering: OHE, cyclical encoding, VIF check |
| `output/model_metrics.json` | Final model performance metrics |
| `output/feature_columns.json` | Feature list required for inference |
| `output/param_search_results.csv` | Full hyperparameter search results |
| `output/plots/` | Evaluation figures |

## Model Not Included
`rf_arrest_model.joblib` (1.9 GB) exceeds GitHub's file size limit and is excluded.
To reproduce the model, run:
```bash
python train_random_forest.py
```

## Final Model Configuration
| Parameter | Value |
|-----------|-------|
| n_estimators | 200 |
| max_depth | 30 |
| max_features | sqrt |
| class_weight | balanced |
| min_samples_split | 5 |
| min_samples_leaf | 2 |
| random_state | 42 |

## Results (2025 Test Set)
| Metric | Score |
|--------|-------|
| AUC-ROC | 0.8563 |
| Recall (Arrest) | 0.6311 |
| F1-Score (Arrest) | 0.5460 |
| CV AUC (3-fold) | 0.8747 ± 0.0003 |

## How to Run
```bash
pip install pandas scikit-learn matplotlib seaborn joblib pyarrow
python train_random_forest.py
```
