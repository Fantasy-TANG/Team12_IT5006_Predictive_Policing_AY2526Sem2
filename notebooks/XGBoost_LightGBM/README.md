## Overview

This program uses optimized gradient boosting frameworks to predict the likelihood of an arrest following a criminal incident in Chicago, utilizing a dataset of approximately 2.8 million records.

## File Descriptions

| File Name | Description |
| --- | --- |
| `model_training_XGBoost_LightGBM.ipynb` | Comprehensive notebook covering preprocessing, hyperparameter tuning, and evaluation. |
| `chicago_train.parquet` | Processed training dataset from 2014 to 2024. |
| `chicago_test.parquet` | 2025 dataset used for final performance validation. |
| `model_xgb.pkl` | Serialized XGBoost model optimized via GridSearchCV. |
| `model_gbm.pkl` | Serialized LightGBM model optimized via RandomizedSearchCV. |
| `/results` | Plots of the optimized models. |


## Hyperparameter Selection

The models were tuned to balance predictive power with generalization, specifically addressing the class imbalance (18.7% arrest rate).

### XGBoost

The objective function was optimized using GridSearchCV:

* **`max_depth`**: 8 
* **`min_child_weight`**: 1
* **`scale_pos_weight`**: Fixed at **4.347** 
* **`subsample`**: 0.8

### LightGBM

Optimization focused on leaf-wise growth efficiency using RandomizedSearchCV:

* **`num_leaves`**: 31
* **`max_depth`**: 20
* **`min_child_samples`**: 500
* **`learning_rate`**: 0.01
* **`colsample_bytree`**: 0.7
* **`subsample`**: 0.7

## Result Comparison

The following table summarizes the performance metrics from the final evaluation phase:

| Metric | XGBoost | LightGBM |
| --- | --- | --- |
| **ROC-AUC** | **0.8625** | 0.8564 |
| **Average Precision** | **0.6563** | 0.6460 |
| **F1-macro** | 0.7305 | 0.7305 |
| **Training Latency** | Moderate (30s) | **Exceptional (<10s)** |

## Conclusion 

XGBoost provides the highest predictive stability and accuracy for strategic planning, while LightGBM offers superior efficiency for real-time retraining and deployment scenarios.
