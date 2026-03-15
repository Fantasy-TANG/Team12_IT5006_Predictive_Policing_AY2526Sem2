# Logistic Regression

## Overview
Logistic Regression serves as the baseline binary classification model, 
predicting whether a reported crime incident results in an arrest.
Trained on the Chicago Crime Dataset (2014–2024), tested on 2025 
held-out data.

---

## Files

| File | Description |
|------|-------------|
| `LR model training.ipynb` | Training, evaluation, coefficient analysis|
| `figures` | Plots |


---

## Model Configuration

| Parameter | Value |
|-----------|-------|
| Solver | saga | 
| Regularization | L2, C = 1.0 | 
| class_weight | balanced |
| max_iter | 300 | 
| Training set | Full | 

---

## Results

| Metric | Score |
|--------|-------|
| Precision | 0.56 |
| Recall | 0.48 |
| F1-Score | 0.52 |
| F1-macro | 0.72 |
| AUC-ROC | 0.835 |

---

## Conclusion
With an AUC of 0.835, Logistic Regression sets a strong baseline and reliably captures the “Arrest” signal.
