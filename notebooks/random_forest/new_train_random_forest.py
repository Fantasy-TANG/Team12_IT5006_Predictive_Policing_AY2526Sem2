"""
============================================================
 Arrest Prediction — Random Forest
 IT5006 Predictive Policing Project
============================================================
 Task    : Binary classification — will a crime lead to arrest?
 Train   : chicago_train.parquet  (Year < 2025)
 Test    : chicago_test.parquet   (Year >= 2025)
 Output  : output/rf_arrest_model.joblib
           output/feature_columns.json
           output/model_metrics.json
           output/plots/  (PNG figures)
============================================================
 Requirements:
   pip install pandas scikit-learn matplotlib seaborn joblib pyarrow
============================================================
"""

import os
import json
import warnings
import joblib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
    confusion_matrix, RocCurveDisplay, PrecisionRecallDisplay,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.calibration import calibration_curve

warnings.filterwarnings("ignore")
np.random.seed(42)

# ────────────────────────────────────────────────────────────
# 0.  CONFIG
# ────────────────────────────────────────────────────────────
TRAIN_PATH   = "chicago_train.parquet"
TEST_PATH    = "chicago_test.parquet"
OUTPUT_DIR   = "output"
CV_FOLDS     = 3
RANDOM_STATE = 42

# Pre-set best parameters (tuned for large crime datasets)
BEST_PARAMS = {
    "n_estimators"     : 200,
    "max_depth"        : 30,
    "max_features"     : "sqrt",  # 比0.3快很多，只差0.002
    "min_samples_split": 5,
    "min_samples_leaf" : 2,
    "bootstrap"        : True,
}

os.makedirs(OUTPUT_DIR, exist_ok=True)
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)


# ────────────────────────────────────────────────────────────
# 1.  LOAD DATA
# ────────────────────────────────────────────────────────────
def load_data():
    print("=" * 60)
    print("STEP 1 — Loading data")
    print("=" * 60)
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df  = pd.read_parquet(TEST_PATH)
    print(f"  Train shape : {train_df.shape}")
    print(f"  Test  shape : {test_df.shape}")
    print(f"  Train Arrest rate : {train_df['Arrest'].mean()*100:.2f}%")
    print(f"  Test  Arrest rate : {test_df['Arrest'].mean()*100:.2f}%")
    return train_df, test_df


# ────────────────────────────────────────────────────────────
# 2.  PREPARE FEATURES
# ────────────────────────────────────────────────────────────
def prepare_features(train_df, test_df):
    print("\n" + "=" * 60)
    print("STEP 2 — Preparing features")
    print("=" * 60)
    TARGET    = "Arrest"
    DROP_COLS = ["Year"]
    feature_cols = [c for c in train_df.columns if c != TARGET and c not in DROP_COLS]
    X_train = train_df[feature_cols]
    y_train = train_df[TARGET]
    X_test  = test_df[feature_cols]
    y_test  = test_df[TARGET]
    X_test  = X_test.reindex(columns=X_train.columns, fill_value=0)
    print(f"  Number of features : {len(feature_cols)}")
    feat_path = os.path.join(OUTPUT_DIR, "feature_columns.json")
    with open(feat_path, "w") as f:
        json.dump(feature_cols, f)
    print(f"  Feature list saved → {feat_path}")
    return X_train, y_train, X_test, y_test, feature_cols


# ────────────────────────────────────────────────────────────
# 3.  CLASS IMBALANCE CHECK
# ────────────────────────────────────────────────────────────
def check_imbalance(y_train):
    print("\n" + "=" * 60)
    print("STEP 3 — Class imbalance check")
    print("=" * 60)
    counts = y_train.value_counts()
    ratio  = counts[0] / counts[1]
    print(f"  No Arrest (0) : {counts[0]:,}")
    print(f"  Arrest    (1) : {counts[1]:,}")
    print(f"  Ratio  (0:1)  : {ratio:.2f}:1")
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(["No Arrest (0)", "Arrest (1)"], counts.values,
           color=["#4C72B0", "#DD8452"], edgecolor="white")
    ax.set_ylabel("Count")
    ax.set_title("Target Class Distribution (Train)")
    for i, v in enumerate(counts.values):
        ax.text(i, v + counts.sum() * 0.01,
                f"{v:,}\n({v/len(y_train)*100:.1f}%)", ha="center", fontsize=9)
    plt.tight_layout()
    out = os.path.join(PLOT_DIR, "01_class_distribution.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved → {out}")
    class_weight = "balanced" if ratio > 2 else None
    print(f"  class_weight set to: \"{class_weight}\"")
    return class_weight


# ────────────────────────────────────────────────────────────
# 4.  BASELINE MODEL
# ────────────────────────────────────────────────────────────
def train_baseline(X_train, y_train, X_test, y_test, class_weight):
    print("\n" + "=" * 60)
    print("STEP 4 — Baseline Random Forest (100 trees, max_depth=10)")
    print("=" * 60)
    rf_base = RandomForestClassifier(
        n_estimators=100, max_depth=10,
        class_weight=class_weight, random_state=RANDOM_STATE, n_jobs=-1,
    )
    rf_base.fit(X_train, y_train)
    y_pred_base  = rf_base.predict(X_test)
    y_proba_base = rf_base.predict_proba(X_test)[:, 1]
    base_metrics = {
        "Accuracy" : accuracy_score(y_test, y_pred_base),
        "Precision": precision_score(y_test, y_pred_base),
        "Recall"   : recall_score(y_test, y_pred_base),
        "F1-Score" : f1_score(y_test, y_pred_base),
        "AUC-ROC"  : roc_auc_score(y_test, y_proba_base),
    }
    print("\n  Baseline metrics:")
    for k, v in base_metrics.items():
        print(f"    {k:<12}: {v:.4f}")
    return rf_base, y_pred_base, y_proba_base, base_metrics


# ────────────────────────────────────────────────────────────
# 5.  FINAL MODEL
# ────────────────────────────────────────────────────────────
def train_final_model(X_train, y_train, X_test, y_test, class_weight):
    print("\n" + "=" * 60)
    print("STEP 5 — Training final model with pre-set best parameters")
    print("=" * 60)
    print("  Parameters:")
    for k, v in BEST_PARAMS.items():
        print(f"    {k}: {v}")
    rf_final = RandomForestClassifier(
        **BEST_PARAMS, class_weight=class_weight,
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    rf_final.fit(X_train, y_train)
    print("  Model trained.")
    y_pred  = rf_final.predict(X_test)
    y_proba = rf_final.predict_proba(X_test)[:, 1]
    metrics = {
        "Accuracy" : accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall"   : recall_score(y_test, y_pred),
        "F1-Score" : f1_score(y_test, y_pred),
        "AUC-ROC"  : roc_auc_score(y_test, y_proba),
    }
    print("\n  Final model test metrics:")
    for k, v in metrics.items():
        print(f"    {k:<12}: {v:.4f}")
    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["No Arrest", "Arrest"]))
    print(f"  Running {CV_FOLDS}-fold CV (AUC-ROC)...")
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(rf_final, X_train, y_train,
                                cv=cv, scoring="roc_auc", n_jobs=-1)
    print(f"  CV AUC-ROC : {cv_scores.round(4)}")
    print(f"  Mean ± Std : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    return rf_final, y_pred, y_proba, metrics, cv_scores


# ────────────────────────────────────────────────────────────
# 6.  PLOTS
# ────────────────────────────────────────────────────────────
def plot_confusion_matrix(y_test, y_pred):
    cm     = confusion_matrix(y_test, y_pred)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, data, fmt, title in zip(
        axes, [cm, cm_pct], ["d", ".1f"],
        ["Confusion Matrix (Counts)", "Confusion Matrix (Row %)"],
    ):
        sns.heatmap(data, annot=True, fmt=fmt, cmap="Blues", ax=ax,
                    xticklabels=["No Arrest", "Arrest"],
                    yticklabels=["No Arrest", "Arrest"])
        ax.set_title(title); ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    plt.suptitle("Random Forest — Confusion Matrix (Test Set)", fontsize=13, y=1.02)
    plt.tight_layout()
    out = os.path.join(PLOT_DIR, "02_confusion_matrix.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved → {out}")

def plot_roc_pr(y_test, y_proba):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    RocCurveDisplay.from_predictions(y_test, y_proba, ax=axes[0],
                                     name="Random Forest (Final)", color="#2196F3")
    axes[0].plot([0,1],[0,1],"k--",lw=1,label="Random"); axes[0].set_title("ROC Curve")
    axes[0].legend(loc="lower right")
    PrecisionRecallDisplay.from_predictions(y_test, y_proba, ax=axes[1],
                                            name="Random Forest (Final)", color="#FF5722")
    axes[1].axhline(y_test.mean(), color="k", linestyle="--", lw=1,
                    label=f"No-skill ({y_test.mean():.2f})")
    axes[1].set_title("Precision–Recall Curve"); axes[1].legend(loc="upper right")
    plt.suptitle("Random Forest — Discrimination Curves (Test Set)", fontsize=13)
    plt.tight_layout()
    out = os.path.join(PLOT_DIR, "03_roc_pr_curves.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved → {out}")

def plot_cv_scores(cv_scores):
    fig, ax = plt.subplots(figsize=(6, 4))
    folds = [f"Fold {i+1}" for i in range(len(cv_scores))]
    bars  = ax.bar(folds, cv_scores, color="#4CAF50", edgecolor="white")
    ax.axhline(cv_scores.mean(), color="red", linestyle="--", lw=1.5,
               label=f"Mean = {cv_scores.mean():.4f}")
    ymin = max(0, cv_scores.min()-0.05); ymax = min(1, cv_scores.max()+0.05)
    ax.set_ylim(ymin, ymax); ax.set_ylabel("AUC-ROC")
    ax.set_title(f"{CV_FOLDS}-Fold Cross-Validation AUC-ROC"); ax.legend()
    for bar, score in zip(bars, cv_scores):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+(ymax-ymin)*0.01,
                f"{score:.4f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    out = os.path.join(PLOT_DIR, "04_cv_scores.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved → {out}")

def plot_feature_importance(rf_final, feature_cols):
    importances = pd.Series(rf_final.feature_importances_, index=feature_cols)
    top30 = importances.nlargest(30).sort_values()
    color_map = {"Primary Type":"#2196F3","Location Description":"#FF9800","District":"#4CAF50"}
    colors = []
    for f in top30.index:
        matched = False
        for key, col in color_map.items():
            if key in f:
                colors.append(col); matched = True; break
        if not matched:
            colors.append("#9C27B0")
    fig, ax = plt.subplots(figsize=(8, 9))
    ax.barh(top30.index, top30.values, color=colors, edgecolor="white")
    ax.set_xlabel("Feature Importance (Mean Decrease in Impurity)")
    ax.set_title("Top 30 Feature Importances\nRandom Forest (Final)")
    ax.legend(handles=[
        mpatches.Patch(facecolor="#2196F3", label="Primary Type"),
        mpatches.Patch(facecolor="#FF9800", label="Location Description"),
        mpatches.Patch(facecolor="#4CAF50", label="District"),
        mpatches.Patch(facecolor="#9C27B0", label="Other"),
    ], loc="lower right", fontsize=8)
    plt.tight_layout()
    out = os.path.join(PLOT_DIR, "05_feature_importance.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved → {out}")
    print("\n  Top 10 Most Important Features:")
    for feat, val in importances.nlargest(10).items():
        print(f"    {feat:<55} {val:.4f}")

def plot_calibration(y_test, y_proba):
    prob_true, prob_pred = calibration_curve(y_test, y_proba, n_bins=10)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(prob_pred, prob_true, "s-", color="#2196F3", label="Random Forest")
    ax.plot([0,1],[0,1],"k--",lw=1,label="Perfect calibration")
    ax.set_xlabel("Mean Predicted Probability"); ax.set_ylabel("Fraction of Positives")
    ax.set_title("Calibration Curve"); ax.legend()
    plt.tight_layout()
    out = os.path.join(PLOT_DIR, "06_calibration_curve.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved → {out}")

def plot_score_distribution(y_test, y_proba):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(y_proba[y_test==0], bins=50, alpha=0.6, color="#4C72B0",
            label="No Arrest (0)", density=True)
    ax.hist(y_proba[y_test==1], bins=50, alpha=0.6, color="#DD8452",
            label="Arrest (1)", density=True)
    ax.set_xlabel("Predicted Probability of Arrest"); ax.set_ylabel("Density")
    ax.set_title("Score Distribution by True Class"); ax.legend()
    plt.tight_layout()
    out = os.path.join(PLOT_DIR, "07_score_distribution.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved → {out}")

def plot_baseline_vs_final(base_metrics, tuned_metrics):
    labels = list(base_metrics.keys())
    x = np.arange(len(labels)); width = 0.35
    fig, ax = plt.subplots(figsize=(9, 4))
    b1 = ax.bar(x-width/2, list(base_metrics.values()),  width,
                label="Baseline RF", color="#78909C", edgecolor="white")
    b2 = ax.bar(x+width/2, list(tuned_metrics.values()), width,
                label="Final RF",    color="#1976D2", edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.1); ax.set_ylabel("Score")
    ax.set_title("Baseline vs Final Random Forest — Test Set Metrics"); ax.legend()
    for bar in list(b1)+list(b2):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    out = os.path.join(PLOT_DIR, "08_baseline_vs_final.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved → {out}")

def run_all_plots(y_test, y_pred, y_proba, cv_scores,
                  rf_final, feature_cols, base_metrics, tuned_metrics):
    print("\n" + "=" * 60)
    print("STEP 6 — Generating evaluation plots")
    print("=" * 60)
    plot_confusion_matrix(y_test, y_pred)
    plot_roc_pr(y_test, y_proba)
    plot_cv_scores(cv_scores)
    plot_feature_importance(rf_final, feature_cols)
    plot_calibration(y_test, y_proba)
    plot_score_distribution(y_test, y_proba)
    plot_baseline_vs_final(base_metrics, tuned_metrics)


# ────────────────────────────────────────────────────────────
# 7.  SAVE MODEL & METRICS
# ────────────────────────────────────────────────────────────
def save_artifacts(rf_final, tuned_metrics, cv_scores):
    print("\n" + "=" * 60)
    print("STEP 7 — Saving model and metrics")
    print("=" * 60)
    model_path = os.path.join(OUTPUT_DIR, "rf_arrest_model.joblib")
    joblib.dump(rf_final, model_path)
    print(f"  Model saved → {model_path}")
    metrics_out = {
        "test_set"   : tuned_metrics,
        "cv_auc_mean": float(cv_scores.mean()),
        "cv_auc_std" : float(cv_scores.std()),
        "best_params": BEST_PARAMS,
    }
    metrics_path = os.path.join(OUTPUT_DIR, "model_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_out, f, indent=2)
    print(f"  Metrics saved → {metrics_path}")


# ────────────────────────────────────────────────────────────
# 8.  INFERENCE HELPER
# ────────────────────────────────────────────────────────────
def predict_arrest(district, location_desc, primary_type,
                   weekday, month, hour, domestic=0,
                   model=None, feature_cols=None):
    row = pd.DataFrame([{c: 0 for c in feature_cols}])
    for feat, val in {
        "Domestic"   : domestic,
        "is_weekend" : int(weekday in [5,6]),
        "hour_sin"   : np.sin(2*np.pi*hour/24),
        "hour_cos"   : np.cos(2*np.pi*hour/24),
        "month_sin"  : np.sin(2*np.pi*month/12),
        "month_cos"  : np.cos(2*np.pi*month/12),
        "weekday_sin": np.sin(2*np.pi*weekday/7),
        "weekday_cos": np.cos(2*np.pi*weekday/7),
    }.items():
        if feat in row.columns:
            row[feat] = val
    for col in [f"District_{district}",
                f"Location Description_{location_desc}",
                f"Primary Type_{primary_type}"]:
        if col in row.columns:
            row[col] = 1
        else:
            print(f"  Warning: '{col}' not found — treated as unseen (zeros)")
    proba = model.predict_proba(row)[0, 1]
    pred  = int(proba >= 0.5)
    return {"prediction": pred,
            "prediction_label": "ARREST" if pred==1 else "NO ARREST",
            "arrest_probability": round(float(proba), 4)}

def run_inference_examples(rf_final, feature_cols):
    print("\n" + "=" * 60)
    print("STEP 8 — Inference examples")
    print("=" * 60)
    examples = [
        dict(district=8,  location_desc="STREET", primary_type="THEFT",
             weekday=4, month=6,  hour=14, domestic=0),
        dict(district=11, location_desc="RESIDENCE", primary_type="BATTERY",
             weekday=5, month=12, hour=23, domestic=1),
        dict(district=1,  location_desc="PARKING LOT / GARAGE (NON RESIDENTIAL)",
             primary_type="MOTOR VEHICLE THEFT", weekday=1, month=3, hour=3, domestic=0),
    ]
    for i, ex in enumerate(examples, 1):
        result = predict_arrest(**ex, model=rf_final, feature_cols=feature_cols)
        print(f"\n  Example {i}:")
        for k, v in ex.items():
            print(f"    {k:<16}: {v}")
        print(f"    → Prediction      : {result['prediction_label']}")
        print(f"    → Arrest Prob     : {result['arrest_probability']:.2%}")


# ────────────────────────────────────────────────────────────
# MAIN
# ────────────────────────────────────────────────────────────
def main():
    print("\n" + "="*60)
    print("  Arrest Prediction — Random Forest Pipeline")
    print("  IT5006 Predictive Policing Project")
    print("="*60)

    train_df, test_df                             = load_data()
    X_train, y_train, X_test, y_test, feat_cols  = prepare_features(train_df, test_df)
    class_weight                                  = check_imbalance(y_train)
    _, _, _, base_metrics                         = train_baseline(
        X_train, y_train, X_test, y_test, class_weight)
    rf_final, y_pred, y_proba, metrics, cv_scores = train_final_model(
        X_train, y_train, X_test, y_test, class_weight)
    run_all_plots(y_test, y_pred, y_proba, cv_scores,
                  rf_final, feat_cols, base_metrics, metrics)
    save_artifacts(rf_final, metrics, cv_scores)
    run_inference_examples(rf_final, feat_cols)

    print("\n" + "="*60)
    print("  Pipeline complete!")
    print(f"  All outputs saved in: ./{OUTPUT_DIR}/")
    print("="*60)


if __name__ == "__main__":
    main()
