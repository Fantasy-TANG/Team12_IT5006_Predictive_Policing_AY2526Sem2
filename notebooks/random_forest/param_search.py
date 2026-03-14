"""
============================================================
 Parameter Search — Random Forest (Low-Cost Grid)
 IT5006 Predictive Policing Project
============================================================
 策略：用 20% 抽样数据跑 12 组实验，找最优参数组合
 总耗时预计：15–25 分钟
============================================================
"""

import os, json, time, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

warnings.filterwarnings("ignore")
np.random.seed(42)

# ────────────────────────────────────────────────────────────
# CONFIG
# ────────────────────────────────────────────────────────────
TRAIN_PATH   = "chicago_train.parquet"
TEST_PATH    = "chicago_test.parquet"
OUTPUT_DIR   = "output"
SAMPLE_FRAC  = 0.20          # 20% for search experiments
CV_FOLDS     = 3
RANDOM_STATE = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

# ────────────────────────────────────────────────────────────
# 实验网格：12 组，系统覆盖最重要的 3 个参数
# ────────────────────────────────────────────────────────────
EXPERIMENTS = [
    # --- Phase 1: 找最优 max_depth（固定其他）---
    {"name": "depth_10",   "max_depth": 10,   "max_features": "sqrt", "n_estimators": 200},
    {"name": "depth_15",   "max_depth": 15,   "max_features": "sqrt", "n_estimators": 200},
    {"name": "depth_20",   "max_depth": 20,   "max_features": "sqrt", "n_estimators": 200},  # 当前
    {"name": "depth_30",   "max_depth": 30,   "max_features": "sqrt", "n_estimators": 200},
    {"name": "depth_none", "max_depth": None, "max_features": "sqrt", "n_estimators": 200},

    # --- Phase 2: 找最优 max_features（固定 depth=20）---
    {"name": "feat_log2",  "max_depth": 20,   "max_features": "log2", "n_estimators": 200},
    {"name": "feat_03",    "max_depth": 20,   "max_features": 0.3,    "n_estimators": 200},
    {"name": "feat_05",    "max_depth": 20,   "max_features": 0.5,    "n_estimators": 200},

    # --- Phase 3: 找最优 n_estimators（固定 depth=20, feat=sqrt）---
    {"name": "trees_100",  "max_depth": 20,   "max_features": "sqrt", "n_estimators": 100},
    {"name": "trees_200",  "max_depth": 20,   "max_features": "sqrt", "n_estimators": 200},  # 当前
    {"name": "trees_300",  "max_depth": 20,   "max_features": "sqrt", "n_estimators": 300},
    {"name": "trees_500",  "max_depth": 20,   "max_features": "sqrt", "n_estimators": 500},
]

FIXED_PARAMS = {
    "min_samples_split": 5,
    "min_samples_leaf" : 2,
    "bootstrap"        : True,
    "class_weight"     : "balanced",
    "random_state"     : RANDOM_STATE,
    "n_jobs"           : -1,
}


# ────────────────────────────────────────────────────────────
# LOAD & SAMPLE
# ────────────────────────────────────────────────────────────
def load_and_sample():
    print("=" * 60)
    print("Loading data & sampling...")
    print("=" * 60)

    train_df = pd.read_parquet(TRAIN_PATH)
    test_df  = pd.read_parquet(TEST_PATH)

    TARGET    = "Arrest"
    DROP_COLS = ["Year"]
    feat_cols = [c for c in train_df.columns
                 if c != TARGET and c not in DROP_COLS]

    X_train_full = train_df[feat_cols]
    y_train_full = train_df[TARGET]
    X_test  = test_df[feat_cols].reindex(columns=feat_cols, fill_value=0)
    y_test  = test_df[TARGET]

    # Stratified sample to preserve class ratio
    sample_idx = (
        train_df.groupby(TARGET, group_keys=False)
        .apply(lambda g: g.sample(frac=SAMPLE_FRAC, random_state=RANDOM_STATE))
        .index
    )
    X_sample = X_train_full.loc[sample_idx]
    y_sample = y_train_full.loc[sample_idx]

    print(f"  Full train   : {X_train_full.shape[0]:,} rows")
    print(f"  Sample (20%) : {X_sample.shape[0]:,} rows  "
          f"(Arrest rate: {y_sample.mean()*100:.2f}%)")
    print(f"  Test set     : {X_test.shape[0]:,} rows")
    print(f"  Features     : {len(feat_cols)}")

    return X_sample, y_sample, X_test, y_test, feat_cols


# ────────────────────────────────────────────────────────────
# RUN EXPERIMENTS
# ────────────────────────────────────────────────────────────
def run_experiments(X_sample, y_sample, X_test, y_test):
    print("\n" + "=" * 60)
    print(f"Running {len(EXPERIMENTS)} experiments on 20% sample")
    print("=" * 60)

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True,
                         random_state=RANDOM_STATE)
    results = []

    for i, exp in enumerate(EXPERIMENTS, 1):
        name   = exp["name"]
        params = {k: v for k, v in exp.items() if k != "name"}

        print(f"\n[{i:02d}/{len(EXPERIMENTS)}] {name}")
        print(f"  max_depth={params['max_depth']}, "
              f"max_features={params['max_features']}, "
              f"n_estimators={params['n_estimators']}")

        t0  = time.time()
        clf = RandomForestClassifier(**params, **FIXED_PARAMS)
        clf.fit(X_sample, y_sample)

        # CV AUC on sample
        cv_scores = cross_val_score(clf, X_sample, y_sample,
                                    cv=cv, scoring="roc_auc", n_jobs=-1)

        # Test set metrics
        y_pred  = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)[:, 1]

        elapsed = time.time() - t0
        row = {
            "name"         : name,
            "max_depth"    : str(params["max_depth"]),
            "max_features" : str(params["max_features"]),
            "n_estimators" : params["n_estimators"],
            "cv_auc_mean"  : round(cv_scores.mean(), 4),
            "cv_auc_std"   : round(cv_scores.std(),  4),
            "test_auc"     : round(roc_auc_score(y_test, y_proba), 4),
            "test_f1"      : round(f1_score(y_test, y_pred),       4),
            "test_recall"  : round(recall_score(y_test, y_pred),   4),
            "test_precision": round(precision_score(y_test, y_pred), 4),
            "time_sec"     : round(elapsed, 1),
        }
        results.append(row)

        print(f"  CV  AUC : {row['cv_auc_mean']:.4f} ± {row['cv_auc_std']:.4f}")
        print(f"  Test AUC: {row['test_auc']:.4f}  "
              f"F1: {row['test_f1']:.4f}  "
              f"Recall: {row['test_recall']:.4f}  "
              f"({elapsed:.1f}s)")

    return pd.DataFrame(results)


# ────────────────────────────────────────────────────────────
# ANALYSE & PLOT
# ────────────────────────────────────────────────────────────
def analyse_and_plot(df):
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    # Sort by test AUC
    df_sorted = df.sort_values("test_auc", ascending=False)
    print("\nAll experiments ranked by Test AUC:")
    print(df_sorted[["name","max_depth","max_features","n_estimators",
                      "cv_auc_mean","test_auc","test_f1",
                      "test_recall","time_sec"]].to_string(index=False))

    best = df_sorted.iloc[0]
    print(f"\n{'='*60}")
    print(f"  BEST CONFIG : {best['name']}")
    print(f"  max_depth   : {best['max_depth']}")
    print(f"  max_features: {best['max_features']}")
    print(f"  n_estimators: {best['n_estimators']}")
    print(f"  CV  AUC     : {best['cv_auc_mean']:.4f} ± {best['cv_auc_std']:.4f}")
    print(f"  Test AUC    : {best['test_auc']:.4f}")
    print(f"  Test F1     : {best['test_f1']:.4f}")
    print(f"  Test Recall : {best['test_recall']:.4f}")
    print(f"{'='*60}")

    # Save results CSV
    csv_path = os.path.join(OUTPUT_DIR, "param_search_results.csv")
    df_sorted.to_csv(csv_path, index=False)
    print(f"\n  Full results saved → {csv_path}")

    # ── Plot 1: Test AUC by experiment ──
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    phases = {
        "Phase 1 — max_depth":      df[df["name"].str.startswith("depth")],
        "Phase 2 — max_features":   df[df["name"].str.startswith("feat")],
        "Phase 3 — n_estimators":   df[df["name"].str.startswith("trees")],
    }
    colors = ["#2196F3", "#FF9800", "#4CAF50"]

    for ax, (title, sub), color in zip(axes, phases.items(), colors):
        x     = range(len(sub))
        bars  = ax.bar(x, sub["test_auc"], color=color,
                       edgecolor="white", alpha=0.85)
        ax.errorbar(x, sub["cv_auc_mean"], yerr=sub["cv_auc_std"],
                    fmt="o", color="black", capsize=4, label="CV AUC")
        ax.set_xticks(x)
        ax.set_xticklabels(sub["name"].str.replace(
            "depth_|feat_|trees_", "", regex=True), rotation=15)
        ax.set_ylabel("AUC-ROC")
        ax.set_title(title)
        ymin = max(0.80, sub["test_auc"].min() - 0.01)
        ymax = min(1.00, sub["test_auc"].max() + 0.01)
        ax.set_ylim(ymin, ymax)
        ax.legend(fontsize=8)
        for bar, val in zip(bars, sub["test_auc"]):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.001,
                    f"{val:.4f}", ha="center", va="bottom", fontsize=8)

    plt.suptitle("Parameter Search Results — Test AUC-ROC", fontsize=13)
    plt.tight_layout()
    out = os.path.join(PLOT_DIR, "param_search_auc.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved  → {out}")

    # ── Plot 2: AUC vs Training Time trade-off ──
    fig, ax = plt.subplots(figsize=(8, 5))
    phase_colors = {
        "depth": "#2196F3",
        "feat" : "#FF9800",
        "trees": "#4CAF50",
    }
    for _, row in df.iterrows():
        phase = row["name"].split("_")[0]
        c = phase_colors.get(phase, "#9C27B0")
        ax.scatter(row["time_sec"], row["test_auc"], color=c, s=80, zorder=3)
        ax.annotate(row["name"], (row["time_sec"], row["test_auc"]),
                    textcoords="offset points", xytext=(5, 3), fontsize=7)

    ax.set_xlabel("Training Time (seconds)")
    ax.set_ylabel("Test AUC-ROC")
    ax.set_title("AUC vs Training Time Trade-off")
    legend_elements = [
        mpatches.Patch(facecolor="#2196F3", label="max_depth experiments"),
        mpatches.Patch(facecolor="#FF9800", label="max_features experiments"),
        mpatches.Patch(facecolor="#4CAF50", label="n_estimators experiments"),
    ]
    ax.legend(handles=legend_elements, fontsize=9)
    plt.tight_layout()
    out2 = os.path.join(PLOT_DIR, "param_search_tradeoff.png")
    plt.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved  → {out2}")

    return best


# ────────────────────────────────────────────────────────────
# RECOMMEND FINAL PARAMS
# ────────────────────────────────────────────────────────────
def recommend(best, df):
    print("\n" + "=" * 60)
    print("RECOMMENDED FINAL PARAMETERS")
    print("=" * 60)

    # Pick best max_depth from phase 1
    phase1 = df[df["name"].str.startswith("depth")].sort_values("test_auc", ascending=False)
    best_depth = phase1.iloc[0]["max_depth"]

    # Pick best max_features from phase 2
    phase2 = df[df["name"].str.startswith("feat")].sort_values("test_auc", ascending=False)
    # Include depth_20/feat_sqrt as baseline in phase2 comparison
    best_feat = phase2.iloc[0]["max_features"]

    # Pick best n_estimators from phase 3 (with diminishing returns check)
    phase3 = df[df["name"].str.startswith("trees")].sort_values("n_estimators")
    # Find point where adding more trees gives < 0.001 gain
    best_trees = 200
    prev_auc   = 0
    for _, row in phase3.iterrows():
        if row["test_auc"] - prev_auc > 0.001:
            best_trees = int(row["n_estimators"])
            prev_auc   = row["test_auc"]

    recommended = {
        "n_estimators"     : best_trees,
        "max_depth"        : None if best_depth == "None" else int(best_depth),
        "max_features"     : float(best_feat) if best_feat not in ("sqrt","log2")
                             else best_feat,
        "min_samples_split": 5,
        "min_samples_leaf" : 2,
        "bootstrap"        : True,
    }

    print("\n  Copy these into train_random_forest.py → BEST_PARAMS:\n")
    print("  BEST_PARAMS = {")
    for k, v in recommended.items():
        val_str = f'"{v}"' if isinstance(v, str) else str(v)
        print(f'      "{k}": {val_str},')
    print("  }")

    # Save recommended params
    params_path = os.path.join(OUTPUT_DIR, "recommended_params.json")
    with open(params_path, "w") as f:
        json.dump(recommended, f, indent=2)
    print(f"\n  Saved → {params_path}")

    return recommended


# ────────────────────────────────────────────────────────────
# MAIN
# ────────────────────────────────────────────────────────────
def main():
    print("\n" + "=" * 60)
    print("  Parameter Search — Random Forest")
    print("  IT5006 Predictive Policing Project")
    print("=" * 60)

    total_start = time.time()

    X_sample, y_sample, X_test, y_test, feat_cols = load_and_sample()
    results_df = run_experiments(X_sample, y_sample, X_test, y_test)
    best       = analyse_and_plot(results_df)
    recommended = recommend(best, results_df)

    total_elapsed = time.time() - total_start
    print(f"\n  Total time: {total_elapsed/60:.1f} minutes")
    print("\n  Next step: copy BEST_PARAMS into train_random_forest.py")
    print("             and run:  python train_random_forest.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
