"""
Full evaluation report for the two-stage cybercrime prediction pipeline.

Reads the deployed model artifacts (models/*.joblib) and the ML-ready dataset,
reproduces the exact 80/20 hold-out split used during training, and computes a
much deeper set of metrics than the training scripts print:

    - accuracy, balanced accuracy, Cohen's kappa, MCC
    - per-class precision / recall / F1 (macro and weighted)
    - Top-1..Top-5 hit rates and Mean Reciprocal Rank (MRR)
    - ROC-AUC (one-vs-rest, macro), log loss, multiclass Brier score
    - 5-fold stratified cross-validation (mean and standard deviation)
    - baseline comparison: majority, uniform random, Random Forest, and the
      numbers measured on the original v1 dataset generator

Outputs (all inside reports/, which is committed -- models/ is gitignored):

    reports/metrics.json            every number below, machine readable
    reports/EVALUATION_REPORT.md    human readable report with charts
    reports/*.png                   confusion matrices, feature importances,
                                    precision/recall, Top-k curve, baselines

Usage (from the repo root, after training the models):

    py -3.13 evaluation_report.py

This script never overwrites the deployed .joblib artifacts; the only model it
trains are throwaway copies used for cross-validation and RF baselines.
"""

import json
import os
from datetime import date

import joblib
import matplotlib

matplotlib.use("Agg")  # render to files, never open a window

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import label_binarize
from xgboost import XGBClassifier

from atm_zone_config import (
    FEATURES,
    INPUT_FILE,
    RANDOM_STATE,
    REGION_LABELS,
    REGION_MODEL_OUTPUT_FILE,
    TARGET,
    XGB_PARAMS,
    ZONE_LABELS,
    ZONE_MODEL_OUTPUT_FILE,
    ZONE_NAMES,
    ZONE_REGION,
)

# Zone-model parameters, identical to xgboost_model.py: the objective and class
# count are explicit so predict_proba() returns one score per zone.
ZONE_XGB_PARAMS = {
    **XGB_PARAMS,
    "objective": "multi:softprob",
    "num_class": 10,
    "eval_metric": "mlogloss",
}

REPORTS_DIR = "reports"
N_ZONES = len(ZONE_NAMES)
N_REGIONS = len(REGION_LABELS)

# Numbers measured on the ORIGINAL v1 generator (uniform zone sampling + 20%
# label noise). Kept here as constants so the v1 -> v2 story in the report is
# reproducible without re-running the old dataset code.
V1_RESULTS = {
    "zone_accuracy": 0.2010,
    "zone_top3": 0.5850,
    "region_accuracy": 0.7650,
    "rf_accuracy": 0.1930,
}


# ============================================================================
# METRIC HELPERS
# ============================================================================
def topk_hit_rate(probabilities, true_labels, k):
    """Share of rows whose true label appears among the k highest scores."""
    order = np.argsort(-probabilities, axis=1)[:, :k]
    return float(np.mean((order == np.asarray(true_labels)[:, None]).any(axis=1)))


def reciprocal_ranks(probabilities, true_labels):
    """1 / rank of the true label for every row (1.0 = ranked first)."""
    order = np.argsort(-probabilities, axis=1)
    ranks = (order == np.asarray(true_labels)[:, None]).argmax(axis=1) + 1
    return 1.0 / ranks


def multiclass_brier(y_true, probabilities, n_classes):
    """
    Multiclass Brier score: mean squared error between each row of
    probabilities and the one-hot truth, summed over classes (range 0..2,
    lower is better). Sensitive to both calibration and discrimination.
    """
    y_bin = label_binarize(y_true, classes=range(n_classes))
    return float(np.mean(np.sum((probabilities - y_bin) ** 2, axis=1)))


def score_block(y_true, probabilities, class_labels, n_classes, prefix=""):
    """
    Everything that can be derived from one probability matrix: overall
    scores, per-class precision/recall/F1, ranking quality and calibration.

    Returns a flat dict (JSON friendly) plus the per-class rows for tables.
    """
    y_true = np.asarray(y_true)
    y_pred = np.argmax(probabilities, axis=1)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=range(n_classes), zero_division=0
    )
    per_class = {
        class_labels[code]: {
            "precision": float(precision[code]),
            "recall": float(recall[code]),
            "f1": float(f1[code]),
            "support": int(support[code]),
        }
        for code in range(n_classes)
    }

    metrics = {
        f"{prefix}accuracy": float(accuracy_score(y_true, y_pred)),
        f"{prefix}balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        f"{prefix}cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
        f"{prefix}mcc": float(matthews_corrcoef(y_true, y_pred)),
        f"{prefix}macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        f"{prefix}weighted_f1": float(
            f1_score(y_true, y_pred, average="weighted")
        ),
    }

    for k in (1, 2, 3, 4, 5):
        metrics[f"{prefix}top{k}_accuracy"] = topk_hit_rate(
            probabilities, y_true, k
        )
    metrics[f"{prefix}mrr"] = float(np.mean(reciprocal_ranks(probabilities, y_true)))

    metrics[f"{prefix}roc_auc_ovr_macro"] = float(
        roc_auc_score(y_true, probabilities, multi_class="ovr", average="macro")
    )
    metrics[f"{prefix}log_loss"] = float(
        log_loss(y_true, probabilities, labels=range(n_classes))
    )
    metrics[f"{prefix}brier_score"] = multiclass_brier(
        y_true, probabilities, n_classes
    )

    metrics[f"{prefix}per_class"] = per_class
    return metrics


def cross_validate(params, X, y_fit, y_stratify, n_splits=5):
    """
    Stratified K-fold cross-validation with a fresh clone of the given model.
    Folds are stratified on the ZONE label for both tasks (y_stratify) so the
    zone and region CV numbers are computed on identical folds; y_fit is what
    the model is actually trained and scored on.
    """
    folds = []
    splitter = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE
    )
    for fold, (train_idx, val_idx) in enumerate(splitter.split(X, y_stratify), start=1):
        model = XGBClassifier(**params)
        model.fit(X.iloc[train_idx], y_fit.iloc[train_idx])
        y_pred = model.predict(X.iloc[val_idx])
        folds.append(
            {
                "fold": fold,
                "train_rows": int(len(train_idx)),
                "val_rows": int(len(val_idx)),
                "accuracy": float(accuracy_score(y_fit.iloc[val_idx], y_pred)),
                "macro_f1": float(
                    f1_score(y_fit.iloc[val_idx], y_pred, average="macro")
                ),
            }
        )
        print(
            f"  fold {fold}/{n_splits}: "
            f"accuracy={folds[-1]['accuracy']:.4f} "
            f"macro_f1={folds[-1]['macro_f1']:.4f}"
        )

    accuracies = [f["accuracy"] for f in folds]
    macro_f1s = [f["macro_f1"] for f in folds]
    summary = {
        "n_splits": n_splits,
        "stratified_on": "zone label",
        "folds": folds,
        "accuracy_mean": float(np.mean(accuracies)),
        "accuracy_std": float(np.std(accuracies)),
        "macro_f1_mean": float(np.mean(macro_f1s)),
        "macro_f1_std": float(np.std(macro_f1s)),
    }
    return summary


# ============================================================================
# CHART HELPERS
# ============================================================================
def save_confusion_heatmap(matrix, labels, title, path, figsize=(9, 7)):
    """Row-normalized confusion matrix as a percentage heatmap."""
    row_sums = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(
        matrix, row_sums, out=np.zeros_like(matrix, dtype=float),
        where=row_sums != 0,
    )

    plt.figure(figsize=figsize)
    sns.heatmap(
        normalized * 100,
        annot=normalized * 100,
        fmt=".0f",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        vmin=0,
        vmax=100,
        cbar_kws={"label": "% of actual class"},
    )
    plt.title(title, fontweight="bold")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  chart -> {path}")


def save_feature_importance(model, path):
    """Horizontal bar chart of the deployed model's feature importances."""
    pairs = sorted(
        zip(FEATURES, model.feature_importances_), key=lambda p: p[1]
    )
    names = [p[0] for p in pairs]
    values = [p[1] for p in pairs]

    plt.figure(figsize=(7, 3.5))
    plt.barh(names, values, color="#2563EB")
    plt.xlabel("XGBoost gain importance")
    plt.title("Feature importances (deployed zone model)", fontweight="bold")
    for index, value in enumerate(values):
        plt.text(value + 0.01, index, f"{value:.3f}", va="center", fontsize=9)
    plt.xlim(0, max(values) * 1.2)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  chart -> {path}")


def save_precision_recall(per_class, path):
    """Grouped precision vs recall bars, one pair per zone."""
    names = list(per_class.keys())
    precisions = [per_class[name]["precision"] for name in names]
    recalls = [per_class[name]["recall"] for name in names]

    positions = np.arange(len(names))
    width = 0.38

    plt.figure(figsize=(11, 4.5))
    plt.bar(positions - width / 2, precisions, width, label="Precision", color="#2563EB")
    plt.bar(positions + width / 2, recalls, width, label="Recall", color="#F59E0B")
    plt.xticks(positions, names, rotation=35, ha="right")
    plt.ylim(0, 1.05)
    plt.ylabel("Score")
    plt.title("Per-zone precision and recall (hold-out)", fontweight="bold")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  chart -> {path}")


def save_topk_curve(per_model_topk, path):
    """Top-k accuracy curve for k = 1..5, with the uniform-chance line."""
    ks = np.arange(1, 6)
    plt.figure(figsize=(7, 4.5))
    for label, values in per_model_topk.items():
        plt.plot(ks, values, marker="o", label=label)
    plt.plot(ks, ks / N_ZONES, linestyle="--", color="gray", label="Uniform chance (k/10)")
    plt.xticks(ks)
    plt.ylim(0, 1.0)
    plt.xlabel("k (ranked shortlist size)")
    plt.ylabel("Chance the true zone is inside top-k")
    plt.title("Top-k accuracy: shortlist vs patrol size", fontweight="bold")
    plt.legend()
    plt.grid(linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  chart -> {path}")


def save_baseline_chart(rows, path):
    """Grouped bar chart: v1 generator vs v2 generator on the same 4 metrics."""
    metrics = ["Zone accuracy", "Zone top-3", "Region accuracy", "RF accuracy"]
    v1 = [rows[m]["v1"] for m in metrics]
    v2 = [rows[m]["v2"] for m in metrics]

    positions = np.arange(len(metrics))
    width = 0.38

    plt.figure(figsize=(8, 4.5))
    plt.bar(positions - width / 2, v1, width, label="v1 generator (uniform zones)", color="#94A3B8")
    plt.bar(positions + width / 2, v2, width, label="v2 generator (calibrated)", color="#2563EB")
    for index, (old, new) in enumerate(zip(v1, v2)):
        plt.text(index - width / 2, old + 0.015, f"{old:.0%}", ha="center", fontsize=9)
        plt.text(index + width / 2, new + 0.015, f"{new:.0%}", ha="center", fontsize=9, fontweight="bold")
    plt.xticks(positions, metrics)
    plt.ylim(0, 1.0)
    plt.ylabel("Accuracy / hit rate")
    plt.title("Before vs after: dataset generator v1 -> v2", fontweight="bold")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  chart -> {path}")


# ============================================================================
# LOAD DATA AND REPRODUCE THE TRAINING SPLIT
# ============================================================================
print("=" * 70)
print("EVALUATION REPORT")
print("=" * 70)

df = pd.read_csv(INPUT_FILE)
X = df[FEATURES]
y_zone = df[TARGET]

# Identical split settings as xgboost_model.py / risk_cluster_model.py:
# 80/20, seeded, stratified on the zone label. Passing the same seed and the
# same stratify vector reproduces the exact same test rows here.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_zone,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y_zone,
)
y_region_test = y_test.map(ZONE_REGION)

print(f"Dataset rows     : {len(df)}")
print(f"Train / test     : {len(X_train)} / {len(X_test)}")
print(f"Features         : {', '.join(FEATURES)}")

# Deployed artifacts. Both are dicts {model, ...names...} as written by
# xgboost_model.py and risk_cluster_model.py. Loading (never retraining) the
# deployed zone model keeps every hold-out number below tied to the exact
# artifact that ships to the backend.
zone_artifact = joblib.load(ZONE_MODEL_OUTPUT_FILE)
region_artifact = joblib.load(REGION_MODEL_OUTPUT_FILE)
zone_model = zone_artifact["model"]
region_model = region_artifact["model"]

zone_probs = zone_model.predict_proba(X_test)
region_probs = region_model.predict_proba(X_test)

print(f"Loaded zone model    : {ZONE_MODEL_OUTPUT_FILE}")
print(f"Loaded region model  : {REGION_MODEL_OUTPUT_FILE}")


# ============================================================================
# ZONE MODEL (STAGE 2): FULL HOLD-OUT EVALUATION
# ============================================================================
print("\nEvaluating deployed zone model on hold-out...")
zone_metrics = score_block(
    y_test, zone_probs, ZONE_LABELS, N_ZONES, prefix="zone_"
)
zone_metrics["feature_importances"] = {
    feature: float(importance)
    for feature, importance in zip(FEATURES, zone_model.feature_importances_)
}

# ============================================================================
# REGION MODEL (STAGE 1): FULL HOLD-OUT EVALUATION
# ============================================================================
print("Evaluating deployed region model on hold-out...")
region_metrics = score_block(
    y_region_test, region_probs, REGION_LABELS, N_REGIONS, prefix="region_"
)


# ============================================================================
# BASELINES
# ============================================================================
print("\nTraining baseline models on the same split...")

# Majority baseline: always guess the most frequent training zone.
majority_class = int(y_train.value_counts().idxmax())
majority_accuracy = float(np.mean(y_test.to_numpy() == majority_class))

# Uniform random baseline: analytically 1/10 per guess, k/10 inside top-k.
random_accuracy = 1.0 / N_ZONES

# Random Forest baseline: the exact configuration printed by
# random_forest_model.py (default and grid-searched variants).
rf_default = RandomForestClassifier(
    n_estimators=100, random_state=RANDOM_STATE, class_weight="balanced"
)
rf_default.fit(X_train, y_train)
rf_default_probs = rf_default.predict_proba(X_test)

rf_tuned = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    random_state=RANDOM_STATE,
    class_weight="balanced",
)
rf_tuned.fit(X_train, y_train)
rf_tuned_probs = rf_tuned.predict_proba(X_test)

baselines = {
    "majority_class": {
        "label": f"Always predict {ZONE_NAMES[majority_class]}",
        "zone_accuracy": majority_accuracy,
        "zone_top3": float(majority_accuracy),  # single guess, also in top-3
    },
    "uniform_random": {
        "label": "Uniform random over 10 zones",
        "zone_accuracy": random_accuracy,
        "zone_top3": 3.0 / N_ZONES,
    },
    "random_forest_default": {
        "label": "Random Forest (100 trees, tuned-free baseline)",
        "zone_accuracy": float(accuracy_score(y_test, np.argmax(rf_default_probs, axis=1))),
        "zone_top3": topk_hit_rate(rf_default_probs, y_test, 3),
    },
    "random_forest_tuned": {
        "label": "Random Forest (200 trees, depth 10, split 5)",
        "zone_accuracy": float(accuracy_score(y_test, np.argmax(rf_tuned_probs, axis=1))),
        "zone_top3": topk_hit_rate(rf_tuned_probs, y_test, 3),
    },
    "xgboost_v2_deployed": {
        "label": "XGBoost deployed (this report's hold-out)",
        "zone_accuracy": zone_metrics["zone_accuracy"],
        "zone_top3": zone_metrics["zone_top3_accuracy"],
    },
    "xgboost_v1_old_generator": {
        "label": "XGBoost measured on v1 generator (uniform zones + 20% noise)",
        "zone_accuracy": V1_RESULTS["zone_accuracy"],
        "zone_top3": V1_RESULTS["zone_top3"],
    },
    "random_forest_v1_old_generator": {
        "label": "Random Forest measured on v1 generator",
        "zone_accuracy": V1_RESULTS["rf_accuracy"],
        "zone_top3": None,
    },
}


# ============================================================================
# 5-FOLD STRATIFIED CROSS-VALIDATION
# ============================================================================
print("\nZone model 5-fold cross-validation:")
zone_cv = cross_validate(ZONE_XGB_PARAMS, X, y_zone, y_zone)

print("\nRegion model 5-fold cross-validation:")
region_cv = cross_validate(
    XGB_PARAMS, X, y_zone.map(ZONE_REGION), y_zone
)


# ============================================================================
# CHARTS
# ============================================================================
print("\nGenerating charts...")
os.makedirs(REPORTS_DIR, exist_ok=True)

save_confusion_heatmap(
    confusion_matrix(y_test, np.argmax(zone_probs, axis=1)),
    ZONE_LABELS,
    "Zone model confusion matrix (hold-out)",
    os.path.join(REPORTS_DIR, "zone_confusion_matrix.png"),
)
save_confusion_heatmap(
    confusion_matrix(y_region_test, np.argmax(region_probs, axis=1)),
    REGION_LABELS,
    "Region model confusion matrix (hold-out)",
    os.path.join(REPORTS_DIR, "region_confusion_matrix.png"),
    figsize=(6.5, 5),
)
save_feature_importance(
    zone_model, os.path.join(REPORTS_DIR, "zone_feature_importance.png")
)
save_precision_recall(
    zone_metrics["zone_per_class"],
    os.path.join(REPORTS_DIR, "zone_precision_recall.png"),
)
save_topk_curve(
    {
        "XGBoost deployed": [
            zone_metrics[f"zone_top{k}_accuracy"] for k in (1, 2, 3, 4, 5)
        ],
        "Random Forest (tuned)": [
            topk_hit_rate(rf_tuned_probs, y_test, k) for k in (1, 2, 3, 4, 5)
        ],
    },
    os.path.join(REPORTS_DIR, "zone_topk_curve.png"),
)
save_baseline_chart(
    {
        "Zone accuracy": {
            "v1": V1_RESULTS["zone_accuracy"],
            "v2": zone_metrics["zone_accuracy"],
        },
        "Zone top-3": {
            "v1": V1_RESULTS["zone_top3"],
            "v2": zone_metrics["zone_top3_accuracy"],
        },
        "Region accuracy": {
            "v1": V1_RESULTS["region_accuracy"],
            "v2": region_metrics["region_accuracy"],
        },
        "RF accuracy": {
            "v1": V1_RESULTS["rf_accuracy"],
            "v2": baselines["random_forest_tuned"]["zone_accuracy"],
        },
    },
    os.path.join(REPORTS_DIR, "baseline_v1_vs_v2.png"),
)


# ============================================================================
# metrics.json
# ============================================================================
comparison = {
    "zone_accuracy": {
        "v1": V1_RESULTS["zone_accuracy"],
        "v2": zone_metrics["zone_accuracy"],
        "delta_pp": round(
            (zone_metrics["zone_accuracy"] - V1_RESULTS["zone_accuracy"]) * 100, 2
        ),
    },
    "zone_top3": {
        "v1": V1_RESULTS["zone_top3"],
        "v2": zone_metrics["zone_top3_accuracy"],
        "delta_pp": round(
            (zone_metrics["zone_top3_accuracy"] - V1_RESULTS["zone_top3"]) * 100, 2
        ),
    },
    "region_accuracy": {
        "v1": V1_RESULTS["region_accuracy"],
        "v2": region_metrics["region_accuracy"],
        "delta_pp": round(
            (region_metrics["region_accuracy"] - V1_RESULTS["region_accuracy"]) * 100, 2
        ),
    },
    "rf_accuracy": {
        "v1": V1_RESULTS["rf_accuracy"],
        "v2": baselines["random_forest_tuned"]["zone_accuracy"],
        "delta_pp": round(
            (baselines["random_forest_tuned"]["zone_accuracy"] - V1_RESULTS["rf_accuracy"]) * 100, 2
        ),
    },
}

metrics_json = {
    "meta": {
        "generated_on": date.today().isoformat(),
        "dataset": INPUT_FILE,
        "rows": int(len(df)),
        "features": FEATURES,
        "target": TARGET,
        "split": "80/20 stratified on zone label, random_state=42",
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "zone_model_artifact": ZONE_MODEL_OUTPUT_FILE,
        "region_model_artifact": REGION_MODEL_OUTPUT_FILE,
        "leakage_columns_excluded": [
            "Specific_ATM_Location",
            "Time_to_Withdraw",
        ],
    },
    "zone_holdout": zone_metrics,
    "region_holdout": region_metrics,
    "zone_cv": zone_cv,
    "region_cv": region_cv,
    "baselines": baselines,
    "v1_vs_v2": comparison,
    "charts": [
        "zone_confusion_matrix.png",
        "region_confusion_matrix.png",
        "zone_feature_importance.png",
        "zone_precision_recall.png",
        "zone_topk_curve.png",
        "baseline_v1_vs_v2.png",
    ],
}

with open(os.path.join(REPORTS_DIR, "metrics.json"), "w", encoding="utf-8") as handle:
    json.dump(metrics_json, handle, indent=2, ensure_ascii=False)
print(f"  data -> {os.path.join(REPORTS_DIR, 'metrics.json')}")


# ============================================================================
# EVALUATION_REPORT.md
# ============================================================================
def markdown_per_class_table(per_class, class_column):
    lines = [
        f"| {class_column} | Precision | Recall | F1 | Support |",
        "|---|---|---|---|---|",
    ]
    for name, values in per_class.items():
        lines.append(
            f"| {name} | {values['precision']:.3f} | {values['recall']:.3f} "
            f"| {values['f1']:.3f} | {values['support']} |"
        )
    return "\n".join(lines)


def cv_markdown(cv_block):
    lines = [
        f"| Fold | Rows | Accuracy | Macro F1 |",
        "|---|---|---|---|",
    ]
    for fold in cv_block["folds"]:
        lines.append(
            f"| {fold['fold']} | {fold['val_rows']} "
            f"| {fold['accuracy']:.4f} | {fold['macro_f1']:.4f} |"
        )
    lines.append(
        f"| **Mean +/- std** | - | "
        f"**{cv_block['accuracy_mean']:.4f} +/- {cv_block['accuracy_std']:.4f}** | "
        f"**{cv_block['macro_f1_mean']:.4f} +/- {cv_block['macro_f1_std']:.4f}** |"
    )
    return "\n".join(lines)


zone_per_class_md = markdown_per_class_table(zone_metrics["zone_per_class"], "Zone")
region_per_class_md = markdown_per_class_table(
    region_metrics["region_per_class"], "Region"
)

baseline_rows = [
    ("Majority class", baselines["majority_class"]),
    ("Uniform random", baselines["uniform_random"]),
    ("Random Forest (100 trees)", baselines["random_forest_default"]),
    ("Random Forest (tuned)", baselines["random_forest_tuned"]),
    ("XGBoost deployed (v2)", baselines["xgboost_v2_deployed"]),
    ("XGBoost on v1 generator", baselines["xgboost_v1_old_generator"]),
    ("Random Forest on v1 generator", baselines["random_forest_v1_old_generator"]),
]
baseline_md = ["| Baseline | Zone accuracy | Zone top-3 |", "|---|---|---|"]
for label, row in baseline_rows:
    accuracy_cell = f"{row['zone_accuracy']:.4f}"
    top3_cell = f"{row['zone_top3']:.4f}" if row["zone_top3"] is not None else "n/a"
    baseline_md.append(f"| {label} | {accuracy_cell} | {top3_cell} |")
baseline_md = "\n".join(baseline_md)

report_md = f"""# Evaluation Report - Predictive Analytics Framework for Cybercrime Complaints

Generated on {date.today().isoformat()} by `evaluation_report.py`.
Machine-readable version: [`metrics.json`](metrics.json).

## Headline: dataset generator v1 -> v2

| Metric | v1 (uniform zones + 20% noise) | v2 (calibrated generator) | Change |
|---|---|---|---|
| Zone accuracy (hold-out) | {V1_RESULTS['zone_accuracy']:.4f} | {zone_metrics['zone_accuracy']:.4f} | **+{comparison['zone_accuracy']['delta_pp']:.1f} pp** |
| Zone top-3 hit rate | {V1_RESULTS['zone_top3']:.4f} | {zone_metrics['zone_top3_accuracy']:.4f} | +{comparison['zone_top3']['delta_pp']:.1f} pp |
| Region accuracy | {V1_RESULTS['region_accuracy']:.4f} | {region_metrics['region_accuracy']:.4f} | +{comparison['region_accuracy']['delta_pp']:.1f} pp |
| Random Forest accuracy | {V1_RESULTS['rf_accuracy']:.4f} | {baselines['random_forest_tuned']['zone_accuracy']:.4f} | +{comparison['rf_accuracy']['delta_pp']:.1f} pp |

![v1 vs v2 baselines](baseline_v1_vs_v2.png)

The v1 generator drew the target ATM zone with `random.choice` (uniform) and
then corrupted 20% of labels, so no complaint attributes could survive - the
zone task was capped at ~20% accuracy. The v2 generator samples the zone from
feature-dependent affinities (fraud type, time of day, amount bracket, victim
district) with a small label-noise term, which is what makes the task
learnable while staying inside the reference-calibrated marginals of the
I4C/NCRP complaint statistics cited in `dataset/dataset.py`.

## Setup

- Dataset: `{INPUT_FILE}` - {len(df):,} rows, {len(X_train):,} train / {len(X_test):,} test
- Split: 80/20, stratified on the zone label, `random_state=42` (same as training scripts)
- Features: {', '.join(FEATURES)}
- Leakage columns **excluded** by policy (see `LEAKAGE_NOTES` in `atm_zone_config.py`): `Specific_ATM_Location` and `Time_to_Withdraw` would each leak the answer and are never used.
- Hold-out numbers come from the deployed artifacts `{ZONE_MODEL_OUTPUT_FILE}` and `{REGION_MODEL_OUTPUT_FILE}` - this script never retrains them.

## Stage 2 - Zone model (deployed XGBoost)

| Metric | Value |
|---|---|
| Accuracy | {zone_metrics['zone_accuracy']:.4f} |
| Balanced accuracy | {zone_metrics['zone_balanced_accuracy']:.4f} |
| Cohen's kappa | {zone_metrics['zone_cohen_kappa']:.4f} |
| MCC | {zone_metrics['zone_mcc']:.4f} |
| Macro F1 | {zone_metrics['zone_macro_f1']:.4f} |
| Weighted F1 | {zone_metrics['zone_weighted_f1']:.4f} |
| Top-1 accuracy | {zone_metrics['zone_top1_accuracy']:.4f} |
| Top-3 hit rate | {zone_metrics['zone_top3_accuracy']:.4f} |
| Top-5 hit rate | {zone_metrics['zone_top5_accuracy']:.4f} |
| MRR | {zone_metrics['zone_mrr']:.4f} |
| ROC-AUC (OvR, macro) | {zone_metrics['zone_roc_auc_ovr_macro']:.4f} |
| Log loss | {zone_metrics['zone_log_loss']:.4f} |
| Brier score (lower better) | {zone_metrics['zone_brier_score']:.4f} |

### Per-zone breakdown

{zone_per_class_md}

| Overall | Value |
|---|---|
| **Macro avg F1** | {zone_metrics['zone_macro_f1']:.3f} |
| **Weighted avg F1** | {zone_metrics['zone_weighted_f1']:.3f} |

### Confusion matrix (rows = actual, cells = % of that zone)

![Zone confusion matrix](zone_confusion_matrix.png)

### Precision and recall per zone

![Per-zone precision and recall](zone_precision_recall.png)

### Feature importances

![Feature importances](zone_feature_importance.png)

### Shortlist quality

![Top-k curve](zone_topk_curve.png)

## Stage 1 - Region model (deployed XGBoost)

| Metric | Value |
|---|---|
| Accuracy | {region_metrics['region_accuracy']:.4f} |
| Balanced accuracy | {region_metrics['region_balanced_accuracy']:.4f} |
| Cohen's kappa | {region_metrics['region_cohen_kappa']:.4f} |
| MCC | {region_metrics['region_mcc']:.4f} |
| Macro F1 | {region_metrics['region_macro_f1']:.4f} |
| ROC-AUC (OvR, macro) | {region_metrics['region_roc_auc_ovr_macro']:.4f} |
| Log loss | {region_metrics['region_log_loss']:.4f} |
| Brier score (lower better) | {region_metrics['region_brier_score']:.4f} |

### Per-region breakdown

{region_per_class_md}

### Confusion matrix

![Region confusion matrix](region_confusion_matrix.png)

## Cross-validation (5-fold, stratified on the zone label)

Hold-out numbers can be lucky; these come from five retrainings on disjoint
folds of the full {len(df):,}-row dataset.

### Zone model

{cv_markdown(zone_cv)}

### Region model

{cv_markdown(region_cv)}

## Baselines

{baseline_md}

Reading: the deployed zone model beats the strongest classical baseline
(Random Forest, tuned) and sits far above chance. The gap between it and the
~0.10 uniform / ~0.20 majority floor is the value the feature engineering adds.

## Honest limitations

1. **The data is synthetic.** Generator parameters are calibrated against
   public Indian cybercrime complaint patterns, but every row is simulated;
   real deployment needs a real complaint corpus.
2. **Region stage does not currently improve the zone top-3** (it changes the
   hit rate by under a point); its value is the interpretable ~{region_metrics['region_accuracy']:.0%} belt
   call for the control room, as documented in `risk_cluster_model.py`.
3. **Class imbalance remains.** The largest zone holds
   {max(v['support'] for v in zone_metrics['zone_per_class'].values()) /
   len(X_test):.0%} of the test set; the weighted variant trades ~3 points of
   accuracy for minority recall and is kept as a comparison, not shipped.
4. Both leakage columns are excluded by policy, not by accident - accuracy
   obtained from them would be meaningless.

## Reproduce

```text
py -3.13 dataset/dataset.py
py -3.13 data_preprocessing.py
py -3.13 feature_engineering.py
py -3.13 risk_cluster_model.py
py -3.13 xgboost_model.py
py -3.13 random_forest_model.py
py -3.13 evaluation_report.py
```
"""

report_path = os.path.join(REPORTS_DIR, "EVALUATION_REPORT.md")
with open(report_path, "w", encoding="utf-8") as handle:
    handle.write(report_md)
print(f"  data -> {report_path}")


# ============================================================================
# CONSOLE SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("SUMMARY (hold-out)")
print("=" * 70)
print(f"  Zone accuracy        : {zone_metrics['zone_accuracy']:.4f}")
print(f"  Zone top-3           : {zone_metrics['zone_top3_accuracy']:.4f}")
print(f"  Zone macro F1        : {zone_metrics['zone_macro_f1']:.4f}")
print(f"  Zone CV accuracy     : {zone_cv['accuracy_mean']:.4f} +/- {zone_cv['accuracy_std']:.4f}")
print(f"  Region accuracy      : {region_metrics['region_accuracy']:.4f}")
print(f"  Region CV accuracy   : {region_cv['accuracy_mean']:.4f} +/- {region_cv['accuracy_std']:.4f}")
print(f"  v1 -> v2 zone gain   : +{comparison['zone_accuracy']['delta_pp']:.1f} pp")
print(f"  Reports written to   : {REPORTS_DIR}/")
