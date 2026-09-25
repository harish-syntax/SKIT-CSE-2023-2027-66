"""
Predictive Analytics Framework for Cybercrime Complaints
=====================================================
Task 1: Train an XGBoost multi-class classifier that forecasts the most likely
        ATM cash-withdrawal zone for a cybercrime complaint.
Task 2: Turn that model into actionable intelligence by returning the TOP 3
        zones (with probabilities) where police should deploy patrols.

Why Top 3 and not Top 1?
    A single guess is a coin flip for a patrol unit. Three ranked zones let the
    control room stage a primary and two fallback checkposts, and the
    "Top-3 hit rate" metric below measures how often the real zone is inside
    that shortlist. On this dataset the top-3 shortlist is roughly three times
    more reliable than a single guess, which is the whole justification for
    ranking rather than returning one label.

A note on the accuracy figure
    Overall accuracy lands near 20% for 10 classes. Read that carefully: the
    measured ceiling for this dataset is ~19.5% (see ARCHITECTURE.md), and a
    Random Forest reaches ~19.3% on the same split. The synthetic generator
    picks the ATM zone with a UNIFORM random draw from a candidate list, so
    fraud type tells you only which four zones the answer came from, never
    which one. ~20% is therefore the information-theoretic ceiling, not an
    under-trained model. Judge the approach on the Top-3 hit rate (~59%).

    If you need a confidently accurate statement rather than a shortlist, see
    risk_cluster_model.py, which predicts the broader cash-draining region at
    ~76% accuracy.

Run it from the repository root:
    py -3.13 xgboost_model.py
"""

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# Shared configuration. Every zone/region mapping lives in one place so this
# script, risk_cluster_model.py and the backend cannot drift apart.
from atm_zone_config import (
    FEATURES,
    INPUT_FILE,
    RANDOM_STATE,
    TARGET,
    XGB_PARAMS,
    ZONE_LABELS,
    ZONE_MODEL_OUTPUT_FILE as MODEL_OUTPUT_FILE,
    ZONE_NAMES,
    as_feature_row,
    compute_sample_weights,
)

# ============================================================================
# ZONE-SPECIFIC XGBOOST SETTINGS
# ============================================================================
# Built on the shared base parameters. The objective and class count are stated
# explicitly here because predict_proba() must return a score per zone for the
# top-3 shortlist to work; the region model in risk_cluster_model.py lets
# XGBoost infer its own.
ZONE_XGB_PARAMS = {
    **XGB_PARAMS,
    "objective": "multi:softprob",
    "num_class": 10,
    "eval_metric": "mlogloss",
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def evaluate(model, X_test, y_test, title):
    """
    Print a full evaluation block for one trained model: accuracy, classification
    report, confusion matrix, feature importances and Top-3 hit rate.
    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    probabilities = model.predict_proba(X_test)

    # Top-3 hit rate: how often the correct zone is anywhere in the model's
    # three highest-probability suggestions (rather than only the top 1).
    top_3_codes = np.argsort(-probabilities, axis=1)[:, :3]
    true_codes = y_test.to_numpy()
    hit_rate = np.mean(
        [true_code in row for true_code, row in zip(true_codes, top_3_codes)]
    )

    print(f"\n{'=' * 70}")
    print(title)
    print("=" * 70)
    print(f"Overall Accuracy : {accuracy:.4f}  ({accuracy * 100:.2f}%)")
    print(f"Top-3 Hit Rate   : {hit_rate:.4f}  ({hit_rate * 100:.2f}%)")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=ZONE_LABELS,
            zero_division=0,
        )
    )

    print("Confusion Matrix (rows = actual zone, columns = predicted zone):")
    print(ZONE_LABELS)
    print(confusion_matrix(y_test, y_pred))

    print("Feature Importances (how much each signal drove the decisions):")
    importances = sorted(
        zip(FEATURES, model.feature_importances_), key=lambda pair: pair[1], reverse=True
    )
    for feature, importance in importances:
        bar = "#" * int(round(importance * 40))
        print(f"  {feature:<24} {importance:.4f} {bar}")

    return {"accuracy": accuracy, "top3_hit_rate": hit_rate}


def predict_top_3_zones(model, input_features):
    """
    Actionable intelligence: return the top 3 most likely ATM zones for one
    complaint, ranked by probability and mapped to readable zone names.

    Parameters
    ----------
    model : trained XGBClassifier
    input_features : dict | list | pd.Series
        The four engineered feature values for a single complaint, for example
        {"Fraud_Type_Code": 0, "Victim_District_Code": 0,
         "Time_of_Day_Code": 3, "Amount_Bracket_Code": 1}

    Returns
    -------
    list of dict: [{"code": int, "zone": str, "probability": float}, ...]
    """
    # Normalise whatever the caller passed into a one-row DataFrame whose columns
    # are in exactly the same order used during training. Without this step a
    # dict or list could silently scramble the feature order and corrupt the
    # prediction.
    row = as_feature_row(input_features)

    # One probability per class, in the order 0..9
    probabilities = model.predict_proba(row)[0]

    # Sort descending and keep the best three
    top_3_indices = np.argsort(probabilities)[::-1][:3]

    results = [
        {
            "code": int(class_code),
            "zone": ZONE_NAMES[int(class_code)],
            "probability": float(probabilities[class_code]),
        }
        for class_code in top_3_indices
    ]

    # Human readable output for the live demo
    print("Top 3 predicted ATM zones (patrol deployment order):")
    for rank, result in enumerate(results, start=1):
        print(
            f"{rank}. {result['zone']} - {result['probability'] * 100:.0f}%"
        )

    return results


# ============================================================================
# TASK 1: LOAD DATA, SPLIT, AND TRAIN
# ============================================================================

print("Predictive Analytics Framework for Cybercrime Complaints")
print("=" * 70)

# Load the manually engineered dataset produced by feature_engineering.py
df = pd.read_csv(INPUT_FILE)

X = df[FEATURES]
y = df[TARGET]

print(f"Dataset          : {INPUT_FILE}")
print(f"Total complaints : {len(df)}")
print(f"Features used    : {len(FEATURES)} -> {', '.join(FEATURES)}")
print(f"Target           : {TARGET} ({len(ZONE_NAMES)} ATM zones)")

# 80/20 hold-out split. stratify=y keeps the same class proportions in both
# halves, which matters because C-Scheme and Jagatpura are already the rarest.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y,
)

print(f"Training samples : {len(X_train)}")
print(f"Testing samples  : {len(X_test)}")

# --- Model A: plain baseline -------------------------------------------------
# Straightforward training, no adjustment for the class imbalance.
print("\nTraining Model A (plain baseline)...")
model_plain = XGBClassifier(**ZONE_XGB_PARAMS)
model_plain.fit(X_train, y_train)

# --- Model B: imbalance weighted -------------------------------------------
# The dataset is skewed (C-Scheme ~235 rows vs ~580 for most zones), so this
# variant up-weights rare zones to give them a fairer chance.
print("Training Model B (inverse-frequency weighted)...")
model_weighted = XGBClassifier(**ZONE_XGB_PARAMS)
model_weighted.fit(X_train, y_train, sample_weight=compute_sample_weights(y_train))

# Score both models on the untouched test set
results_plain = evaluate(model_plain, X_test, y_test, "MODEL A - PLAIN BASELINE")
results_weighted = evaluate(
    model_weighted, X_test, y_test, "MODEL B - IMBALANCE WEIGHTED"
)

# --- Pick the winner ---------------------------------------------------------
# Top-3 hit rate is the primary decision metric (it is what patrol deployment
# actually relies on), with overall accuracy as the tie-breaker.
print(f"\n{'=' * 70}")
print("SIDE-BY-SIDE COMPARISON")
print("=" * 70)
print(
    f"{'Model':<14}{'Accuracy':<14}{'Top-3 Hit Rate':<18}"
)
print("-" * 70)
print(
    f"{'A - Plain':<14}{results_plain['accuracy']:<14.4f}"
    f"{results_plain['top3_hit_rate']:<18.4f}"
)
print(
    f"{'B - Weighted':<14}{results_weighted['accuracy']:<14.4f}"
    f"{results_weighted['top3_hit_rate']:<18.4f}"
)

if (
    results_weighted["top3_hit_rate"],
    results_weighted["accuracy"],
) > (results_plain["top3_hit_rate"], results_plain["accuracy"]):
    best_model = model_weighted
    best_model_name = "Model B (imbalance weighted)"
else:
    best_model = model_plain
    best_model_name = "Model A (plain baseline)"

print(f"\nSelected for deployment: {best_model_name}")


# ============================================================================
# EXPORT THE TRAINED MODEL FOR THE BACKEND API
# ============================================================================
# Saved as a dict rather than a bare estimator so the FastAPI service receives
# the code->name mapping and the feature order alongside the model. That keeps
# backend/app in sync with this file instead of duplicating the mapping.
os.makedirs(os.path.dirname(MODEL_OUTPUT_FILE), exist_ok=True)

joblib.dump(
    {
        "model": best_model,
        "zone_names": ZONE_NAMES,
        "features": FEATURES,
    },
    MODEL_OUTPUT_FILE,
)

print(f"Trained model saved to: {MODEL_OUTPUT_FILE}")


# ============================================================================
# TASK 2: DEMO OF THE TOP 3 PREDICTION FUNCTION
# ============================================================================
# Mock complaint, spoken in plain language:
#   "A Jaipur resident was targeted by a UPI scam and lost Rs 18,000,
#    reporting it at 14:00 (Afternoon)."
#
# Encoded by hand using the dictionaries in feature_engineering.py:
#   Fraud_Type_Code       0 = UPI Scam
#   Victim_District_Code  0 = Jaipur
#   Time_of_Day_Code      1 = Afternoon  (complaint logged at 14:00)
#   Amount_Bracket_Code   1 = Medium (15k-50k), because 18,000 falls in that band
#
# IMPORTANT: this mock must stay internally consistent with the way the dataset
# was generated. dataset/dataset.py treats a UPI Scam as a low-value, daytime
# crime (Rs 2,000-30,000, between 09:00 and 20:00). An earlier version of this
# demo used "UPI Scam at 22:00 for Rs 32,000", which contradicts that on both
# counts and handed the model an impossible complaint -- it still returned
# confident-looking output, which is the dangerous kind of wrong. Keep the
# amount and time-of-day plausible for the fraud type.
mock_complaint = {
    "Fraud_Type_Code": 0,
    "Victim_District_Code": 0,
    "Time_of_Day_Code": 1,
    "Amount_Bracket_Code": 1,
}

print(f"\n{'=' * 70}")
print("TASK 2 - TOP 3 ZONE PREDICTION (LIVE DEMO)")
print("=" * 70)
print("Mock complaint: 'UPI Scam, victim in Jaipur, reported at 14:00, Rs 18,000 lost'")
print(f"Engineered features: {mock_complaint}\n")

top_zones = predict_top_3_zones(best_model, mock_complaint)

# The function also returns structured data, ready to be posted straight into
# the backend's Prediction table (predicted_zone + confidence_score).
print("\nStructured output for the API:")
for result in top_zones:
    print(f"  {result['zone']}: {result['probability']:.4f}")

print(
    "\nNote: these zone-level probabilities are near-uniform because the model "
    "has little\nsignal to separate the 10 zones. For a confident, actionable "
    "statement, run\nrisk_cluster_model.py, which predicts the broader "
    "cash-draining region at ~76%."
)
