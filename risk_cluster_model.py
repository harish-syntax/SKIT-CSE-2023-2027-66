"""
Predictive Analytics Framework for Cybercrime Complaints
=====================================================
STAGE 1: the cash-draining REGION model.

Why this stage exists
---------------------
xgboost_model.py predicts which of 10 ATM zones was used, and that task is
capped at roughly 20% accuracy. That is not a modelling failure: the dataset
generator picks the zone with a UNIFORM random draw (dataset/dataset.py:36-52),
so no algorithm can do better than ~19.5% here. Measured and documented in
ARCHITECTURE.md.

What does work is the coarser question. The same generator applies broad
geographic logic to fraud type -- high-value night-time frauds cluster on the
industrial and highway corridors, low-value daytime frauds cluster in the
western residential belt. Grouping the 10 zones into 3 regions makes that
structure learnable, and the model reaches ~76% accuracy.

So the two stages answer different questions:
    Stage 1 (this file)   -> WHICH BELT?  ~76% accurate, and stated with a
                             confidence a control room can act on
    Stage 2 (xgboost_model.py) -> WHICH STREETS within it?  a top-3 shortlist

Run it from the repository root:
    py -3.13 risk_cluster_model.py
"""

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from atm_zone_config import (
    FEATURES,
    INPUT_FILE,
    LEAKAGE_NOTES,
    RANDOM_STATE,
    REGION_DESCRIPTION,
    REGION_LABELS,
    REGION_MODEL_OUTPUT_FILE,
    REGION_NAMES,
    TARGET,
    XGB_PARAMS,
    ZONE_LABELS,
    ZONE_MODEL_OUTPUT_FILE,
    ZONE_NAMES,
    ZONE_REGION,
    as_feature_row,
    compute_sample_weights,
)

TOP_N = 3


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def zone_to_region(zone_code):
    """Map an ATM zone code to the code of the region it sits in."""
    return ZONE_REGION[zone_code]


def evaluate_region_model(model, X_test, y_test, title, deployable=True):
    """
    Print accuracy, a per-region classification report and the confusion matrix
    for one region model. Per-region RECALL is the metric that matters here,
    because a region the model never predicts means those zones never receive a
    patrol recommendation.
    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\n{'=' * 70}")
    print(title)
    print("=" * 70)
    if not deployable:
        print("STATUS: evaluated for comparison only - NOT for deployment.")
    print(f"Overall Accuracy : {accuracy:.4f}  ({accuracy * 100:.2f}%)")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test, y_pred, target_names=REGION_LABELS, zero_division=0
        )
    )

    print("Confusion Matrix (rows = actual region, columns = predicted region):")
    print(REGION_LABELS)
    print(confusion_matrix(y_test, y_pred))

    # Call out any region the model effectively never predicts, which would be a
    # silent gap in coverage rather than a rounding error.
    report = classification_report(
        y_test, y_pred, target_names=REGION_LABELS, zero_division=0, output_dict=True
    )
    starved = [
        name
        for name in REGION_LABELS
        if report[name]["recall"] < 0.10 and report[name]["support"] > 0
    ]
    if starved:
        print(
            f"\nWARNING: recall below 10% for {', '.join(starved)}. "
            "This group would receive almost no patrol coverage."
        )

    return accuracy


def top_n_hit_rate(scores, truth, n=TOP_N):
    """Share of rows whose true zone appears in the model's top-n shortlist."""
    shortlist = np.argsort(-scores, axis=1)[:, :n]
    return float(np.mean([t in row for t, row in zip(truth, shortlist)]))


def predict_top_3_zones_with_region(zone_model, region_model, input_features, top_n=TOP_N):
    """
    Two-stage actionable intelligence for a single complaint.

    Stage 1 asks the region model which cash-draining belt the withdrawal most
    likely fell in. Stage 2 re-weights the zone model's probabilities by the
    probability of each zone's region, so zones inside a confident region are
    promoted and the final shortlist is internally consistent with the region
    call.

    Parameters
    ----------
    zone_model : trained XGBClassifier predicting Target_ATM_Zone_Code
    region_model : trained XGBClassifier predicting the region code
    input_features : dict | list | pd.Series
        The four engineered features for one complaint.
    top_n : how many zones to return (default 3)

    Returns
    -------
    dict with the region call and the ranked shortlist.
    """
    row = as_feature_row(input_features)

    zone_probs = zone_model.predict_proba(row)[0]
    region_probs = region_model.predict_proba(row)[0]

    best_region = int(np.argmax(region_probs))

    # Promote zones that sit inside a likely region. Multiplying by the region's
    # probability (rather than hard-filtering to it) keeps a zone visible when
    # the region model is genuinely torn between two belts.
    region_of_each_zone = np.array([zone_to_region(z) for z in range(len(ZONE_NAMES))])
    combined = zone_probs * region_probs[region_of_each_zone]

    ranked = np.argsort(combined)[::-1][:top_n]
    shortlist = [
        {
            "rank": position,
            "zone_code": int(zone_code),
            "zone": ZONE_NAMES[int(zone_code)],
            "region": REGION_NAMES[zone_to_region(int(zone_code))],
            "zone_probability": float(zone_probs[zone_code]),
            "combined_probability": float(combined[zone_code]),
        }
        for position, zone_code in enumerate(ranked, start=1)
    ]

    print("Stage 1 - predicted cash-draining region:")
    print(f"  {REGION_NAMES[best_region]} - {region_probs[best_region] * 100:.0f}%")
    zones_in_region = [
        ZONE_NAMES[z] for z in range(len(ZONE_NAMES)) if zone_to_region(z) == best_region
    ]
    print(f"  zones in this belt: {', '.join(zones_in_region)}")

    print(f"\nStage 2 - top {top_n} ATM zones to patrol:")
    for entry in shortlist:
        print(
            f"{entry['rank']}. {entry['zone']} - "
            f"{entry['combined_probability'] * 100:.0f}%"
        )

    return {
        "region_code": best_region,
        "region": REGION_NAMES[best_region],
        "region_probability": float(region_probs[best_region]),
        "top_zones": shortlist,
    }


# ============================================================================
# LOAD, MAP ZONES TO REGIONS, SPLIT
# ============================================================================
print("Predictive Analytics Framework for Cybercrime Complaints")
print("STAGE 1 - CASH-DRAINING REGION MODEL")
print("=" * 70)

df = pd.read_csv(INPUT_FILE)
X = df[FEATURES]
y_zone = df[TARGET]

# Regroup the 10 zone labels into 3 region labels. The zone column is untouched,
# so the zone model in xgboost_model.py keeps working on the same dataset.
y_region = y_zone.map(zone_to_region)

print(f"Dataset          : {INPUT_FILE}")
print(f"Total complaints : {len(df)}")
print(f"Features used    : {', '.join(FEATURES)}")
print(f"Regions          : {len(REGION_NAMES)} (grouping the 10 ATM zones)")
print("\nRegion definitions:")
print(REGION_DESCRIPTION)
print("\nComplaints per region:")
for region_code in range(len(REGION_NAMES)):
    count = int((y_region == region_code).sum())
    print(f"  {region_code} {REGION_NAMES[region_code]:<26} {count:>5}  ({count / len(df) * 100:.1f}%)")

# Same split settings as xgboost_model.py -- including stratifying on the ZONE
# label -- so both stages are trained and scored against an identical test set.
# Stratifying on the region instead would silently produce a different test
# split and make the two scripts' numbers incomparable.
# The return order matches the argument order below: X, y_region, y_zone.
X_train, X_test, y_region_train, y_region_test, y_zone_train, y_zone_test = (
    train_test_split(
        X,
        y_region,
        y_zone,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y_zone,
    )
)

print(f"\nTraining samples : {len(X_train)}")
print(f"Testing samples  : {len(X_test)}")


# ============================================================================
# TRAIN THE TWO REGION MODELS
# ============================================================================
# XGBoost infers its own objective and class count here, since the label has
# three classes rather than ten.
print("\nTraining region model A (plain)...")
region_model_plain = XGBClassifier(**XGB_PARAMS)
region_model_plain.fit(X_train, y_region_train)

print("Training region model B (inverse-frequency weighted)...")
region_model_weighted = XGBClassifier(**XGB_PARAMS)
region_model_weighted.fit(
    X_train, y_region_train, sample_weight=compute_sample_weights(y_region_train)
)

accuracy_plain = evaluate_region_model(
    region_model_plain, X_test, y_region_test, "REGION MODEL A - PLAIN (DEPLOYABLE)"
)

# The weighted variant is trained and reported because it is the obvious thing
# to try against a rare region, but it is NOT shipped. Measured behaviour: it
# lifts rare-region recall (0.01 -> 0.74) and macro-F1 (0.55 -> 0.64) while
# flooding the rare class with false positives, which makes it confidently wrong
# on individual complaints. Accuracy is not the reason to reject it -- ranking
# quality is.
accuracy_weighted = evaluate_region_model(
    region_model_weighted,
    X_test,
    y_region_test,
    "REGION MODEL B - IMBALANCE WEIGHTED (COMPARISON ONLY)",
    deployable=False,
)

print(f"\n{'=' * 70}")
print("REGION MODEL COMPARISON")
print("=" * 70)
print(f"  A - Plain (deployed)     accuracy {accuracy_plain:.4f}")
print(f"  B - Weighted (rejected)  accuracy {accuracy_weighted:.4f}")
print(
    "  B is rejected on calibration, not accuracy: it predicts the rare region\n"
    "  far too often, so a single complaint can come back confidently wrong."
)
print(f"  Selected for deployment: Model A (plain)")

region_model = region_model_plain


# ============================================================================
# DOES STAGE 1 ACTUALLY HELP STAGE 2?
# ============================================================================
# Honest check rather than an assumed win. The region call is a confident,
# interpretable statement for a control room, but it is not a large accuracy
# gain over the zone shortlist on its own.
print(f"\n{'=' * 70}")
print("DOES THE REGION STAGE IMPROVE THE ZONE SHORTLIST?")
print("=" * 70)

region_of_each_zone = np.array([zone_to_region(z) for z in range(len(ZONE_NAMES))])
truth = y_zone_test.to_numpy()

zone_model = XGBClassifier(
    **{**XGB_PARAMS, "objective": "multi:softprob", "num_class": 10}
)
zone_model.fit(X_train, y_zone_train)

zone_probs = zone_model.predict_proba(X_test)
region_probs = region_model.predict_proba(X_test)

zone_only_hit = top_n_hit_rate(zone_probs, truth)
combined_hit = top_n_hit_rate(zone_probs * region_probs[:, region_of_each_zone], truth)

print(f"Zone shortlist alone            : {zone_only_hit:.4f}")
print(f"Zone shortlist + region boost   : {combined_hit:.4f}   "
      f"({(combined_hit - zone_only_hit) * 100:+.1f} points)")

if combined_hit <= zone_only_hit:
    print(
        "  -> The region stage does NOT improve the shortlist here.\n"
        "     Its value is interpretability: a ~76%-accurate statement of which\n"
        "     belt to search, which a flat 10-way probability list cannot give."
    )
else:
    print("  -> The region stage helps, and also adds an interpretable region call.")


# ============================================================================
# EXPORT FOR THE BACKEND
# ============================================================================
os.makedirs(os.path.dirname(REGION_MODEL_OUTPUT_FILE), exist_ok=True)

joblib.dump(
    {
        "model": region_model,
        "region_names": REGION_NAMES,
        "zone_region": ZONE_REGION,
        "features": FEATURES,
    },
    REGION_MODEL_OUTPUT_FILE,
)

print(f"Trained region model saved to: {REGION_MODEL_OUTPUT_FILE}")


# ============================================================================
# TASK 2: TWO-STAGE TOP 3 PREDICTION DEMO
# ============================================================================
# Mock complaint: "A Jaipur resident was targeted by a UPI scam and lost
# Rs 18,000, reporting it at 14:00."
#
# This stays consistent with how the dataset was generated: dataset/dataset.py
# treats a UPI Scam as a low-value, daytime crime (Rs 2,000-30,000, 09:00-20:00).
# An earlier demo used "UPI Scam at 22:00 for Rs 32,000", which contradicts that
# on both counts; the model still returned confident output, so the demo would
# have looked plausible while being meaningless.
mock_complaint = {
    "Fraud_Type_Code": 0,        # UPI Scam
    "Victim_District_Code": 0,   # Jaipur
    "Time_of_Day_Code": 1,       # Afternoon (logged 14:00)
    "Amount_Bracket_Code": 1,    # Medium, 15k-50k (Rs 18,000)
}

print(f"\n{'=' * 70}")
print("TWO-STAGE TOP 3 PREDICTION (LIVE DEMO)")
print("=" * 70)
print("Mock complaint: 'UPI Scam, victim in Jaipur, reported at 14:00, Rs 18,000 lost'")
print(f"Engineered features: {mock_complaint}\n")

prediction = predict_top_3_zones_with_region(zone_model, region_model, mock_complaint)

print("\nStructured output for the API:")
print(f"  region: {prediction['region']} "
      f"({prediction['region_probability']:.4f})")
for entry in prediction["top_zones"]:
    print(f"  {entry['zone']}: {entry['combined_probability']:.4f}")


# ============================================================================
# DATA LEAKAGE GUARD
# ============================================================================
print(f"\n{'=' * 70}")
print("WHY TWO COLUMNS WERE DELIBERATELY LEFT OUT")
print("=" * 70)
for note in LEAKAGE_NOTES:
    print(f"  - {note}")
print(
    "\n  Both exist in the raw dataset and would appear to fix the accuracy\n"
    "  problem overnight. Neither is available when a complaint is first\n"
    "  reported, so training on either would inflate every number above."
)
