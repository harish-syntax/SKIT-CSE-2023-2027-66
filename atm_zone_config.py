"""
Shared configuration for the Predictive Analytics Framework for Cybercrime
Complaints.

Every mapping the ML layer depends on lives here exactly once, so the training
scripts, the exported model artifacts and the FastAPI backend cannot drift
apart. If a zone code or a region boundary changes, it changes here.

Imported by:
    - xgboost_model.py       (stage 2: exact ATM zone, top-3 shortlist)
    - risk_cluster_model.py  (stage 1: cash-draining region)
"""

import pandas as pd

# ============================================================================
# PATHS AND SHARED SETTINGS
# ============================================================================

INPUT_FILE = "dataset/jaipur_ml_ready_manual_features.csv"
ZONE_MODEL_OUTPUT_FILE = "models/xgboost_atm_zone.joblib"
REGION_MODEL_OUTPUT_FILE = "models/xgboost_risk_region.joblib"

# Fixed seed so every run reproduces the same numbers, which matters when you
# are quoting accuracy figures on stage.
RANDOM_STATE = 42

# The four engineered features (X) fed to every model in this project
FEATURES = [
    "Fraud_Type_Code",
    "Victim_District_Code",
    "Time_of_Day_Code",
    "Amount_Bracket_Code",
]

# The label (y): which of the 10 Jaipur ATM zones the money was withdrawn from
TARGET = "Target_ATM_Zone_Code"


# ============================================================================
# ATM ZONES (stage 2 target classes, 0-9)
# ============================================================================
# Numeric class code -> human readable zone name. This exact mapping is
# produced by feature_engineering.py; reversing it is what turns a raw model
# output into a name a police officer can act on.
ZONE_NAMES = {
    0: "Mansarovar",
    1: "Sitapura Industrial",
    2: "Vaishali Nagar",
    3: "Malviya Nagar",
    4: "Jhotwara",
    5: "C-Scheme",
    6: "Pratap Nagar",
    7: "Bani Park",
    8: "Jagatpura",
    9: "Vidyadhar Nagar",
}

# Same names ordered by class code, for readable scikit-learn reports
ZONE_LABELS = [ZONE_NAMES[code] for code in range(len(ZONE_NAMES))]


# ============================================================================
# CASH-DRAINING REGIONS (stage 1, groups the 10 zones into 3 regions)
# ============================================================================
# Why group them at all?
# The zone-level task has a hard ceiling of ~20% accuracy (see ARCHITECTURE.md
# for the measurement). The region-level task is genuinely learnable because
# the data generator's fraud-type logic separates the zones into broad
# geographic belts. Predicting the belt first gives the control room a
# confident, actionable statement, then the zone model ranks streets inside it.
#
# Region 0 - West / North-West residential belt
# Region 1 - South / East industrial and highway corridor
# Region 2 - Central and outlying mixed group. This one is deliberately honest
#            about itself: C-Scheme is central and Jagatpura is south, so it is
#            a residual group rather than a tight geographic cluster.
REGION_NAMES = {
    0: "West & NW Residential",
    1: "South & East Industrial",
    2: "Central & Outlying Mixed",
}

# Region code -> the ATM zone codes that belong to it
REGION_ZONES = {
    0: [0, 2, 3, 7],  # Mansarovar, Vaishali Nagar, Malviya Nagar, Bani Park
    1: [1, 4, 6, 9],  # Sitapura Industrial, Jhotwara, Pratap Nagar, Vidyadhar Nagar
    2: [5, 8],        # C-Scheme, Jagatpura
}

# Region code -> readable name, ordered for reports
REGION_LABELS = [REGION_NAMES[code] for code in range(len(REGION_NAMES))]

# Flat lookup: ATM zone code -> its region code. Used to re-weight a zone's
# probability by the probability of the region it sits in.
ZONE_REGION = {
    zone_code: region_code
    for region_code, zone_codes in REGION_ZONES.items()
    for zone_code in zone_codes
}

# Region boundaries as readable text, used when printing the model summary
REGION_DESCRIPTION = "\n".join(
    f"  {region_code} {REGION_NAMES[region_code]:<26}: "
    f"{', '.join(ZONE_NAMES[zone] for zone in zone_codes)}"
    for region_code, zone_codes in REGION_ZONES.items()
)


# ============================================================================
# XGBOOST HYPERPARAMETERS
# ============================================================================
# Shared by the zone model and the region model so the two stages are directly
# comparable and either can be retrained without re-deriving settings.
#
# Note on the settings: the signal in this dataset is thin and spread across
# only 4 coarse features, so the models want to be heavily regularised. Many
# small, shallow trees (depth 3, slow learning rate) generalise noticeably
# better here than a few deep ones -- a depth-8 / 300-tree variant was tested
# and lost roughly 3 accuracy points.
XGB_PARAMS = {
    "n_estimators": 1000,
    "learning_rate": 0.05,
    "max_depth": 3,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,  # use all CPU cores for a fast demo
}


# ============================================================================
# DATA LEAKAGE GUARD
# ============================================================================
# The raw detailed dataset (dataset/jaipur_cybercrime_5000_detailed.csv) also
# contains two columns that would appear to solve this problem spectacularly.
# Neither is legitimate, and this comment exists so nobody "fixes" the accuracy
# number with them later.
#
# 1. Specific_ATM_Location
#    All 391 distinct ATM locations in the dataset map to exactly ONE zone
#    each, so it determines the target perfectly. Training on it yields ~100%
#    accuracy. It is the *outcome* of the crime: it is unknowable at the moment
#    the complaint arrives, and the whole purpose of the system is to predict
#    it. Using it is textbook target leakage.
#
# 2. Time_to_Withdraw
#    Mutual information with the zone is 0.22, on par with Fraud_Type (0.32).
#    It cleanly separates the two large geographic belts, which makes it look
#    predictive. But it is the delay between the fraud and the withdrawal, so
#    it too is only known after the crime. Also leakage.
#
# The rule: only features available at complaint-intake time may be used. Every
# column in FEATURES passes that test; these two do not.
LEAKAGE_NOTES = [
    "Specific_ATM_Location - determines the zone with certainty, but is the "
    "crime's outcome, not a complaint attribute. EXCLUDED.",
    "Time_to_Withdraw - predictive of the region (MI 0.22) but only known "
    "after the withdrawal occurs. EXCLUDED.",
]


# ============================================================================
# SHARED HELPER
# ============================================================================
def compute_sample_weights(y_train):
    """
    Build one weight per training row for class-imbalance training.

    Rare classes are up-weighted and common classes down-weighted, so the model
    stops ignoring minority groups just because there are fewer of them in the
    file.

    weight = (total rows) / (number of classes * rows in that class)

    Caveat learned the hard way: on the region task this weighting raised
    macro-F1 but wrecked probability calibration, making the model confidently
    wrong on some inputs. Prefer it only where minority *recall* is the goal.
    """
    counts = y_train.value_counts()
    total_rows = len(y_train)
    total_classes = counts.size

    return y_train.map(
        lambda code: total_rows / (total_classes * counts[code])
    ).to_numpy()


def as_feature_row(input_features):
    """
    Normalise a single complaint into a one-row DataFrame.

    Accepts a dict, a list/tuple, or a pd.Series, and always returns columns in
    exactly the order given by FEATURES. Without this step a caller could pass a
    dict or list whose feature order differed from training, and the model would
    return a confident, completely wrong prediction.

    Raises a clear ValueError naming the missing features rather than silently
    feeding NaNs into the model.
    """
    if isinstance(input_features, pd.Series):
        row = input_features.to_frame().transpose()
    elif isinstance(input_features, dict):
        row = pd.DataFrame([input_features])
    else:
        row = pd.DataFrame([list(input_features)], columns=FEATURES)

    row = row.reindex(columns=FEATURES)

    if row.isna().any().any():
        missing = [column for column in FEATURES if row[column].isna().all()]
        raise ValueError(
            f"Missing feature value(s) for: {', '.join(missing)}. "
            f"All of {FEATURES} must be supplied."
        )

    return row
