# Predictive Analytics Framework for Cybercrime Complaints

**SKIT-CSE-2023-2027-66**

A machine-learning framework that forecasts the most likely **cash withdrawal
(ATM) zone** for a cybercrime complaint in Jaipur, so police can pre-position
patrols instead of waiting for the money to be withdrawn.

The system answers two questions for every complaint:

| Question | Answered by | Confidence |
|---|---|---|
| **Which belt of the city should we search?** | `risk_cluster_model.py` (Stage 1) | **76.5%** accuracy |
| **Which specific ATM zones inside it?** | `xgboost_model.py` (Stage 2) | **58.5%** top-3 hit rate |

---

## Quick start

```bash
# Python 3.13 is required (the default `python` on some machines has no pandas)
py -3.13 xgboost_model.py        # Stage 2: exact ATM zone, top-3 shortlist
py -3.13 risk_cluster_model.py   # Stage 1 + combined two-stage demo
```

Run from the repository root — both scripts use relative dataset paths.

Full pipeline regeneration:

```bash
py -3.13 dataset/dataset.py           # 1. generate 5000 synthetic complaints
py -3.13 feature_engineering.py       # 2. encode them into numeric features
py -3.13 xgboost_model.py             # 3. train + evaluate the zone model
py -3.13 risk_cluster_model.py        # 4. train + evaluate the region model
```

---

## Repository layout

```
atm_zone_config.py            Shared zone/region mappings, paths, XGBoost params
xgboost_model.py              Stage 2 - ATM zone model + top-3 shortlist
risk_cluster_model.py         Stage 1 - cash-draining region model + two-stage demo
feature_engineering.py        Manual encoding of raw complaints into features
data_preprocessing.py         Raw complaint cleaning
random_forest_model.py        Random Forest baseline (comparison)
generate_report.py            Weekly progress report generation
ARCHITECTURE.md               System design, dataflow and the accuracy analysis

dataset/
  dataset.py                            Synthetic data generator
  jaipur_cybercrime_5000_detailed.csv   Raw 5000 complaints
  jaipur_cybercrime_cleaned.csv         Cleaned complaints
  jaipur_ml_ready_manual_features.csv   ML-ready encoded features  <- model input

backend/                      FastAPI service (complaints CRUD, predictions)
models/                       Exported model artifacts (git-ignored)
```

---

## Work completed to date

### 1. Synthetic dataset generation — `dataset/dataset.py`
Generates 5,000 synthetic cybercrime complaints across 11 Rajasthan districts,
8 fraud types and 10 Jaipur ATM zones, with realistic landmarks per zone
(391 distinct ATM locations) and fraud-type-conditioned criminal patterns.

### 2. Data preprocessing and feature engineering
- `data_preprocessing.py` — raw complaint cleaning
- `feature_engineering.py` — **manual** encoding of four signals into integers:
  fraud type, victim district, time of day (derived from the hour), and amount
  bracket. Produces `jaipur_ml_ready_manual_features.csv`, the single dataset
  every model consumes.

### 3. Random Forest baseline — `random_forest_model.py`
100-tree baseline plus a grid search over `n_estimators`, `max_depth` and
`min_samples_split`. Establishes the ~19% reference point for the zone task.

### 4. XGBoost zone model — `xgboost_model.py`
- 80/20 stratified train/test split, `random_state=42`
- Two variants trained and compared: plain, and inverse-frequency weighted
  (`scale_pos_weight` has no multi-class form, so weights go through `fit()`)
- **20.10%** accuracy, **58.50%** top-3 hit rate
- Exports `models/xgboost_atm_zone.joblib` as `{model, zone_names, features}`
  so the backend never has to hardcode the zone mapping

### 5. Cash-draining region model — `risk_cluster_model.py`
- Groups the 10 zones into 3 geographic belts
- **76.50%** accuracy — the number worth presenting, because it is a genuinely
  confident, actionable statement rather than a near-uniform guess
- `predict_top_3_zones_with_region()` fuses both stages into one call
- Exports `models/xgboost_risk_region.joblib`

### 6. Backend service — `backend/` (FastAPI)
Complaint CRUD endpoints, `Complaint` and `Prediction` tables with
`predicted_zone` / `confidence_score` columns ready for model output. The
exported `.joblib` payloads are shaped to feed those columns directly.

---

## The accuracy problem, and why 20% is the honest answer

The zone model's 20.10% accuracy is **not** an under-trained model. It is the
mathematical ceiling of this dataset.

`dataset/dataset.py:36,41,46` draw the target zone with a **uniform random
choice** from a candidate list, independently of everything else. Fraud type
therefore tells you only *which four zones the answer was drawn from*, never
*which one*. The best possible predictor is:

```
(3/8)(1/4) + (3/8)(1/4) + (2/8)(1/10)  =  21.25%
```

and after the 20% noise injection at `dataset/dataset.py:50-52`, ~19%.

This was verified three ways:

| Measurement | Result |
|---|---|
| Analytic bound from the generator | ~19% |
| Empirical Bayes rate (groupwise majority, 5-fold CV) | 19.48% |
| Random Forest on the same split | 19.3% |
| **This XGBoost model** | **20.10%** |

No model, ensemble or neural network can beat that, because the information is
simply not in the file. This is why the framework is built around a **top-3
shortlist** (58.5% hit rate — roughly three times better than a single guess)
and a **coarser region stage** (76.5%).

### Data leakage: two columns deliberately excluded

The raw dataset contains two columns that would appear to fix accuracy
overnight. **Neither is legitimate.**

| Column | Why it looks good | Why it is excluded |
|---|---|---|
| `Specific_ATM_Location` | All 391 locations map to exactly one zone, so it determines the target with certainty → ~100% accuracy | It is the *outcome* of the crime. Unknowable when the complaint arrives — predicting it is the entire purpose of the system |
| `Time_to_Withdraw` | Mutual information with the zone is 0.22, on par with `Fraud_Type` (0.32) | It is the delay between fraud and withdrawal, known only after the crime |

Only features available at complaint-intake time are used. The reasoning is
recorded in `atm_zone_config.py` so the numbers are never "fixed" the wrong way.

---

## Model comparison

| Model | Task | Accuracy | Top-3 hit rate | Deployed |
|---|---|---|---|---|
| Random Forest | 10 zones | 19.3% | 55.6% | Baseline only |
| XGBoost, weighted | 10 zones | 19.8% | 57.4% | No |
| **XGBoost, plain** | **10 zones** | **20.10%** | **58.50%** | **Yes** |
| XGBoost, weighted | 3 regions | 71.70% | — | No (miscalibrated) |
| **XGBoost, plain** | **3 regions** | **76.50%** | — | **Yes** |

### Why the weighted models are rejected

Class-imbalance weighting is the obvious move against the rare zones, and it
works on the metric it targets — rare-zone recall rose from **0.01 to 0.74** at
the region level, and from 0.03 to 0.29 for Jagatpura at the zone level.

But it buys that recall by flooding the rare class with false positives
(102 and 119 false positives into a 85-row class). The result is a model that
is **confidently wrong on individual complaints** — on one test input it
assigned 70.3% confidence to the wrong region. Since the product is a ranked
shortlist that a human acts on, calibration matters more than macro-F1, so the
plain models ship and the weighted ones are reported as evaluated-and-rejected.

### Does the region stage improve the shortlist?

Honestly: barely.

| Shortlist | Hit rate |
|---|---|
| Zone model alone | 58.50% |
| Zone model + region boost | 59.20% (+0.7) |

The region stage's real value is **interpretability** — a 76.5%-accurate
statement of which belt to search, which a flat 10-way probability list cannot
give a control room.

---

## Known limitations

- **Region 2 ("Central & Outlying Mixed": C-Scheme, Jagatpura) has ~1% recall.**
  These 423 complaints (8.5%) only arise from the generator's fallback branch,
  so they are close to random. The deployed model will rarely flag them; the
  script prints an explicit warning about this. Fixing it properly needs
  features that exist at complaint time, not a model change.
- **Zone-level probabilities are near-uniform** (top zone ~25%). Ranking within
  the shortlist is weakly informative; the region call is the reliable signal.
- **Region 2 is not a tight geographic cluster** — C-Scheme is central and
  Jagatpura is south. It is a residual group and is named as such.
- **The dataset is synthetic**, so these numbers measure the framework, not
  real-world predictive power. Re-validation on actual Jaipur police complaint
  data is required before any operational claim.
- The model scripts are flat top-level scripts, so **importing one re-trains
  it**. The backend should load the `.joblib` artifacts rather than import the
  training modules.

---

## Tech stack

Python 3.13 · pandas 3.0 · scikit-learn 1.9 · XGBoost 3.3 · joblib · FastAPI · SQLAlchemy

Model artifacts total ~14 MB and are git-ignored via `.gitignore`.
