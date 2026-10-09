# Predictive Analytics Framework for Cybercrime Complaints

**SKIT-CSE-2023-2027-66**

A machine-learning framework that forecasts the most likely **cash withdrawal
(ATM) zone** for a cybercrime complaint in Jaipur, so police can pre-position
patrols instead of waiting for the money to be withdrawn.


---

## Quick start



---

## Repository layout

```
atm_zone_config.py            Shared zone/region mappings, paths, XGBoost params
xgboost_model.py              Stage 2 - ATM zone model + top-3 shortlist
risk_cluster_model.py         Stage 1 - cash-draining region model + two-stage demo
feature_engineering.py        Manual encoding of raw complaints into features
data_preprocessing.py         Raw complaint cleaning
random_forest_model.py        Random Forest baseline (comparison)
evaluation_report.py          Extended evaluation -> reports/ (metrics, charts, CV)
generate_report.py            Weekly progress report generation
ARCHITECTURE.md               System design, dataflow and the accuracy analysis

dataset/
  dataset.py                            Synthetic data generator (v2 calibrated)
  jaipur_cybercrime_5000_detailed.csv   Raw 5000 complaints
  jaipur_cybercrime_cleaned.csv         Cleaned complaints
  jaipur_ml_ready_manual_features.csv   ML-ready encoded features  <- model input

reports/                      Evaluation report: metrics.json, charts, markdown
backend/                      FastAPI service (complaints CRUD, predictions)
models/                       Exported model artifacts (git-ignored)
```

---

## Work completed to date

### 1. Synthetic dataset generation — `dataset/dataset.py` (v2 calibrated)
Generates 5,000 synthetic cybercrime complaints across 11 Rajasthan districts,
8 fraud types and 10 Jaipur ATM zones, with realistic landmarks per zone
(391 distinct ATM locations) and fraud-type-conditioned criminal patterns.
The generator is **seeded** (`random.seed(42)`), its marginals are calibrated
to published Indian cybercrime complaint patterns (NCRP/I4C composition,
right-skewed loss amounts, business-hours complaint timing), and the target
zone is sampled from **feature-dependent affinities** rather than a uniform
draw — see the accuracy story below.

### 2. Data preprocessing and feature engineering
- `data_preprocessing.py` — raw complaint cleaning
- `feature_engineering.py` — **manual** encoding of four signals into integers:
  fraud type, victim district, time of day (derived from the hour), and amount
  bracket. Produces `jaipur_ml_ready_manual_features.csv`, the single dataset
  every model consumes.

### 3. Random Forest baseline — `random_forest_model.py`
100-tree baseline plus a grid search over `n_estimators`, `max_depth` and
`min_samples_split`. Establishes the **71.2%** reference point for the zone
task (v1 generator: 19.3%).

### 4. XGBoost zone model — `xgboost_model.py`
- 80/20 stratified train/test split, `random_state=42`
- Two variants trained and compared: plain, and inverse-frequency weighted
  (`scale_pos_weight` has no multi-class form, so weights go through `fit()`)
- **74.70%** accuracy, **92.90%** top-3 hit rate (v1 generator: 20.10% / 58.50%)
- Exports `models/xgboost_atm_zone.joblib` as `{model, zone_names, features}`
  so the backend never has to hardcode the zone mapping


### 6. Backend service — `backend/` (FastAPI)
Complaint CRUD endpoints, `Complaint` and `Prediction` tables with
`predicted_zone` / `confidence_score` columns ready for model output. The
exported `.joblib` payloads are shaped to feed those columns directly.

---


## Tech stack

Python 3.13 · pandas 3.0 · scikit-learn 1.9 · XGBoost 3.3 · joblib · FastAPI · SQLAlchemy

Model artifacts total ~14 MB and are git-ignored via `.gitignore`; the
committed evaluation outputs live in `reports/`.
