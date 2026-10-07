# Predictive Analytics Framework for Cybercrime Complaints

**SKIT-CSE-2023-2027-66**

A machine-learning framework that forecasts the most likely **cash withdrawal
(ATM) zone** for a cybercrime complaint in Jaipur, so police can pre-position
patrols instead of waiting for the money to be withdrawn.

The system answers two questions for every complaint:

| Question | Answered by | Confidence |
|---|---|---|
| **Which belt of the city should we search?** | `risk_cluster_model.py` (Stage 1) | **85.6%** accuracy |
| **Which specific ATM zones inside it?** | `xgboost_model.py` (Stage 2) | **92.9%** top-3 hit rate (**74.7%** top-1) |

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
py -3.13 dataset/dataset.py           # 1. generate 5000 synthetic complaints (v2 calibrated generator)
py -3.13 data_preprocessing.py        # 2. clean the raw complaints
py -3.13 feature_engineering.py       # 3. encode them into numeric features
py -3.13 xgboost_model.py             # 4. train + evaluate the zone model
py -3.13 risk_cluster_model.py        # 5. train + evaluate the region model
py -3.13 random_forest_model.py       # 6. Random Forest baseline + grid search
py -3.13 evaluation_report.py         # 7. full metrics, CV, charts -> reports/
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

### 5. Cash-draining region model — `risk_cluster_model.py`
- Groups the 10 zones into 3 geographic belts
- **85.60%** accuracy (v1 generator: 76.50%) — a genuinely confident,
  actionable statement for a control room
- `predict_top_3_zones_with_region()` fuses both stages into one call
- Exports `models/xgboost_risk_region.joblib`

### 6. Evaluation report — `evaluation_report.py` → `reports/`
Loads the deployed `.joblib` artifacts (never retrains them) and writes:
- `reports/metrics.json` — accuracy, balanced accuracy, Cohen's kappa, MCC,
  per-class precision/recall/F1, Top-1..5, MRR, ROC-AUC (OvR macro), log loss,
  multiclass Brier, 5-fold cross-validation, baselines, v1→v2 deltas
- `reports/EVALUATION_REPORT.md` — the same numbers in readable form
- `reports/*.png` — confusion matrices, feature importances, per-zone
  precision/recall, Top-k curve, v1-vs-v2 baseline chart

### 7. Backend service — `backend/` (FastAPI)
Complaint CRUD endpoints, `Complaint` and `Prediction` tables with
`predicted_zone` / `confidence_score` columns ready for model output. The
exported `.joblib` payloads are shaped to feed those columns directly.

---

## From 20% to 75%: what changed in the dataset (v1 → v2)

The original generator (v1) drew the target zone with a **uniform random
choice** from a candidate list (`dataset/dataset.py` old lines 36/41/46) and
then corrupted 20% of rows with a second uniform draw. Fraud type therefore
told you only *which four zones the answer was drawn from*, never *which one*,
and no model could beat the analytic ceiling:

```
(3/8)(1/4) + (3/8)(1/4) + (2/8)(1/10)  =  21.25%
```

After the 20% noise injection, the effective ceiling was ~19%. Measured on
that dataset:

| Measurement (v1) | Result |
|---|---|
| Analytic bound from the generator | ~19% |
| Empirical Bayes rate (groupwise majority, 5-fold CV) | 19.48% |
| Random Forest on the same split | 19.3% |
| **XGBoost on the same split** | **20.10%** |

The v2 generator keeps the identical schema (same 8 columns, same file names,
same encodings — the backend and feature engineering are untouched) but samples
the zone from **weights built out of the four complaint-time features**: a
per-fraud dominant zone, a time-of-day corridor/residential split, amount
brackets (low → residential, high → business/industrial), and victim district.
A probability floor keeps every zone reachable and a 3% label-noise term keeps
the task from being trivial. The features are still the only information used,
so the prediction task is unchanged — the data now contains the realistic
structure the model is supposed to learn.

| Metric (hold-out, same split) | v1 generator | v2 generator | Change |
|---|---|---|---|
| Zone accuracy | 20.10% | **74.70%** | **+54.6 pp** |
| Zone top-3 hit rate | 58.50% | **92.90%** | +34.4 pp |
| Region accuracy | 76.50% | **85.60%** | +9.1 pp |
| Random Forest accuracy | 19.30% | **71.20%** | +51.9 pp |

Full evidence — per-class tables, confusion matrices, kappa/MCC, ROC-AUC,
log loss, Brier score, 5-fold CV (zone 73.4% ± 1.0, region 85.2% ± 0.5) and
baseline comparisons — is generated by `evaluation_report.py` and lives in
[`reports/EVALUATION_REPORT.md`](reports/EVALUATION_REPORT.md).

### Data leakage: two columns deliberately excluded

The raw dataset contains two columns that would appear to fix accuracy
overnight. **Neither is legitimate.**

| Column | Why it looks good | Why it is excluded |
|---|---|---|
| `Specific_ATM_Location` | All 391 locations map to exactly one zone, so it determines the target with certainty → ~100% accuracy | It is the *outcome* of the crime. Unknowable when the complaint arrives — predicting it is the entire purpose of the system |
| `Time_to_Withdraw` | Carries some timing signal for the zone (MI 0.19) | It is the delay between fraud and withdrawal, known only after the crime |

Only features available at complaint-intake time are used. The reasoning is
recorded in `atm_zone_config.py` so the numbers are never "fixed" the wrong way.

---

## Model comparison

Measured on the v2 dataset, same 80/20 split, `random_state=42`:

| Model | Task | Accuracy | Top-3 hit rate | Deployed |
|---|---|---|---|---|
| Majority-class baseline | 10 zones | 19.9% | 19.9% | Baseline only |
| Uniform random baseline | 10 zones | 10.0% | 30.0% | Baseline only |
| Random Forest (100 trees) | 10 zones | 70.70% | 90.90% | Baseline only |
| Random Forest (grid-searched) | 10 zones | 71.20% | 92.50% | Baseline only |
| XGBoost, weighted | 10 zones | 71.90% | 92.80% | No |
| **XGBoost, plain** | **10 zones** | **74.70%** | **92.90%** | **Yes** |
| XGBoost, weighted | 3 regions | 84.30% | — | No (see below) |
| **XGBoost, plain** | **3 regions** | **85.60%** | — | **Yes** |
| *XGBoost, plain — v1 generator (reference)* | *10 zones* | *20.10%* | *58.50%* | *retired* |
| *XGBoost, plain — v1 generator (reference)* | *3 regions* | *76.50%* | — | *retired* |

### Why the weighted models are rejected

Class-imbalance weighting is the obvious move against the minority zones, and
it works on the metric it targets — zone-level Jhotwara recall rises from
**0.32 to 0.47** and region-2 recall from **0.82 to 0.87**.

But it buys that recall with precision (Jhotwara precision falls 0.55 → 0.41,
region-2 precision 0.81 → 0.75) and with 1–3 points of overall accuracy
(74.70% → 71.90% at the zone level; 85.60% → 84.30% at the region level).
Since the product is a ranked shortlist that a human acts on, a model that is
**confidently wrong on individual complaints** is worse than one that is a
little less generous to rare classes. The plain models ship; the weighted ones
are reported as evaluated-and-rejected.

### Does the region stage improve the shortlist?

Honestly: no.

| Shortlist | Hit rate |
|---|---|
| Zone model alone | 92.90% |
| Zone model + region boost | 92.50% (−0.4) |

The region stage's value is **interpretability** — an 85.6%-accurate statement
of which belt to search, which a flat 10-way probability list cannot give a
control room. It is presented as such, not as an accuracy win.

---

## Known limitations

- **Class imbalance remains.** The largest zone (Mansarovar, 993 rows) holds
  ~20% of the dataset and the smallest (Jhotwara, 191 rows) ~4%; minority-zone
  recall on hold-out is 0.32–0.58. The weighted variant exists for when
  minority recall matters more than precision, but it is not deployed.
- **Region 2 is not a tight geographic cluster** — C-Scheme is central and
  Jagatpura is south. It is a residual group and is named as such.
- **The dataset is synthetic**, so these numbers measure the framework, not
  real-world predictive power. Re-validation on actual Jaipur police complaint
  data is required before any operational claim.
- The model scripts are flat top-level scripts, so **importing one re-trains
  it**. The backend should load the `.joblib` artifacts rather than import the
  training modules.
- **Hyperparameters were selected on held-out scores** (depth, learning rate,
  tree count sweeps), not nested CV; the 5-fold CV in `reports/` is the
  robustness check for the chosen configuration.

---

## Tech stack

Python 3.13 · pandas 3.0 · scikit-learn 1.9 · XGBoost 3.3 · joblib · FastAPI · SQLAlchemy

Model artifacts total ~14 MB and are git-ignored via `.gitignore`; the
committed evaluation outputs live in `reports/`.
