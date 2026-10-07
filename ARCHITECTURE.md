# Architecture

**Predictive Analytics Framework for Cybercrime Complaints** — system design,
dataflow, and the reasoning behind the modelling decisions.

---

## 1. System overview

```
   ┌─────────────────────────────────────────────────────────────────┐
   │  OFFLINE (run once, committed results)                          │
   │                                                                 │
   │  dataset/dataset.py                                             │
   │      5,000 synthetic complaints                                 │
   │      11 districts · 8 fraud types · 10 ATM zones                │
   │              │                                                  │
   │              ▼                                                  │
   │  data_preprocessing.py  →  jaipur_cybercrime_cleaned.csv        │
   │              │                                                  │
   │              ▼                                                  │
   │  feature_engineering.py                                         │
   │      manual encode → 4 numeric features                        │
   │              │                                                  │
   │              ▼                                                  │
   │  jaipur_ml_ready_manual_features.csv   (single ML input file)  │
   │              │                                                  │
   │      ┌───────┴────────┐                                         │
   │      ▼                ▼                                         │
    │  STAGE 1          STAGE 2                                       │
    │  risk_cluster_    xgboost_model.py                             │
    │  model.py         zone model, 10 classes                        │
    │  3 regions        74.70% acc / 92.90% top-3 hit rate            │
    │  85.60% acc                                                     │
    │      │                │                                         │
    │      ▼                ▼                                         │
    │  models/*.joblib   (dict payloads, git-ignored)                 │
    │      │                                                          │
    │      ▼                                                          │
    │  evaluation_report.py  →  reports/ (metrics.json, charts, CV)   │
    └──────────┬─────────────────┬───────────────────────────────────┘
              │                 │
              ▼                 ▼
   ┌─────────────────────────────────────────────────────────────────┐
   │  ONLINE (FastAPI, backend/)                                     │
   │                                                                 │
   │  POST /complaints  →  store complaint                           │
   │         │                                                       │
   │         ▼                                                       │
   │  load .joblib  →  predict_top_3_zones_with_region()             │
   │         │                                                       │
   │         ▼                                                       │
   │  Prediction table: predicted_zone, confidence_score             │
   └─────────────────────────────────────────────────────────────────┘
```

---

## 2. Module responsibilities

| File | Responsibility |
|---|---|
| `dataset/dataset.py` | Synthetic data generator |
| `data_preprocessing.py` | Raw complaint cleaning |
| `feature_engineering.py` | Manual encoding into numeric features |
| `atm_zone_config.py` | **Single source of truth**: zone/region maps, feature order, XGBoost params, leakage notes, input validation |
| `xgboost_model.py` | Stage 2 — zone model, evaluation, `predict_top_3_zones()` |
| `risk_cluster_model.py` | Stage 1 — region model, evaluation, `predict_top_3_zones_with_region()` |
| `random_forest_model.py` | Random Forest baseline + grid search |
| `evaluation_report.py` | Extended evaluation of the deployed artifacts → `reports/` (metrics.json, markdown report, charts, 5-fold CV, baselines) |
| `backend/` | FastAPI complaint CRUD, prediction persistence |

`atm_zone_config.py` exists so the zone mapping is defined **once**. Three
places previously needed it (the zone model, the region model, and the backend
payload). A mapping that exists in two places eventually disagrees, and the
disagreement is silent.

---

## 3. Feature set

Four features, all available at complaint-intake time, all produced by manual
encoding in `feature_engineering.py`:

| Feature | Codes | Source |
|---|---|---|
| `Fraud_Type_Code` | 0–7 | Fraud type dictionary |
| `Victim_District_Code` | 0–10 | Victim district dictionary |
| `Time_of_Day_Code` | 0–3 | Hour parsed from `Time_of_Complaint`, then bucketed |
| `Amount_Bracket_Code` | 0–3 | `Amount_INR` bucketed at 15k / 50k / 100k |

Measured mutual information with the zone target (v2 dataset):

| Feature | MI |
|---|---|
| `Fraud_Type_Code` | **1.107** |
| `Amount_Bracket_Code` | 0.328 |
| `Time_of_Day_Code` | 0.219 |
| `Victim_District_Code` | 0.147 |

On the v1 dataset, `Victim_District_Code` was effectively noise (MI 0.011)
because the generator drew districts independently of the zone. The v2
generator gives each district a geographically consistent zone affinity, so
the district now carries real — if modest — signal. It is the weakest of the
four features (feature importance ~0.13 in the trained model) but a genuine
tie-breaker rather than a dead column.

---

## 4. The two-stage design

### Stage 2 — ATM zone (10 classes)

Predicts the exact zone. **74.70%** accuracy, **92.90%** top-3 hit rate
(v1 generator: 20.10% / 58.50%).

The product still returns a **ranked top-3 shortlist**: a primary patrol
target plus two fallbacks. The top-3 hit rate lifts the hit rate from 74.7%
(top 1) to 92.9%, which is the justification for ranking rather than acting
on a single label.

### Stage 1 — cash-draining region (3 classes)

The 10 zones group into three geographic belts:

| Region | Zones | Complaints | Character |
|---|---|---|---|
| 0 — West & NW Residential | Mansarovar, Vaishali Nagar, Malviya Nagar, Bani Park | 2,612 (52.2%) | Low-value, daytime frauds |
| 1 — South & East Industrial | Sitapura Industrial, Jhotwara, Pratap Nagar, Vidyadhar Nagar | 1,260 (25.2%) | High-value, night-time frauds |
| 2 — Central & Outlying Mixed | C-Scheme, Jagatpura | 1,128 (22.6%) | Residual group |

**85.60%** accuracy (v1 generator: 76.50%). This is the number to present,
because it is a confident, actionable statement rather than a 10-way ranking.

### Fusion

```python
combined_probability[zone] = zone_probability[zone] × region_probability[region_of(zone)]
```

Multiplication rather than hard-filtering: when the region model is torn
between two belts, zones in either stay visible instead of being discarded.

| Shortlist | Hit rate |
|---|---|
| Zone model alone | 92.90% |
| Zone + region boost | 92.50% (−0.4) |

**The region stage is not an accuracy win.** Its value is that a control room
gets "97% confident: West & NW residential" — a defensible instruction — rather
than a flat list of ten probabilities.

---

## 5. From 20% to 75%: the zone task before and after

### v1 — uniform draw, ~20% ceiling (retired)

The original generator drew the zone with a **uniform random choice**:

```python
atm_zone = random.choice(['Sitapura Industrial', 'Pratap Nagar', 'Jhotwara', 'Vidyadhar Nagar'])
```

The draw was independent of amount, hour, district and everything else. Fraud
type determined *which candidate set* the zone came from, never *which zone
within it*. The best achievable predictor:

```
3/8 of rows → 4 candidates, best guess 1/4
3/8 of rows → 4 candidates, best guess 1/4
2/8 of rows → 10 candidates, best guess 1/10
──────────────────────────────────────────
             = 21.25%
```

With the 20% noise injection forcing a uniform re-draw, the effective ceiling
was **~19%**. Verified independently on that dataset:

| Method | Result |
|---|---|
| Analytic bound from generator logic | ~19% |
| Empirical Bayes rate (groupwise majority, 5-fold CV) | 19.48% |
| Random Forest, 500 trees, same split | 19.3% |
| HistGradientBoosting, same split | 18.4% |
| **XGBoost (this project)** | **20.10%** |

No model, ensemble or neural network could beat that, because the information
was simply not in the file.

### v2 — feature-dependent weights, 74.7% (current)

`dataset/dataset.py` now samples the zone from a weighted distribution built
from the four complaint-time features (naive-Bayes style fusion of
`exp(temperature × score)` vectors for fraud type, district, time of day and
amount bracket), with a probability floor so no zone is impossible and only
3% label noise. The schema is unchanged — same 8 columns, same file names,
same encodings — so `feature_engineering.py`, the training scripts and the
backend all work unmodified.

| Method (v2 dataset, same split) | Zone accuracy | Zone top-3 |
|---|---|---|
| Majority-class baseline | 19.9% | 19.9% |
| Uniform random | 10.0% | 30.0% |
| Random Forest (tuned) | 71.2% | 92.5% |
| Empirical ceiling check (4-feature Bayes) | 83.5%* | — |
| **XGBoost (deployed)** | **74.70%** | **92.90%** |

\* The 4-feature Bayes rate (the best any classifier can do from these four
columns) is 83.5%; the deployed model reaches ~90% of it, and 5-fold CV puts
it at 73.4% ± 1.0. The remaining gap is irreducible feature ambiguity — some
complaints genuinely fit several zones.

Every number in this section, plus per-class reports, kappa/MCC, ROC-AUC,
log loss, Brier scores, CV folds and charts, is regenerated by
`evaluation_report.py` into `reports/`.

---

## 6. Data leakage guard

Two columns in the raw dataset would appear to solve the accuracy problem
overnight. Both are excluded, and the reasoning lives in `atm_zone_config.py`.

| Column | Apparent value | Why excluded |
|---|---|---|
| `Specific_ATM_Location` | All 391 locations map to exactly **one** zone each → ~100% accuracy | It is the *outcome* of the crime. The purpose of the system is to predict it, so it cannot be an input |
| `Time_to_Withdraw` | MI with zone = 0.19 (v2 dataset); some timing structure | It is the fraud→withdrawal delay, known only after the crime |

**Rule: only features available at complaint-intake time may be used.**

This is recorded in code rather than in a comment thread so that nobody later
"fixes" the accuracy number by reaching for these columns, and so a reviewer can
see the exclusion was deliberate.

---

## 7. Model selection decisions

### Hyperparameters

Shared in `atm_zone_config.XGB_PARAMS`:

```python
n_estimators=1000, learning_rate=0.05, max_depth=3,
subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1
```

The signal is spread across 4 coarse features, so the models want heavy
regularisation — many small shallow trees beat a few deep ones. Measured on
the v1 zone task:

| Config | Accuracy | Top-3 |
|---|---|---|
| n=300, lr=0.1, depth=8 | 17.20% | 55.7% |
| n=300, lr=0.1, depth=6 | 19.00% | 56.3% |
| **n=1000, lr=0.05, depth=3** | **20.10%** | **58.5%** |

The same ordering held on the v2 dataset: every deeper variant tested
(depths 4–7) scored worse than depth 3, and depth 3 / lr 0.05 / 1000 trees
remains the deployed configuration (now 74.70% / 92.90%).

Caveat: this was selected against held-out scores rather than nested CV. The
direction is consistent across configurations on both datasets, and the 5-fold
CV in `reports/` is the robustness check, but a `GridSearchCV` would be the
rigorous form.

### Why plain models ship and weighted models do not

Inverse-frequency weighting (`total_rows / (classes × class_count)`) is the
standard response to class imbalance, and it does improve the metric it targets
(measured on the v2 dataset):

| | Plain | Weighted |
|---|---|---|
| Zone accuracy | **74.70%** | 71.90% |
| Zone top-3 | **92.90%** | 92.80% |
| Zone Jhotwara recall | 0.32 | **0.47** |
| Zone Jhotwara precision | **0.55** | 0.41 |
| Region accuracy | **85.60%** | 84.30% |
| Region 2 recall | 0.82 | **0.87** |
| Region 2 precision | **0.81** | 0.75 |

It buys minority recall by flooding those classes with false positives —
Jhotwara's precision is nearly halved, so almost half of the complaints the
weighted model assigns to it are wrong. Because the product is a ranked
shortlist that a human acts on, calibration matters more than macro-F1. The
plain models deploy; the weighted ones are trained, reported, and explicitly
labelled "comparison only".

---

## 8. Evaluation methodology

- **Split:** 80/20, `random_state=42`, stratified on the **zone** label in both
  scripts so the two stages share one test boundary and their numbers are
  directly comparable.
- **Metrics:** accuracy, classification report with readable class names,
  confusion matrix, feature importances, and **top-3 hit rate** (share of rows
  whose true zone appears in the shortlist) — the metric that reflects
  deployment value.
- **Extended report:** `evaluation_report.py` loads the deployed artifacts and
  adds balanced accuracy, Cohen's kappa, MCC, per-class tables, Top-1..5,
  MRR, ROC-AUC (OvR macro), log loss, multiclass Brier, 5-fold stratified CV
  (zone 73.4% ± 1.0, region 85.2% ± 0.5), baseline comparisons
  (majority / uniform / Random Forest / v1 reference) and charts. Outputs are
  committed under `reports/` (`metrics.json`, `EVALUATION_REPORT.md`, PNGs).
- **Coverage check:** the region evaluation warns explicitly when any region
  has recall below 10%, because a region the model never predicts means those
  zones never receive a patrol recommendation. On the v2 dataset this does
  **not** fire — the lowest region recall is 0.82 — but the guard stays in
  place for future datasets.

---

## 9. Contract with the backend

Each script exports a dict rather than a bare estimator, so the API receives the
mappings and the feature order alongside the model:

```python
# models/xgboost_atm_zone.joblib
{"model": XGBClassifier, "zone_names": {0: "Mansarovar", ...}, "features": [...]}

# models/xgboost_risk_region.joblib
{"model": XGBClassifier, "region_names": {...}, "zone_region": {...}, "features": [...]}
```

`predict_top_3_zones_with_region()` returns:

```python
{
  "region_code": 0,
  "region": "West & NW Residential",
  "region_probability": 0.8947,
  "top_zones": [
    {"rank": 1, "zone_code": 3, "zone": "Malviya Nagar",
     "region": "West & NW Residential",
     "zone_probability": 0.2595, "combined_probability": 0.2339},
    ...
  ]
}
```

This maps directly onto the `Prediction` table's `predicted_zone` (String) and
`confidence_score` (Float) columns.

**Loading discipline:** the training scripts are flat top-level scripts, so
*importing* one re-trains it. The backend must load the `.joblib` artifacts, not
import the training modules. Converting the scripts to `main()` guards is a
sensible follow-up.

---

## 10. Known limitations

1. **Class imbalance remains.** The largest zone (Mansarovar, 993 rows) is
   ~20% of the dataset and the smallest (Jhotwara, 191 rows) ~4%; hold-out
   recall for the minority zones is 0.32–0.58. The weighted variant trades
   precision and 1–3 points of accuracy for minority recall; it is measured
   and reported, not deployed (§7).
2. **Region 2 is not geographically tight** — C-Scheme is central, Jagatpura is
   south. It is a residual group, named as such rather than presented as a
   coherent region.
3. **The dataset is synthetic.** These numbers measure the framework, not
   real-world predictive power. Re-validation on actual complaint data is
   required before any operational claim.
4. **Hyperparameters were chosen on held-out scores**, not nested CV (§7);
   the 5-fold CV in `reports/` is the robustness check.
5. **No drift or feedback loop yet.** Nothing monitors whether predictions stay
   accurate as crime patterns shift.

---

## 11. Suggested next steps

1. Wrap the training scripts in `if __name__ == "__main__":` so they become
   safely importable.
2. Add a `/predict` endpoint to the FastAPI service that loads both artifacts
   and calls `predict_top_3_zones_with_region()`.
3. Add nested-CV hyperparameter selection to replace the current held-out
   comparison.
4. Improve minority-zone recall (Jhotwara, Malviya Nagar) with more training
   rows for the rare zones or complaint-time features that separate them from
   their neighbours — rather than with class weighting, which §7 shows costs
   more than it buys.
5. Add a drift monitor that re-evaluates accuracy as new complaints arrive.
6. Validate against real Jaipur police complaint data before any deployment
   claim.
