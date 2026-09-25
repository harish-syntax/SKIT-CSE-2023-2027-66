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
   │  3 regions        20.10% acc / 58.50% top-3 hit rate           │
   │  76.50% acc                                                     │
   │      │                │                                         │
   │      ▼                ▼                                         │
   │  models/*.joblib   (dict payloads, git-ignored)                 │
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

Measured mutual information with the zone target:

| Feature | MI |
|---|---|
| `Fraud_Type_Code` | 0.318 |
| `Amount_Bracket_Code` | 0.264 |
| `Time_of_Day_Code` | 0.229 |
| `Victim_District_Code` | **0.011** |

`Victim_District_Code` is effectively noise. The generator draws the victim
district uniformly and independently of the zone, so it carries almost no
information — worth knowing before anyone invests effort in district-level
features.

---

## 4. The two-stage design

### Stage 2 — ATM zone (10 classes)

Predicts the exact zone. **20.10%** accuracy, **58.50%** top-3 hit rate.

A single zone guess is barely better than chance, so the product returns a
**ranked top-3 shortlist**: a primary patrol target plus two fallbacks. The
top-3 hit rate (58.5%) is roughly three times the top-1 rate, which is the
entire justification for ranking.

### Stage 1 — cash-draining region (3 classes)

The 10 zones group into three geographic belts:

| Region | Zones | Complaints | Character |
|---|---|---|---|
| 0 — West & NW Residential | Mansarovar, Vaishali Nagar, Malviya Nagar, Bani Park | 2,301 (46.0%) | Low-value, daytime frauds |
| 1 — South & East Industrial | Sitapura Industrial, Jhotwara, Pratap Nagar, Vidyadhar Nagar | 2,276 (45.5%) | High-value, night-time frauds |
| 2 — Central & Outlying Mixed | C-Scheme, Jagatpura | 423 (8.5%) | Residual group |

**76.50%** accuracy. This is the number to present, because it is a confident,
actionable statement rather than a near-uniform guess.

### Fusion

```python
combined_probability[zone] = zone_probability[zone] × region_probability[region_of(zone)]
```

Multiplication rather than hard-filtering: when the region model is torn
between two belts, zones in either stay visible instead of being discarded.

| Shortlist | Hit rate |
|---|---|
| Zone model alone | 58.50% |
| Zone + region boost | 59.20% (+0.7) |

**The region stage is not an accuracy win.** Its value is that a control room
gets "89% confident: West & NW residential" — a defensible instruction — rather
than a flat list of ten near-equal probabilities.

---

## 5. Why the zone task is capped at ~20%

`dataset/dataset.py:36,41,46` draw the zone with a **uniform random choice**:

```python
atm_zone = random.choice(['Sitapura Industrial', 'Pratap Nagar', 'Jhotwara', 'Vidyadhar Nagar'])
```

The draw is independent of amount, hour, district and everything else. Fraud
type determines *which candidate set* the zone came from, never *which zone
within it*. The best achievable predictor:

```
3/8 of rows → 4 candidates, best guess 1/4
3/8 of rows → 4 candidates, best guess 1/4
2/8 of rows → 10 candidates, best guess 1/10
──────────────────────────────────────────────────
             = 21.25%
```

With the 20% noise injection (`dataset/dataset.py:50-52`) forcing a uniform
re-draw, the effective ceiling is **~19%**.

Verified independently:

| Method | Result |
|---|---|
| Analytic bound from generator logic | ~19% |
| Empirical Bayes rate (groupwise majority, 5-fold CV) | 19.48% |
| Random Forest, 500 trees, same split | 19.3% |
| HistGradientBoosting, same split | 18.4% |
| **XGBoost (this project)** | **20.10%** |

**The model is at the ceiling.** The 20% is a property of the data, not a
tuning failure, and no additional model effort will move it. This is why the
framework is architected around a shortlist plus a coarser region call instead
of chasing a single high-accuracy label.

### Consequence

Achieving genuinely high zone-level accuracy requires changing the *data*, not
the model: the generator would need to sample the zone from **feature-dependent
weights** (and make victim district geographically meaningful) rather than
uniformly. That is out of scope for the current iteration, which works with the
dataset as given.

---

## 6. Data leakage guard

Two columns in the raw dataset would appear to solve the accuracy problem
overnight. Both are excluded, and the reasoning lives in `atm_zone_config.py`.

| Column | Apparent value | Why excluded |
|---|---|---|
| `Specific_ATM_Location` | All 391 locations map to exactly **one** zone each → ~100% accuracy | It is the *outcome* of the crime. The purpose of the system is to predict it, so it cannot be an input |
| `Time_to_Withdraw` | MI with zone = 0.22, comparable to `Fraud_Type` (0.32); cleanly separates the two large belts | It is the fraud→withdrawal delay, known only after the crime |

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

The signal is thin across 4 coarse features, so the models want heavy
regularisation — many small shallow trees beat a few deep ones. Measured on the
zone task:

| Config | Accuracy | Top-3 |
|---|---|---|
| n=300, lr=0.1, depth=8 | 17.20% | 55.7% |
| n=300, lr=0.1, depth=6 | 19.00% | 56.3% |
| **n=1000, lr=0.05, depth=3** | **20.10%** | **58.5%** |

Caveat: this was selected against held-out scores rather than nested CV. The
direction is consistent across five configurations, so it should generalise, but
a `GridSearchCV` would be the rigorous form.

### Why plain models ship and weighted models do not

Inverse-frequency weighting (`total_rows / (classes × class_count)`) is the
standard response to class imbalance, and it does improve the metric it targets:

| | Plain | Weighted |
|---|---|---|
| Region accuracy | **76.50%** | 71.70% |
| Region 2 recall | 0.01 | **0.74** |
| Region macro-F1 | 0.55 | **0.64** |
| Zone-level Jagatpura recall | 0.03 | **0.29** |

But it achieves that by flooding the rare class with false positives — 102 and
119 of them into an 85-row class. The consequence is a model that is
**confidently wrong on individual complaints**: on one test input the weighted
region model assigned **70.3% confidence to the wrong region**.

Because the product is a ranked shortlist that a human acts on, calibration
matters more than macro-F1. The plain models deploy; the weighted ones are
trained, reported, and explicitly labelled "comparison only".

---

## 8. Evaluation methodology

- **Split:** 80/20, `random_state=42`, stratified on the **zone** label in both
  scripts so the two stages share one test boundary and their numbers are
  directly comparable.
- **Metrics:** accuracy, classification report with readable class names,
  confusion matrix, feature importances, and **top-3 hit rate** (share of rows
  whose true zone appears in the shortlist) — the metric that reflects
  deployment value.
- **Coverage check:** the region evaluation warns explicitly when any region has
  recall below 10%, because a region the model never predicts means those zones
  never receive a patrol recommendation. This currently fires for Region 2.

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

1. **Region 2 (C-Scheme, Jagatpura) has ~1% recall** under the deployed model.
   These 423 rows arise only from the generator's fallback branch and are close
   to random. The script warns about this at runtime. Closing it needs features
   available at complaint time, not a model change.
2. **Zone-level probabilities are near-uniform** (top zone ~25%), so ranking
   *within* the shortlist is weakly informative. The region call is the
   dependable signal.
3. **Region 2 is not geographically tight** — C-Scheme is central, Jagatpura is
   south. It is a residual group, named as such rather than presented as a
   coherent region.
4. **The dataset is synthetic.** These numbers measure the framework, not
   real-world predictive power. Re-validation on actual complaint data is
   required before any operational claim.
5. **Hyperparameters were chosen on held-out scores**, not nested CV (§7).
6. **No drift or feedback loop yet.** Nothing monitors whether predictions stay
   accurate as crime patterns shift.

---

## 11. Suggested next steps

1. Wrap the training scripts in `if __name__ == "__main__":` so they become
   safely importable.
2. Add a `/predict` endpoint to the FastAPI service that loads both artifacts
   and calls `predict_top_3_zones_with_region()`.
3. Add nested-CV hyperparameter selection to replace the current held-out
   comparison.
4. Address Region 2 with genuinely predictive complaint-time features — victim
   geography relative to known cash-dumping sites is the most defensible
   candidate.
5. Add a drift monitor that re-evaluates accuracy as new complaints arrive.
6. Validate against real Jaipur police complaint data before any deployment
   claim.
