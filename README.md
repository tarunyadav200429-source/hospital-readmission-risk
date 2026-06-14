# 🏥 Hospital Readmission Risk — an end-to-end ML system

Predicting whether a diabetic patient will be **readmitted to hospital within 30
days** of discharge, from a real dataset of **100,000+ US hospital encounters**.

This project is built as a **complete, production-style machine-learning system** —
not a notebook. It covers the full lifecycle: raw data → cleaning → feature
engineering → tracked model training → honest evaluation → a served REST API → a
containerised deployment → a live interactive app → drift monitoring.

> **🔗 Live demo:** _deploying — URL coming here_
> **📊 Experiment tracking, REST API, Docker, CI, monitoring** — all included.

---

## Why this problem matters

Unplanned 30-day readmissions are a major cost and quality problem in healthcare —
hospitals are financially penalised for them. If a hospital can flag the patients
most likely to bounce back, it can target follow-up care (calls, medication
review, home visits) where it helps most. This is a real, deployed use-case for
machine learning, and a genuinely **hard** prediction problem.

## Headline results (honest)

| | |
|---|---|
| Best model | **XGBoost** (chosen on validation; vs LogReg, RF, LightGBM) |
| ROC-AUC (held-out test) | **0.659** (95% CI 0.642–0.675) |
| PR-AUC | **0.190** (≈ 2× the 9% base rate) |
| Probabilities | **calibrated** (isotonic) — a "10%" really means ~10% |
| Decision threshold | **0.128**, tuned on *validation* (not test) |

**Readmission is intrinsically hard to predict** — published studies on this
dataset report ROC-AUC around **0.64–0.68**, and this system sits squarely in that
range. The value is in *ranking* who is most at risk, not perfect prediction. I
deliberately report this honestly rather than over-claiming, and the app says so
too.

---

## Architecture

```
data download ──► cleaning ──► feature engineering ──► model training
 (UCI API)      (leakage &     (ICD-9 grouping,        (4 models, 5-fold CV)
                label fixes)    derived features)             │
                                                   MLflow experiment tracking
                                                   + model registry
                                                              │
                              ┌───────────────────────────────┴───────────────┐
                       FastAPI /predict service              Streamlit demo app
                       (validated REST API)                  (live, clickable)
                              │                                      │
                       Docker container                     Streamlit Community Cloud
                              │
              pytest  +  GitHub Actions CI  +  PSI data-drift monitor
```

## Tech stack

| Layer | Tools |
|---|---|
| Data & modelling | pandas, scikit-learn, **XGBoost**, **LightGBM** |
| Experiment tracking | **MLflow** (params, metrics, model registry) |
| Model serving | **FastAPI** + Pydantic validation, Uvicorn |
| Packaging | **Docker** |
| Testing & CI | **pytest** + **GitHub Actions** |
| App | **Streamlit** + Plotly |
| Monitoring | Population Stability Index (PSI) drift report |

---

## What makes this a *data-scientist* project (not just analysis)

1. **Leakage control.** The same patients appear many times. I keep only each
   patient's **first encounter** — this removed **29,353 leaking rows** (~30% of
   the data) that would otherwise inflate the score. (Methodology of Strack et
   al., 2014.)
2. **Label-contamination fix.** Encounters ending in **death or hospice** are
   dropped — those patients cannot be readmitted, so leaving them in poisons the
   target.
3. **Real feature engineering.** 700+ raw ICD-9 diagnosis codes are grouped into
   **9 clinical categories**; derived features (`total_prior_visits`,
   `num_med_changes`) capture clinical intuition.
4. **Imbalanced-learning done right.** The positive class is ~9%. Rather than
   reweighting (which distorts probabilities), I keep natural probabilities,
   **calibrate** them (isotonic), handle the imbalance with a **tuned decision
   threshold**, and report **threshold-independent metrics** (ROC-AUC, PR-AUC).
5. **Leakage-safe pipeline.** All encoding/scaling lives inside a scikit-learn
   `Pipeline` fitted on the training fold only — so cross-validation and serving
   are honest.
6. **Reproducibility & MLOps.** Config-driven, version-pinned, experiment-tracked,
   tested, containerised, CI-gated, and monitored for drift.

## Model comparison (selection on the validation set)

| Model | Validation ROC-AUC | Validation PR-AUC |
|---|---|---|
| **XGBoost** ✅ | **0.658** | 0.181 |
| Random Forest | 0.650 | 0.176 |
| LightGBM | 0.646 | 0.173 |
| Logistic Regression | 0.642 | 0.164 |

XGBoost is selected on the **validation** set, then evaluated **once** on the
untouched test set: **ROC-AUC 0.659 (95% CI 0.642–0.675)**. Note the models span
only ~0.016 AUC and their bootstrap intervals overlap heavily — they are
**statistically close to indistinguishable**, so "XGBoost is best" is a mild
preference, not a strong claim (a point the project states honestly rather than
overselling).

## Methodology & rigor (how leakage is avoided)

This project is deliberately strict about evaluation honesty:

- **Three-way split — train / validation / test.** Models are *fit* on train,
  *selected* on validation, and the threshold is *tuned* on validation. The test
  set is touched **exactly once**, at the very end, so its numbers are unbiased.
- **No threshold tuning on test, no model selection on test** — both are common
  silent leaks that inflate reported scores. Here both happen on validation.
- **Calibrated probabilities.** Isotonic calibration (5-fold) means the predicted
  probabilities are trustworthy, not just rank-ordered — important because the app
  shows a probability to a user.
- **Bootstrap 95% confidence intervals** on the test metrics, so claims are
  reported with their uncertainty (and competing models shown to be
  statistically close).
- **Leakage-safe feature pipeline** — all encoding/scaling is fit inside the
  scikit-learn `Pipeline` on training folds only.

---

## Project structure

```
hospital-readmission-risk/
├── config/config.yaml          # all settings in one place
├── src/
│   ├── config.py               # config loader
│   ├── data/
│   │   ├── download.py         # download + audit raw data
│   │   └── preprocess.py       # cleaning, leakage/label fixes, target
│   ├── features/build_features.py  # ICD-9 grouping + derived features
│   ├── models/
│   │   ├── pipeline.py         # split + leakage-safe preprocessing
│   │   ├── train.py            # train 4 models, MLflow tracking + registry
│   │   └── evaluate.py         # threshold tuning + diagnostic plots
│   ├── api/main.py             # FastAPI /predict service
│   └── monitoring/drift.py     # PSI data-drift monitor
├── app/streamlit_app.py        # interactive demo
├── tests/test_features.py      # pytest unit tests
├── .github/workflows/ci.yml    # GitHub Actions CI
├── Dockerfile                  # containerised API
├── requirements.txt            # full pinned deps
└── README.md
```

## How to reproduce

```bash
# 1. install
pip install -r requirements.txt

# 2. run the pipeline (each step prints what it does)
python -m src.data.download            # download + audit raw data
python -m src.features.build_features   # clean + engineer features
python -m src.models.train              # train 4 models, log to MLflow
python -m src.models.evaluate           # tune threshold + save plots
python -m src.monitoring.drift          # data-drift report

# 3. view the tracked experiments
mlflow ui --backend-store-uri sqlite:///mlflow.db   # -> http://127.0.0.1:5000

# 4. serve the model as an API
uvicorn src.api.main:app --reload       # -> http://127.0.0.1:8000/docs

# 5. or run it in Docker
docker build -t readmission-api .
docker run -p 8000:8000 readmission-api

# 6. run the interactive app
streamlit run app/streamlit_app.py

# 7. run the tests
pytest -v
```

## Limitations & future work

- **Predictability ceiling.** ROC-AUC ~0.65 reflects how hard readmission is from
  administrative data alone; richer clinical/lab/time-series data would help.
- **Single dataset, single era** (1999–2008, US hospitals) — external validity is
  limited; the model should be re-validated and re-calibrated before any real use.
- **Not for clinical use.** This is a portfolio/educational project, not a
  validated medical device.
- **Next steps:** SHAP explanations per prediction, hyperparameter search with
  Optuna, automated retraining when drift is detected, deploy the API behind a
  cloud host.

## Data

UCI Machine Learning Repository — *Diabetes 130-US hospitals for years 1999–2008*.
Strack et al. (2014), *Impact of HbA1c Measurement on Hospital Readmission Rates*,
BioMed Research International.
```bash
python -m src.data.download   # fetches it automatically
```
