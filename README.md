# 🏥 Hospital Readmission Risk — an end-to-end ML system

Predicting whether a diabetic patient will be **readmitted to hospital within 30
days** of discharge, from a real dataset of **100,000+ US hospital encounters**.

This project is built as a **complete, production-style machine-learning system** —
not a notebook. It covers the full lifecycle: raw data → cleaning → feature
engineering → tracked model training → honest evaluation → a served REST API → a
containerised deployment → a live interactive app → drift monitoring.

> **🔗 Live demo:** https://tarun-readmission-risk.streamlit.app
> **📊 Experiment tracking, REST API, Docker, CI, monitoring** — all included.

---

## What I set out to do

Most of my earlier work lived in notebooks, and I knew that to move toward a data-scientist
role I had to show I could actually ship a model, not just train one. So I set myself a harder
challenge: take a real, messy healthcare problem — predicting whether a diabetic patient is
readmitted within 30 days of discharge — and build it the way it would actually exist in
production.

That meant doing every stage myself and being careful where it counts. I cleaned the data with
leakage in mind (the same patient shows up many times, so I kept only their first encounter),
engineered the features, tuned the models with Bayesian optimization, calibrated the
probabilities so a "10%" really means 10%, tracked the experiments in MLflow, served the model
through a FastAPI endpoint, containerised it with Docker, added tests and CI, deployed a live
Streamlit app, and set up drift monitoring.

**How it turned out:** the system side came together fully — it's live and you can try it. On
the prediction side I made a point of staying honest: the ROC-AUC lands around 0.66, which is
right in line with published results on this dataset because readmission is genuinely hard to
predict. I'd rather show a model that's good at ranking who's most at risk than dress it up as
something it isn't.

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
| ROC-AUC (held-out test) | **0.658** (95% CI 0.642–0.675) |
| PR-AUC | **0.193** (≈ 2× the 9% base rate) |
| Probabilities | **calibrated** (isotonic) — a "10%" really means ~10% |
| Decision threshold | **0.121**, tuned on *validation* (not test) |

**Readmission is intrinsically hard to predict** — published studies on this
dataset report ROC-AUC around **0.64–0.68**, and this system sits squarely in that
range. The value is in *ranking* who is most at risk, not perfect prediction. I
deliberately report this honestly rather than over-claiming, and the app says so
too.

---

## Architecture

```
data download ──► cleaning ──► feature engineering ──► model training
 (UCI API)      (leakage &     (ICD-9 grouping,     (4 models, Bayesian-tuned,
                label fixes)    derived features)     calibrated, 5-fold CV)
                                                              │
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
| Hyperparameter tuning | **scikit-optimize** (`BayesSearchCV`, Bayesian optimisation) |
| Statistics | scipy (Friedman + Wilcoxon-Holm model comparison; VIF) |
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
5. **Bayesian hyperparameter optimisation.** Every model's hyperparameters are
   searched with **scikit-optimize `BayesSearchCV`** (PR-AUC objective, fixed
   seed) — efficient, reproducible tuning rather than hand-picked defaults.
6. **Assumptions are checked, not assumed.** Parametric (logistic: independence,
   multicollinearity/VIF, events-per-variable) vs non-parametric (tree) model
   requirements are verified, and the models are compared **statistically**
   (Friedman + Wilcoxon-Holm), not by eyeballing one number.
7. **Leakage-safe pipeline.** All encoding/scaling lives inside a scikit-learn
   `Pipeline` fitted on the training fold only — so cross-validation and serving
   are honest.
8. **Reproducibility & MLOps.** Config-driven, version-pinned, experiment-tracked,
   tested, containerised, CI-gated, and monitored for drift.

## Model comparison (selection on the validation set)

| Model | Validation ROC-AUC | Validation PR-AUC |
|---|---|---|
| **XGBoost** ✅ | **0.658** | **0.181** |
| LightGBM | 0.657 | 0.181 |
| Random Forest | 0.653 | 0.178 |
| Logistic Regression | 0.643 | 0.164 |

XGBoost is selected on the **validation** set (by PR-AUC), then evaluated **once**
on the untouched test set: **ROC-AUC 0.658 (95% CI 0.642–0.675)**. A Friedman test
across the cross-validation folds does detect *some* difference among the four
(χ²=10.7, **p=0.014**) — but that is driven almost entirely by **logistic
regression lagging**; the three tree models are statistically indistinguishable
from one another (every pairwise Wilcoxon–Holm comparison is non-significant, XGBoost
vs LightGBM p=1.0). So "XGBoost is best" is a mild preference over LightGBM/RF, not a
strong claim — and the project says so rather than overselling.

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
- **Bayesian hyperparameter optimisation** (`BayesSearchCV`, PR-AUC objective)
  tunes each model via cross-validation **inside the training set only** — the
  search never sees validation or test data. `scale_pos_weight`/class weights are
  deliberately excluded from the search so the probabilities stay calibratable.
- **Model assumptions verified** (`python -m src.models.diagnostics`): logistic
  regression's parametric assumptions (independence — engineered via first-
  encounter dedup; VIF multicollinearity; events-per-variable) are checked, while
  the tree models' freedom from those assumptions is documented. The four models
  are then compared with a **Friedman test + Wilcoxon-Holm** post-hoc — reported
  honestly as illustrative given a single dataset.
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
│   │   ├── train.py            # Bayesian-tune 4 models, calibrate, MLflow + registry
│   │   ├── diagnostics.py      # assumption checks + Friedman/Wilcoxon comparison
│   │   └── evaluate.py         # honest test report + diagnostic plots
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
python -m src.models.train              # Bayesian-tune 4 models, calibrate, log to MLflow
python -m src.models.diagnostics        # assumption checks + statistical model comparison
python -m src.models.evaluate           # honest test report + save plots
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
- **Next steps:** SHAP explanations per prediction, native categorical / target
  encoding, automated retraining when drift is detected, deploy the API behind a
  cloud host.

## Data

UCI Machine Learning Repository — *Diabetes 130-US hospitals for years 1999–2008*.
Strack et al. (2014), *Impact of HbA1c Measurement on Hospital Readmission Rates*,
BioMed Research International.
```bash
python -m src.data.download   # fetches it automatically
```
