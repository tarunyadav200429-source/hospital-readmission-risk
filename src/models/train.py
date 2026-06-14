"""
train.py  --  STEP 3: Bayesian-tune the models, pick the best, calibrate it, and
track everything in MLflow.

WHAT CHANGED (and why): the candidate models are no longer trained with hand-
picked hyperparameters. Each one is tuned with BAYESIAN OPTIMISATION
(scikit-optimize's BayesSearchCV) -- the same approach used in the credit-risk
thesis. Bayesian optimisation builds a probabilistic model of
"hyperparameters -> score" and uses it to choose the next combination to try, so
it finds strong settings in far fewer trials than grid or random search.

We keep a clean THREE-WAY split so nothing leaks:
    * train      -> the Bayesian search cross-validates ON THIS ONLY
    * validation -> choose the best tuned model AND tune the decision threshold
    * test       -> NEVER touched here (only src/models/evaluate.py uses it)

Design decisions (and why):
  * No class reweighting. Reweighting (scale_pos_weight / class_weight) does not
    improve ranking and it DISTORTS the predicted probabilities, which we show in
    the app. So -- unlike the thesis search space -- we deliberately leave
    scale_pos_weight OUT of the search and handle the 9% imbalance purely through
    (a) a PR-AUC tuning objective and (b) a tuned decision threshold.
  * Selection metric = PR-AUC (average_precision). On a ~9%-positive problem
    ROC-AUC can look flattering; PR-AUC focuses on how well we rank the rare
    positive (readmitted) patients, so we tune AND select on it.
  * Probability calibration. The winner is wrapped in CalibratedClassifierCV
    (isotonic) so a "10%" really means ~10% readmission risk.
  * Threshold tuned on VALIDATION, never on test.

    python -m src.models.train
"""

import json
import warnings

import joblib
import mlflow
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from skopt import BayesSearchCV
from skopt.space import Categorical, Integer, Real
from xgboost import XGBClassifier

from src.config import load_config, PROJECT_ROOT
from src.models.pipeline import build_preprocessor, load_model_input, split_data

# skopt is a little noisy about deprecations on newer scikit-learn; quiet it.
warnings.filterwarnings("ignore")


def get_models_and_spaces(seed: int) -> dict:
    """Each candidate = (estimator, Bayesian search space).

    Search-space keys are prefixed with ``clf__`` because the estimator is the
    LAST step of a Pipeline (prep -> clf); the preprocessing is therefore re-fit
    inside every CV fold, so NOTHING leaks from validation folds into training.

    NOTE: every search space deliberately OMITS class weighting (e.g.
    scale_pos_weight) -- see the module docstring. We want calibratable
    probabilities, not reweighted scores.
    """
    return {
        # ---- Logistic Regression: PARAMETRIC (linear decision boundary in the
        # log-odds). liblinear lets us search both L1 and L2 penalties. ----------
        "logistic_regression": (
            LogisticRegression(max_iter=2000, solver="liblinear", random_state=seed),
            {
                "clf__C": Real(1e-3, 1e2, prior="log-uniform"),
                "clf__penalty": Categorical(["l1", "l2"]),
            },
        ),
        # ---- Random Forest: NON-PARAMETRIC (tree ensemble, no dist/linearity
        # assumptions). Ranges trimmed from the thesis for laptop runtime. -------
        "random_forest": (
            RandomForestClassifier(random_state=seed, n_jobs=1),
            {
                "clf__n_estimators": Integer(100, 400),
                "clf__max_depth": Integer(4, 20),
                "clf__min_samples_split": Integer(2, 20),
                "clf__min_samples_leaf": Integer(1, 20),
                "clf__max_features": Real(0.1, 1.0, prior="uniform"),
            },
        ),
        # ---- XGBoost: NON-PARAMETRIC gradient boosting. Same knobs as the
        # thesis MINUS scale_pos_weight (kept out on purpose). -------------------
        "xgboost": (
            XGBClassifier(eval_metric="logloss", n_jobs=1, random_state=seed),
            {
                "clf__n_estimators": Integer(100, 500),
                "clf__learning_rate": Real(0.01, 0.3, prior="log-uniform"),
                "clf__max_depth": Integer(3, 8),
                "clf__min_child_weight": Integer(1, 10),
                "clf__subsample": Real(0.5, 1.0, prior="uniform"),
                "clf__colsample_bytree": Real(0.5, 1.0, prior="uniform"),
                "clf__gamma": Real(0.0, 5.0, prior="uniform"),
                "clf__reg_lambda": Real(1e-3, 10.0, prior="log-uniform"),
                "clf__reg_alpha": Real(1e-3, 10.0, prior="log-uniform"),
            },
        ),
        # ---- LightGBM: NON-PARAMETRIC gradient boosting (leaf-wise). -----------
        "lightgbm": (
            LGBMClassifier(n_jobs=1, random_state=seed, verbose=-1, subsample_freq=1),
            {
                "clf__n_estimators": Integer(100, 500),
                "clf__num_leaves": Integer(15, 127),
                "clf__learning_rate": Real(0.01, 0.3, prior="log-uniform"),
                "clf__subsample": Real(0.5, 1.0, prior="uniform"),
                "clf__colsample_bytree": Real(0.5, 1.0, prior="uniform"),
                "clf__min_child_samples": Integer(5, 60),
                "clf__reg_lambda": Real(1e-3, 10.0, prior="log-uniform"),
            },
        ),
    }


def best_f1_threshold(y_true, proba) -> float:
    """Threshold that maximises F1 -- tuned on VALIDATION data only."""
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    return float(thresholds[np.argmax(f1[:-1])])


def tune_one(name, estimator, space, preprocessor, X, y, tcfg):
    """Bayesian-optimise one model's hyperparameters via cross-validation.

    Returns (BayesSearchCV or None, fitted tuned pipeline, per-fold CV scores of
    the BEST configuration). We keep those folds so diagnostics.py can run a
    Friedman test comparing the models statistically -- exactly as the thesis did.
    """
    seed = tcfg["random_state"]
    pipe = Pipeline([("prep", clone(preprocessor)), ("clf", estimator)])

    if not tcfg.get("enabled", True):
        # Fast path: skip the search, just fit defaults once and report no folds.
        pipe.fit(X, y)
        return None, pipe, [float("nan")]

    cv = StratifiedKFold(n_splits=tcfg["cv"], shuffle=True, random_state=seed)
    opt = BayesSearchCV(
        pipe,
        search_spaces=space,
        n_iter=tcfg["n_iter"],
        cv=cv,
        scoring=tcfg["scoring"],
        n_jobs=-1,          # parallelise the SEARCH; estimators stay n_jobs=1 to
        random_state=seed,  #   avoid thread oversubscription (a known gotcha)
        refit=True,
    )
    np.random.seed(seed)    # mirrors the thesis: lock the search RNG too
    opt.fit(X, y)

    bi = opt.best_index_
    fold_scores = [float(opt.cv_results_[f"split{k}_test_score"][bi])
                   for k in range(tcfg["cv"])]
    return opt, opt.best_estimator_, fold_scores


def main():
    cfg = load_config()
    tcfg = cfg["tuning"]
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])
    seed = cfg["split"]["random_state"]

    # --- three-way split: train_full -> (train, val); test held out ----------
    df = load_model_input()
    X_train_full, X_test, y_train_full, y_test = split_data(df)   # test untouched
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.25,
        random_state=seed, stratify=y_train_full)
    print(f"train={len(X_train):,}  val={len(X_val):,}  test(held out)={len(X_test):,}")
    if tcfg["enabled"]:
        print(f"Bayesian tuning: n_iter={tcfg['n_iter']}, cv={tcfg['cv']}, "
              f"scoring={tcfg['scoring']}\n")

    preprocessor = build_preprocessor(X_train_full)

    # --- 1) Bayesian-tune each model, then score the winner on VALIDATION -----
    results = {}
    cv_fold_scores = {}                 # for the Friedman/Wilcoxon comparison
    best_name, best_pipe, best_score = None, None, -1.0
    for name, (estimator, space) in get_models_and_spaces(seed).items():
        opt, tuned_pipe, fold_scores = tune_one(
            name, estimator, space, preprocessor, X_train, y_train, tcfg)
        cv_fold_scores[name] = fold_scores

        val_proba = tuned_pipe.predict_proba(X_val)[:, 1]
        best_params = (
            {k: (v.item() if hasattr(v, "item") else v)
             for k, v in dict(opt.best_params_).items()}
            if opt is not None else "defaults")
        m = {
            "val_roc_auc": roc_auc_score(y_val, val_proba),
            "val_pr_auc": average_precision_score(y_val, val_proba),
            "cv_best_pr_auc": float(np.nanmean(fold_scores)),
            "best_params": best_params,
        }
        results[name] = m
        print(f"  {name:20} CV PR-AUC={m['cv_best_pr_auc']:.4f}  "
              f"val ROC-AUC={m['val_roc_auc']:.4f}  val PR-AUC={m['val_pr_auc']:.4f}")

        with mlflow.start_run(run_name=name):
            mlflow.log_param("model", name)
            if opt is not None:
                mlflow.log_params({k: str(v) for k, v in best_params.items()})
            mlflow.log_metrics({k: v for k, v in m.items() if isinstance(v, float)})

        # SELECT on validation PR-AUC (imbalance-appropriate ranking metric).
        if m["val_pr_auc"] > best_score:
            best_name, best_pipe, best_score = name, tuned_pipe, m["val_pr_auc"]
    print(f"\nSelected (by validation PR-AUC): {best_name} ({best_score:.4f})")

    # --- 2) calibrate the winner on TRAIN, tune threshold on VALIDATION ------
    # clone(best_pipe) gives an UNFITTED pipeline carrying the tuned
    # hyperparameters; CalibratedClassifierCV then refits it internally (5-fold)
    # and maps its raw scores onto honest probabilities.
    calib_for_threshold = CalibratedClassifierCV(
        clone(best_pipe), method="isotonic", cv=5)
    calib_for_threshold.fit(X_train, y_train)
    val_proba = calib_for_threshold.predict_proba(X_val)[:, 1]
    threshold = best_f1_threshold(y_val, val_proba)
    print(f"Tuned decision threshold (on validation): {threshold:.3f}")

    # --- 3) refit the FINAL calibrated model on ALL non-test data ------------
    final_model = CalibratedClassifierCV(
        clone(best_pipe), method="isotonic", cv=5)
    final_model.fit(X_train_full, y_train_full)

    # --- 4) register + save --------------------------------------------------
    with mlflow.start_run(run_name=f"best_{best_name}_calibrated"):
        mlflow.log_param("model", best_name)
        mlflow.log_param("tuning", "bayesian" if tcfg["enabled"] else "defaults")
        mlflow.log_param("calibration", "isotonic")
        mlflow.log_param("decision_threshold", threshold)
        mlflow.log_metrics({k: v for k, v in results[best_name].items()
                            if isinstance(v, float)})
        mlflow.sklearn.log_model(
            final_model, name="model",
            registered_model_name=cfg["model"]["registered_name"])

    out_dir = PROJECT_ROOT / cfg["model"]["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, out_dir / "model.joblib")
    with open(out_dir / "threshold.json", "w") as f:
        json.dump({"threshold": threshold}, f, indent=2)

    schema = {"numeric": [], "categorical": {}}
    for col in X_train_full.columns:
        if X_train_full[col].dtype == object:
            schema["categorical"][col] = sorted(X_train_full[col].astype(str).unique())
        else:
            schema["numeric"].append(col)
    with open(out_dir / "feature_schema.json", "w") as f:
        json.dump(schema, f, indent=2)

    outputs_dir = PROJECT_ROOT / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    with open(outputs_dir / "model_comparison.json", "w") as f:
        json.dump({"best": best_name, "selection_metric": "val_pr_auc",
                   "tuning": "bayesian" if tcfg["enabled"] else "defaults",
                   "results": results}, f, indent=2)
    # Per-fold CV scores of each model's best config -> used by diagnostics.py
    # for the Friedman + Wilcoxon-Holm statistical comparison.
    with open(outputs_dir / "cv_fold_scores.json", "w") as f:
        json.dump({"scoring": tcfg["scoring"], "cv": tcfg["cv"],
                   "fold_scores": cv_fold_scores}, f, indent=2)

    print(f"\nSaved calibrated model + threshold ({threshold:.3f}) + schema.")
    print("Run `python -m src.models.diagnostics` for the assumption checks "
          "+ statistical model comparison,")
    print("then `python -m src.models.evaluate` for the honest TEST-set report.")


if __name__ == "__main__":
    main()
