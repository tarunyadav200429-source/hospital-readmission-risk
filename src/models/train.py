"""
train.py  --  STEP 3: select a model, calibrate it, and track everything in MLflow.

This uses a clean THREE-WAY split so nothing leaks:
    * train      -> fit the candidate models
    * validation -> choose the best model AND tune the decision threshold
    * test       -> NEVER touched here (only src/models/evaluate.py uses it)

Design decisions (and why):
  * No class reweighting. Reweighting (scale_pos_weight / class_weight) does not
    improve ranking (ROC-AUC) and it DISTORTS the predicted probabilities, which
    we display in the app. Instead we keep natural probabilities and handle the
    9% imbalance purely through the tuned decision threshold.
  * Probability calibration. We wrap the chosen model in CalibratedClassifierCV
    (isotonic) so the numbers it outputs behave like real probabilities -- a
    patient the model scores at "10%" really is readmitted ~10% of the time.
  * Threshold tuned on VALIDATION, never on test, so the reported test numbers
    are honest.

    python -m src.models.train
"""

import json

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
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.config import load_config, PROJECT_ROOT
from src.models.pipeline import build_preprocessor, load_model_input, split_data


def get_candidate_models() -> dict:
    """Candidate classifiers. Imbalance is handled by the threshold, NOT by
    reweighting, so that the predicted probabilities stay calibratable."""
    return {
        "logistic_regression": LogisticRegression(max_iter=1000),
        "random_forest": RandomForestClassifier(
            n_estimators=300, max_depth=12, n_jobs=1, random_state=42),
        "xgboost": XGBClassifier(
            n_estimators=400, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="logloss", n_jobs=1, random_state=42),
        "lightgbm": LGBMClassifier(
            n_estimators=400, max_depth=-1, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            n_jobs=1, random_state=42, verbose=-1),
    }


def best_f1_threshold(y_true, proba) -> float:
    """Threshold that maximises F1 -- tuned on VALIDATION data only."""
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    return float(thresholds[np.argmax(f1[:-1])])


def main():
    cfg = load_config()
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

    preprocessor = build_preprocessor(X_train_full)

    # --- 1) choose the best model on VALIDATION ROC-AUC ----------------------
    results = {}
    best_name, best_pipe, best_score = None, None, -1.0
    for name, clf in get_candidate_models().items():
        pipe = Pipeline([("prep", clone(preprocessor)), ("clf", clf)])
        pipe.fit(X_train, y_train)
        val_proba = pipe.predict_proba(X_val)[:, 1]
        m = {
            "val_roc_auc": roc_auc_score(y_val, val_proba),
            "val_pr_auc": average_precision_score(y_val, val_proba),
        }
        results[name] = m
        print(f"  {name:20} val ROC-AUC={m['val_roc_auc']:.4f} "
              f"PR-AUC={m['val_pr_auc']:.4f}")
        with mlflow.start_run(run_name=name):
            mlflow.log_param("model", name)
            mlflow.log_params(clf.get_params())
            mlflow.log_metrics(m)
        if m["val_roc_auc"] > best_score:
            best_name, best_pipe, best_score = name, pipe, m["val_roc_auc"]
    print(f"\nSelected (by validation ROC-AUC): {best_name} ({best_score:.4f})")

    # --- 2) calibrate the winner on TRAIN, tune threshold on VALIDATION ------
    # CalibratedClassifierCV refits the model internally with 5-fold CV and maps
    # its raw scores onto honest probabilities.
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
        mlflow.log_param("calibration", "isotonic")
        mlflow.log_param("decision_threshold", threshold)
        mlflow.log_metrics(results[best_name])
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
        json.dump({"best": best_name, "selection_metric": "val_roc_auc",
                   "results": results}, f, indent=2)

    print(f"\nSaved calibrated model + threshold ({threshold:.3f}) + schema.")
    print("Run `python -m src.models.evaluate` for the honest TEST-set report.")


if __name__ == "__main__":
    main()
