"""
train.py  --  STEP 3: train several models and track every run with MLflow.

What "experiment tracking" means: instead of running models and losing the
numbers, we log each run to MLflow -- its parameters, its metrics, and the
trained model file. Later we (or a teammate) can open the MLflow UI and compare
every run side by side. This is standard practice on real data-science teams and
is exactly the production skill this project is meant to demonstrate.

For each candidate model we:
  1. Build a full Pipeline = preprocessing + classifier (so encoding/scaling are
     re-fit on each training fold -> no leakage).
  2. Estimate honest performance with 5-fold cross-validation on the TRAIN set.
  3. Re-fit on all of TRAIN and evaluate once on the held-out TEST set.
  4. Log params + metrics + the model to MLflow.

Imbalance handling: the positive class is ~9%. Rather than SMOTE (used in the
thesis), here we use cost-sensitive learning -- class_weight="balanced" for the
linear/forest models and scale_pos_weight for the boosters. It is lighter to
serve in production and avoids synthesising fake patients.

    python -m src.models.train
"""

import json

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.config import load_config, PROJECT_ROOT
from src.models.pipeline import build_preprocessor, load_model_input, split_data


def get_candidate_models(scale_pos_weight: float) -> dict:
    """Return the dict of {name: classifier}. All are imbalance-aware."""
    return {
        "logistic_regression": LogisticRegression(
            max_iter=1000, class_weight="balanced"
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300, max_depth=12, n_jobs=1,
            class_weight="balanced", random_state=42
        ),
        "xgboost": XGBClassifier(
            n_estimators=400, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss", n_jobs=1, random_state=42
        ),
        "lightgbm": LGBMClassifier(
            n_estimators=400, max_depth=-1, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            n_jobs=1, random_state=42, verbose=-1
        ),
    }


def evaluate_on_test(pipe, X_test, y_test) -> dict:
    """Compute a suite of metrics on the held-out test set."""
    # predict_proba gives P(readmit). [:, 1] = probability of the positive class.
    proba = pipe.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)   # default-threshold labels (tuned later)
    return {
        # Threshold-INDEPENDENT (the fair headline metrics for imbalance):
        "test_roc_auc": roc_auc_score(y_test, proba),
        "test_pr_auc": average_precision_score(y_test, proba),  # area under PR curve
        "test_brier": brier_score_loss(y_test, proba),          # calibration (lower=better)
        # Threshold-dependent (at 0.5, before tuning):
        "test_f1": f1_score(y_test, pred),
        "test_recall": recall_score(y_test, pred),
        "test_precision": precision_score(y_test, pred, zero_division=0),
    }


def main():
    cfg = load_config()

    # --- point MLflow at our local tracking store + experiment ---
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    # --- load data + split ---
    df = load_model_input()
    X_train, X_test, y_train, y_test = split_data(df)
    print(f"Train: {len(X_train):,} rows | Test: {len(X_test):,} rows | "
          f"positive rate train={y_train.mean():.3f} test={y_test.mean():.3f}")

    # scale_pos_weight = (#negatives / #positives) tells the boosters to pay
    # proportionally more attention to the rare positive class.
    n_pos = int(y_train.sum())
    n_neg = int(len(y_train) - n_pos)
    spw = n_neg / n_pos
    print(f"scale_pos_weight (neg/pos) = {spw:.2f}")

    preprocessor = build_preprocessor(X_train)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    results = {}
    best_name, best_pipe, best_score = None, None, -1.0

    for name, clf in get_candidate_models(spw).items():
        print(f"\n=== {name} ===")
        pipe = Pipeline([("prep", preprocessor), ("clf", clf)])

        # 5-fold CV on TRAIN -> honest, leakage-free performance estimate.
        # Parallelism rule that avoids thread oversubscription: run the 5 FOLDS
        # in parallel (n_jobs=-1 here) while each MODEL stays single-threaded
        # (n_jobs=1, set on the estimators above). The opposite combination
        # makes LightGBM stall.
        cv_auc = cross_val_score(pipe, X_train, y_train, cv=cv,
                                 scoring="roc_auc", n_jobs=-1)
        print(f"  CV ROC-AUC: {cv_auc.mean():.4f} +/- {cv_auc.std():.4f}")

        # Fit on all of train, evaluate once on test.
        pipe.fit(X_train, y_train)
        metrics = evaluate_on_test(pipe, X_test, y_test)
        metrics["cv_roc_auc_mean"] = float(cv_auc.mean())
        metrics["cv_roc_auc_std"] = float(cv_auc.std())
        for k, v in metrics.items():
            print(f"  {k:18}: {v:.4f}")

        # --- log this run's params + metrics to MLflow ---
        # (We log the model ARTIFACT only once, for the winner, below -- saving
        #  the heavy serialization on the models we won't ship.)
        with mlflow.start_run(run_name=name):
            mlflow.log_param("model", name)
            mlflow.log_params(clf.get_params())
            mlflow.log_metrics(metrics)

        results[name] = metrics
        # Pick the best model by test ROC-AUC.
        if metrics["test_roc_auc"] > best_score:
            best_name, best_pipe, best_score = name, pipe, metrics["test_roc_auc"]

    # --- log + REGISTER the winning model in MLflow's model registry ---
    # The model registry is how teams promote a chosen model to "production".
    with mlflow.start_run(run_name=f"best_{best_name}"):
        mlflow.log_param("model", best_name)
        mlflow.log_metrics(results[best_name])
        mlflow.sklearn.log_model(
            best_pipe, name="model",
            registered_model_name=cfg["model"]["registered_name"],
        )

    # --- also save the winning pipeline + a feature schema for the API/app ---
    out_dir = PROJECT_ROOT / cfg["model"]["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipe, out_dir / "model.joblib")

    # Save the input schema (column -> dtype, plus categories) so the API and the
    # Streamlit form know exactly what inputs the model expects.
    schema = {"numeric": [], "categorical": {}}
    for col in X_train.columns:
        if X_train[col].dtype == object:
            # astype(str) guards against a column that mixes text and numbers
            # when read back from CSV (sorted() can't compare str vs float).
            schema["categorical"][col] = sorted(X_train[col].astype(str).unique())
        else:
            schema["numeric"].append(col)
    with open(out_dir / "feature_schema.json", "w") as f:
        json.dump(schema, f, indent=2)

    # Save the full model-comparison table for the README and the app.
    outputs_dir = PROJECT_ROOT / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    with open(outputs_dir / "model_comparison.json", "w") as f:
        json.dump({"best": best_name, "results": results}, f, indent=2)

    print(f"\nBEST MODEL: {best_name}  (test ROC-AUC = {best_score:.4f})")
    print(f"Saved -> {(out_dir / 'model.joblib').relative_to(PROJECT_ROOT)}")
    print(f"Saved -> {(out_dir / 'feature_schema.json').relative_to(PROJECT_ROOT)}")
    print(f"\nView all runs with:  mlflow ui --backend-store-uri "
          f"{cfg['mlflow']['tracking_uri']}")


if __name__ == "__main__":
    main()
