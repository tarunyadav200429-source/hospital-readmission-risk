"""
evaluate.py  --  STEP 4: the HONEST test-set report.

This script touches the test set for the FIRST and ONLY time. The model and the
decision threshold were both chosen earlier on train/validation data (see
train.py), so nothing here is tuned to the test set -- the numbers are unbiased.

It reports:
  * ROC-AUC and PR-AUC with a BOOTSTRAP 95% confidence interval, so we can say
    honestly whether the model is meaningfully better than chance / other models
    rather than trusting a single point estimate.
  * a classification report at the pre-tuned threshold,
  * ROC/PR curves, a confusion matrix, and a calibration curve.

    python -m src.models.evaluate
"""

import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    classification_report,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from src.config import load_config, PROJECT_ROOT
from src.models.pipeline import load_model_input, split_data


def bootstrap_ci(y_true, proba, metric, n=1000, seed=42):
    """Bootstrap a 95% confidence interval for a metric on the test set.

    We resample the test rows WITH replacement n times, recompute the metric each
    time, and take the 2.5th / 97.5th percentiles. A wide interval = the score is
    uncertain; overlapping intervals between models = they are indistinguishable.
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    scores = []
    for _ in range(n):
        idx = rng.integers(0, len(y_true), len(y_true))   # resample indices
        if len(np.unique(y_true[idx])) < 2:               # need both classes
            continue
        scores.append(metric(y_true[idx], proba[idx]))
    lo, hi = np.percentile(scores, [2.5, 97.5])
    return float(np.mean(scores)), float(lo), float(hi)


def main():
    cfg = load_config()
    out_dir = PROJECT_ROOT / "outputs"
    model_dir = PROJECT_ROOT / cfg["model"]["output_dir"]

    model = joblib.load(model_dir / "model.joblib")
    threshold = json.loads((model_dir / "threshold.json").read_text())["threshold"]

    # rebuild the split; we only use the test set here
    df = load_model_input()
    _, X_test, _, y_test = split_data(df)
    proba = model.predict_proba(X_test)[:, 1]

    # --- headline metrics with bootstrap 95% CIs ---
    roc_m, roc_lo, roc_hi = bootstrap_ci(y_test, proba, roc_auc_score)
    pr_m, pr_lo, pr_hi = bootstrap_ci(y_test, proba, average_precision_score)
    print(f"Test ROC-AUC = {roc_auc_score(y_test, proba):.4f} "
          f"(95% CI {roc_lo:.3f}-{roc_hi:.3f})")
    print(f"Test PR-AUC  = {average_precision_score(y_test, proba):.4f} "
          f"(95% CI {pr_lo:.3f}-{pr_hi:.3f}); base rate = {y_test.mean():.3f}")

    # --- report at the PRE-TUNED threshold ---
    pred = (proba >= threshold).astype(int)
    print(f"\nDecision threshold (tuned on validation) = {threshold:.3f}")
    print("\nClassification report on the held-out test set:")
    print(classification_report(y_test, pred, digits=3,
                                target_names=["not readmitted", "readmitted<30"]))

    # persist a small JSON summary for the README/app
    with open(out_dir / "test_metrics.json", "w") as f:
        json.dump({"roc_auc": roc_m, "roc_ci": [roc_lo, roc_hi],
                   "pr_auc": pr_m, "pr_ci": [pr_lo, pr_hi],
                   "threshold": threshold, "base_rate": float(y_test.mean())},
                  f, indent=2)

    # --- plots ---------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    fpr, tpr, _ = roc_curve(y_test, proba)
    ax1.plot(fpr, tpr, label=f"ROC (AUC={roc_m:.3f})")
    ax1.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax1.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curve")
    ax1.legend()
    prec, rec, _ = precision_recall_curve(y_test, proba)
    ax2.plot(rec, prec, label=f"PR (AP={pr_m:.3f})")
    ax2.axhline(y_test.mean(), ls="--", color="k", alpha=0.4,
                label=f"baseline={y_test.mean():.3f}")
    ax2.set(xlabel="Recall", ylabel="Precision", title="Precision-Recall curve")
    ax2.legend()
    fig.tight_layout(); fig.savefig(out_dir / "roc_pr_curves.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4.5))
    ConfusionMatrixDisplay.from_predictions(
        y_test, pred, display_labels=["not", "readmit<30"],
        cmap="Blues", ax=ax, colorbar=False)
    ax.set_title(f"Confusion matrix @ threshold {threshold:.2f}")
    fig.tight_layout(); fig.savefig(out_dir / "confusion_matrix.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4.5))
    frac_pos, mean_pred = calibration_curve(y_test, proba, n_bins=10)
    ax.plot(mean_pred, frac_pos, "o-", label="model (calibrated)")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="perfect")
    ax.set(xlabel="Mean predicted probability", ylabel="Observed frequency",
           title="Calibration curve")
    ax.legend()
    fig.tight_layout(); fig.savefig(out_dir / "calibration_curve.png", dpi=120)
    plt.close(fig)

    print(f"\nSaved plots + test_metrics.json to {out_dir.relative_to(PROJECT_ROOT)}/")


if __name__ == "__main__":
    main()
