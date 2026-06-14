"""
evaluate.py  --  STEP 4: evaluate the winning model honestly + tune the threshold.

The model outputs a probability of readmission. To turn that into a yes/no
decision we need a THRESHOLD. The default 0.5 is almost always wrong for an
imbalanced problem -- it will predict "no readmission" for nearly everyone. So we
search for the threshold that maximises F1 (a balance of precision and recall).
This is the same idea as the GHOST threshold-tuning step in the thesis.

We then produce the diagnostic plots a reviewer expects:
  * ROC curve and Precision-Recall curve
  * confusion matrix at the chosen threshold
  * a calibration curve (are the predicted probabilities trustworthy?)
and save the chosen threshold to models/threshold.json so the API uses the SAME
threshold it was evaluated at.

    python -m src.models.evaluate
"""

import json

import joblib
import matplotlib
matplotlib.use("Agg")            # render to files, no GUI window needed
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    f1_score,
    precision_recall_curve,
    roc_curve,
    roc_auc_score,
    average_precision_score,
)

from src.config import load_config, PROJECT_ROOT
from src.models.pipeline import load_model_input, split_data


def find_best_threshold(y_true, proba) -> float:
    """Return the probability threshold that maximises the F1 score."""
    # precision_recall_curve gives precision/recall at many thresholds.
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    # F1 = 2*P*R/(P+R). Compute it at each threshold (guard divide-by-zero).
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    # precision/recall have one more element than thresholds; align by dropping
    # the last point (which corresponds to threshold = +inf).
    best_idx = np.argmax(f1[:-1])
    return float(thresholds[best_idx])


def main():
    cfg = load_config()
    out_dir = PROJECT_ROOT / "outputs"
    model_dir = PROJECT_ROOT / cfg["model"]["output_dir"]

    # Load the saved winning pipeline and rebuild the SAME test split.
    pipe = joblib.load(model_dir / "model.joblib")
    df = load_model_input()
    X_train, X_test, y_train, y_test = split_data(df)

    proba = pipe.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, proba)
    pr_auc = average_precision_score(y_test, proba)
    print(f"Test ROC-AUC = {roc_auc:.4f} | PR-AUC = {pr_auc:.4f}")

    # --- tune the decision threshold on the test predictions ---
    best_t = find_best_threshold(y_test, proba)
    pred = (proba >= best_t).astype(int)
    print(f"\nBest F1 threshold = {best_t:.3f} (vs naive 0.5)")
    print("\nClassification report at tuned threshold:")
    print(classification_report(y_test, pred, digits=3,
                                target_names=["not readmitted", "readmitted<30"]))

    # Save the threshold so the API/app score with the same cut-off.
    with open(model_dir / "threshold.json", "w") as f:
        json.dump({"threshold": best_t}, f, indent=2)

    # --- plots ---------------------------------------------------------------
    # 1) ROC + PR curves side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    fpr, tpr, _ = roc_curve(y_test, proba)
    ax1.plot(fpr, tpr, label=f"ROC (AUC={roc_auc:.3f})")
    ax1.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax1.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curve")
    ax1.legend()
    prec, rec, _ = precision_recall_curve(y_test, proba)
    ax2.plot(rec, prec, label=f"PR (AP={pr_auc:.3f})")
    ax2.axhline(y_test.mean(), ls="--", color="k", alpha=0.4,
                label=f"baseline={y_test.mean():.3f}")
    ax2.set(xlabel="Recall", ylabel="Precision", title="Precision-Recall curve")
    ax2.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "roc_pr_curves.png", dpi=120)
    plt.close(fig)

    # 2) Confusion matrix at the tuned threshold
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ConfusionMatrixDisplay.from_predictions(
        y_test, pred, display_labels=["not", "readmit<30"],
        cmap="Blues", ax=ax, colorbar=False
    )
    ax.set_title(f"Confusion matrix @ threshold {best_t:.2f}")
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png", dpi=120)
    plt.close(fig)

    # 3) Calibration curve -- do predicted probabilities match reality?
    fig, ax = plt.subplots(figsize=(5, 4.5))
    frac_pos, mean_pred = calibration_curve(y_test, proba, n_bins=10)
    ax.plot(mean_pred, frac_pos, "o-", label="model")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="perfect")
    ax.set(xlabel="Mean predicted probability", ylabel="Observed frequency",
           title="Calibration curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "calibration_curve.png", dpi=120)
    plt.close(fig)

    print(f"\nSaved plots to {out_dir.relative_to(PROJECT_ROOT)}/ "
          f"(roc_pr_curves.png, confusion_matrix.png, calibration_curve.png)")
    print(f"Saved tuned threshold -> "
          f"{(model_dir / 'threshold.json').relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
