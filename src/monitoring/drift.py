"""
drift.py  --  STEP 8: a simple data-drift monitor.

Once a model is live, the incoming data slowly changes (new patient mix, new
coding practices). If it drifts too far from what the model was trained on, the
predictions get unreliable. Production teams MONITOR for this. Here we implement
the standard metric, the Population Stability Index (PSI), which compares the
distribution of each feature between a REFERENCE set (training data) and a NEW
set (e.g. last month's patients).

PSI interpretation (industry rule of thumb):
    < 0.10  -> no significant change
    0.10-0.25 -> moderate drift, keep an eye on it
    > 0.25  -> significant drift, consider retraining

As a self-check this script compares the train vs test split (which should show
almost no drift). In production you would feed it live data instead.

    python -m src.monitoring.drift
"""

import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT
from src.models.pipeline import load_model_input, split_data


def psi_numeric(reference: pd.Series, new: pd.Series, bins: int = 10) -> float:
    """PSI for a numeric feature, using deciles of the reference distribution."""
    # Build bin edges from the reference quantiles (unique to avoid empty bins).
    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(reference, quantiles))
    if len(edges) < 3:                       # too few distinct values to bin
        return 0.0
    ref_counts, _ = np.histogram(reference, bins=edges)
    new_counts, _ = np.histogram(new, bins=edges)
    return _psi_from_counts(ref_counts, new_counts)


def psi_categorical(reference: pd.Series, new: pd.Series) -> float:
    """PSI for a categorical feature, one bin per category."""
    categories = reference.unique()
    ref_counts = np.array([(reference == c).sum() for c in categories])
    new_counts = np.array([(new == c).sum() for c in categories])
    return _psi_from_counts(ref_counts, new_counts)


def _psi_from_counts(ref_counts, new_counts) -> float:
    """Core PSI formula from two count vectors."""
    # Convert counts to proportions; add a tiny epsilon to avoid log(0)/div-by-0.
    eps = 1e-6
    ref_pct = ref_counts / ref_counts.sum() + eps
    new_pct = new_counts / new_counts.sum() + eps
    return float(np.sum((new_pct - ref_pct) * np.log(new_pct / ref_pct)))


def compute_drift(reference: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Compute PSI for every feature and label the severity."""
    records = []
    for col in reference.columns:
        if reference[col].dtype == object:
            psi = psi_categorical(reference[col], new[col])
        else:
            psi = psi_numeric(reference[col], new[col])
        severity = ("none" if psi < 0.10 else
                    "moderate" if psi < 0.25 else "significant")
        records.append({"feature": col, "psi": round(psi, 4), "drift": severity})
    return pd.DataFrame(records).sort_values("psi", ascending=False)


def main():
    df = load_model_input()
    X_train, X_test, _, _ = split_data(df)

    print("Comparing TRAIN (reference) vs TEST (new) -- expect little drift:\n")
    report = compute_drift(X_train, X_test)
    print(report.to_string(index=False))

    flagged = report[report["drift"] != "none"]
    print(f"\n{len(flagged)} feature(s) with moderate/significant drift.")

    out = PROJECT_ROOT / "outputs" / "drift_report.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(out, index=False)
    print(f"Saved -> {out.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
