"""
diagnostics.py  --  STEP 3.5: check the ASSUMPTIONS behind our models, and
compare the models STATISTICALLY (not just "this number is bigger").

A senior data scientist never just reports the best score; they show the chosen
methods were APPROPRIATE for the data. This script does two things:

  PART A -- Assumption checks, split by model family
    * Logistic Regression is a PARAMETRIC model: it assumes a specific functional
      form (the log-odds are a LINEAR combination of the features). That form only
      behaves well if a few assumptions hold, so we check them:
        1. Independence of observations
        2. Low multicollinearity (VIF) among numeric predictors
        3. Enough positive events per predictor (EPV) to estimate coefficients
    * Random Forest / XGBoost / LightGBM are NON-PARAMETRIC (tree-based): they make
      NO assumption about distribution, linearity, or multicollinearity, and need
      no feature scaling. We state what they DO need instead.

  PART B -- Statistical model comparison (the thesis approach)
    * FRIEDMAN test: across the cross-validation folds, are the models' scores
      different at all? (the non-parametric ANOVA for repeated measures)
    * If so, pairwise WILCOXON signed-rank tests with HOLM correction tell us
      WHICH models differ -- exactly the Friedman + Wilcoxon-Holm protocol used in
      the credit-risk thesis.

    python -m src.models.diagnostics      (run AFTER train.py)
"""

import json
import warnings

import numpy as np
import pandas as pd
from itertools import combinations
from scipy.stats import friedmanchisquare, wilcoxon
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from src.config import load_config, PROJECT_ROOT
from src.models.pipeline import ID_CODE_COLUMNS, load_model_input, split_data

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# PART A -- assumption checks
# ---------------------------------------------------------------------------
def variance_inflation_factors(X_numeric: pd.DataFrame) -> dict:
    """VIF measures multicollinearity: how well feature i can be PREDICTED from
    the other numeric features. VIF = 1 / (1 - R^2). Rule of thumb: VIF > 5 is
    moderate, > 10 is severe collinearity (a problem for a linear/logistic model
    because coefficients become unstable; harmless for trees).

    We compute it without statsmodels: regress each (standardised) feature on the
    rest with plain least squares and turn its R^2 into a VIF.
    """
    Xs = StandardScaler().fit_transform(X_numeric)
    cols = list(X_numeric.columns)
    vifs = {}
    for i, col in enumerate(cols):
        y = Xs[:, i]
        others = np.delete(Xs, i, axis=1)
        r2 = LinearRegression().fit(others, y).score(others, y)
        vifs[col] = float(1.0 / (1.0 - r2)) if r2 < 1 - 1e-12 else float("inf")
    return vifs


def assumption_checks() -> dict:
    cfg = load_config()
    tname = cfg["target"]["name"]
    df = load_model_input()
    X_train_full, _, y_train_full, _ = split_data(df)

    # numeric predictors = columns that are NOT text and NOT the ID code columns
    # (those are category codes we one-hot, not real numbers).
    numeric_cols = [c for c in X_train_full.columns
                    if X_train_full[c].dtype != object and c not in ID_CODE_COLUMNS]
    X_numeric = X_train_full[numeric_cols].astype(float)

    report = {"logistic_regression_parametric": {}, "tree_models_nonparametric": {}}

    # ---- 1) independence of observations ----
    # Each row is one hospital ENCOUNTER. We deliberately kept only the FIRST
    # encounter per patient (in cleaning), so the same patient does not appear
    # twice -> observations are independent. This is an assumption we ENGINEERED
    # rather than just hoped for.
    report["logistic_regression_parametric"]["independence"] = (
        "Satisfied by design: only the first encounter per patient is kept, so no "
        "patient contributes multiple correlated rows.")

    # ---- 2) multicollinearity (VIF) ----
    vifs = variance_inflation_factors(X_numeric)
    high = {k: round(v, 2) for k, v in vifs.items() if v > 5}
    report["logistic_regression_parametric"]["vif"] = {
        "all": {k: round(v, 2) for k, v in vifs.items()},
        "flagged_gt_5": high,
        "verdict": ("OK -- no severe multicollinearity (all VIF < 5)"
                    if not high else
                    f"{len(high)} numeric feature(s) with VIF>5: {list(high)} -- "
                    "watch the logistic coefficients (trees are unaffected)."),
    }

    # ---- 3) events per variable (EPV) ----
    # A classic stability rule for logistic regression: at least ~10 POSITIVE
    # events per estimated coefficient. We approximate the predictor count by the
    # one-hot width (numeric + sum of category levels).
    n_pos = int(y_train_full.sum())
    n_predictors = len(numeric_cols)
    for c in X_train_full.columns:
        if X_train_full[c].dtype == object or c in ID_CODE_COLUMNS:
            n_predictors += max(X_train_full[c].astype(str).nunique() - 1, 1)
    epv = n_pos / max(n_predictors, 1)
    report["logistic_regression_parametric"]["events_per_variable"] = {
        "positive_events": n_pos,
        "approx_predictors_after_onehot": int(n_predictors),
        "epv": round(epv, 1),
        "verdict": ("OK -- EPV >= 10, coefficients well-supported"
                    if epv >= 10 else
                    f"EPV={epv:.1f} < 10 -- regularisation (the tuned L1/L2 penalty) "
                    "is doing real work here; report logistic coefficients cautiously."),
    }
    report["logistic_regression_parametric"]["linearity_in_logit"] = (
        "Assumed, not guaranteed. We mitigate it two ways: (a) the regularised "
        "logistic model is only a BASELINE; (b) the selected production model is "
        "tree-based, which captures non-linearities the logit form cannot.")

    # ---- tree models ----
    report["tree_models_nonparametric"] = {
        "distribution": "None assumed -- trees split on thresholds, not on any "
                        "distributional form.",
        "linearity": "Not required -- trees model non-linearities and interactions "
                     "automatically.",
        "multicollinearity": "Harmless to predictions (only splits feature "
                             "IMPORTANCE among correlated features).",
        "feature_scaling": "Not required -- splits are scale-invariant (we scale "
                           "only for the logistic baseline).",
        "what_they_DO_need": "Enough samples per leaf (we tune min_samples_leaf / "
                             "min_child_samples) and a stable train/serve "
                             "distribution (checked via PSI drift monitoring).",
    }
    return report


# ---------------------------------------------------------------------------
# PART B -- statistical model comparison (Friedman + Wilcoxon-Holm)
# ---------------------------------------------------------------------------
def holm_correction(pairs_pvals: dict) -> dict:
    """Holm step-down correction for multiple comparisons. Controls the chance of
    ANY false positive across all pairwise tests (less conservative than
    Bonferroni). Returns adjusted p-values."""
    items = sorted(pairs_pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    adj, running_max = {}, 0.0
    for rank, (pair, p) in enumerate(items):
        a = min((m - rank) * p, 1.0)
        running_max = max(running_max, a)   # enforce monotonic non-decreasing
        adj[pair] = running_max
    return adj


def statistical_comparison() -> dict:
    outputs_dir = PROJECT_ROOT / "outputs"
    data = json.loads((outputs_dir / "cv_fold_scores.json").read_text())
    scoring, n_folds = data["scoring"], data["cv"]
    fold_scores = {k: v for k, v in data["fold_scores"].items()
                   if not any(np.isnan(v))}     # drop models run without CV

    names = list(fold_scores)
    means = {k: float(np.mean(v)) for k, v in fold_scores.items()}
    ranking = sorted(means, key=means.get, reverse=True)

    out = {"scoring": scoring, "cv_folds": n_folds,
           "mean_cv_score": {k: round(means[k], 4) for k in ranking},
           "ranking_best_first": ranking}

    if len(names) < 3 or n_folds < 3:
        out["note"] = ("Need >=3 models and >=3 folds for the Friedman test; "
                       "reporting the mean-score ranking only.")
        return out

    # FRIEDMAN: are the models different at all, accounting for paired folds?
    stat, p = friedmanchisquare(*[fold_scores[n] for n in names])
    out["friedman"] = {"chi2": round(float(stat), 4), "p_value": round(float(p), 4),
                       "verdict": ("models DIFFER (reject 'all equal')" if p < 0.05
                                   else "no significant difference between models")}

    # WILCOXON pairwise + Holm correction
    raw = {}
    for a, b in combinations(names, 2):
        da, db = np.array(fold_scores[a]), np.array(fold_scores[b])
        try:
            _, pw = wilcoxon(da, db)
        except ValueError:            # e.g. all differences zero
            pw = 1.0
        raw[f"{a} vs {b}"] = float(pw)
    adj = holm_correction(raw)
    out["wilcoxon_holm"] = {pair: {"p_raw": round(raw[pair], 4),
                                   "p_holm": round(adj[pair], 4),
                                   "significant": adj[pair] < 0.05}
                            for pair in raw}
    out["caveat"] = ("Only ONE dataset and {} folds -> this comparison is "
                     "ILLUSTRATIVE and low-powered. The thesis ran the same "
                     "Friedman+Wilcoxon-Holm protocol across many datasets, where "
                     "it is properly powered.").format(n_folds)
    return out


def main():
    print("=" * 70)
    print("PART A -- model ASSUMPTION checks")
    print("=" * 70)
    checks = assumption_checks()
    lr = checks["logistic_regression_parametric"]
    print("\n[Logistic Regression -- PARAMETRIC]")
    print("  Independence :", lr["independence"])
    print("  VIF          :", lr["vif"]["verdict"])
    print("  EPV          :", lr["events_per_variable"]["verdict"])
    print("  Linearity    :", lr["linearity_in_logit"])
    print("\n[Random Forest / XGBoost / LightGBM -- NON-PARAMETRIC]")
    for k, v in checks["tree_models_nonparametric"].items():
        print(f"  {k:18}: {v}")

    print("\n" + "=" * 70)
    print("PART B -- statistical model comparison (Friedman + Wilcoxon-Holm)")
    print("=" * 70)
    comp = statistical_comparison()
    print("\n  Mean CV", comp["scoring"], "per model (best first):")
    for k in comp["ranking_best_first"]:
        print(f"    {k:20} {comp['mean_cv_score'][k]:.4f}")
    if "friedman" in comp:
        print(f"\n  Friedman: chi2={comp['friedman']['chi2']}, "
              f"p={comp['friedman']['p_value']} -> {comp['friedman']['verdict']}")
        print("  Pairwise Wilcoxon (Holm-adjusted):")
        for pair, r in comp["wilcoxon_holm"].items():
            flag = "*" if r["significant"] else " "
            print(f"   {flag} {pair:42} p_holm={r['p_holm']}")
        print("\n  Caveat:", comp["caveat"])
    else:
        print("  ", comp.get("note", ""))

    out_dir = PROJECT_ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "diagnostics.json", "w") as f:
        json.dump({"assumptions": checks, "model_comparison": comp}, f, indent=2)
    print(f"\nSaved full report -> outputs/diagnostics.json")


if __name__ == "__main__":
    main()
