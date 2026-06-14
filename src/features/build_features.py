"""
build_features.py  --  STEP 2b: turn cleaned rows into model-ready features.

This is where domain knowledge shows up. The big move is grouping the raw
diagnosis codes: diag_1/2/3 hold ICD-9 codes (e.g. "428", "250.83", "V45") and
there are ~700 distinct values -- far too many to one-hot encode usefully. We
collapse them into 9 clinically meaningful groups (the standard grouping used in
the Strack et al. 2014 paper on this dataset).

We also engineer two features a clinician would care about:
  * total_prior_visits = outpatient + emergency + inpatient visits in the past year
  * num_med_changes    = how many diabetes medications were adjusted this stay

Finally we drop near-constant columns (e.g. rare drugs that are "No" for ~everyone)
because they carry no information.

Running this module is STEP 2 overall: it loads raw -> cleans -> engineers ->
saves the final modelling table to data/processed/.

    python -m src.features.build_features
"""

import pandas as pd

from src.config import load_config, PROJECT_ROOT
from src.data.preprocess import load_raw, clean

# The 23 diabetes-medication columns. Each is "No"/"Steady"/"Up"/"Down".
DRUG_COLUMNS = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
    "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
    "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide",
    "examide", "citoglipton", "insulin", "glyburide-metformin",
    "glipizide-metformin", "glimepiride-pioglitazone", "metformin-rosiglitazone",
    "metformin-pioglitazone",
]

# Columns where a missing value MEANS "the lab test was not done" -- that is
# informative, so we label it "None" rather than guessing a value.
NOT_MEASURED_COLUMNS = ["max_glu_serum", "A1Cresult"]


def map_icd9_to_group(code) -> str:
    """Map one ICD-9 diagnosis code to one of 9 clinical groups.

    ICD-9 codes are mostly numeric ranges. Codes starting with 'E' or 'V' are
    external-cause / supplementary codes -> we bucket them as 'Other'. Diabetes
    (250.xx) gets its own group because this is a diabetes cohort.
    """
    if pd.isna(code):
        return "Missing"
    code = str(code)
    # E and V codes are not in the numeric ranges -> Other.
    if code.startswith(("E", "V")):
        return "Other"
    # Diabetes codes look like 250 or 250.83.
    if code.startswith("250"):
        return "Diabetes"
    try:
        num = float(code)
    except ValueError:
        return "Other"
    # Numeric ranges from the published grouping:
    if 390 <= num <= 459 or num == 785:
        return "Circulatory"
    if 460 <= num <= 519 or num == 786:
        return "Respiratory"
    if 520 <= num <= 579 or num == 787:
        return "Digestive"
    if 800 <= num <= 999:
        return "Injury"
    if 710 <= num <= 739:
        return "Musculoskeletal"
    if 580 <= num <= 629 or num == 788:
        return "Genitourinary"
    if 140 <= num <= 239:
        return "Neoplasms"
    return "Other"


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features and tidy categoricals. Returns the model table."""
    cfg = load_config()
    tname = cfg["target"]["name"]

    # --- diagnosis grouping: 3 raw code columns -> 3 grouped columns ---
    for col in ["diag_1", "diag_2", "diag_3"]:
        df[col + "_group"] = df[col].apply(map_icd9_to_group)
    df = df.drop(columns=["diag_1", "diag_2", "diag_3"])

    # --- derived feature 1: total prior visits in the past year ---
    df["total_prior_visits"] = (
        df["number_outpatient"] + df["number_emergency"] + df["number_inpatient"]
    )

    # --- derived feature 2: how many diabetes meds were changed this stay ---
    # For each drug column, is it "Up" or "Down"? Sum those True/False across
    # all drug columns (axis=1 = sum across columns, per row).
    df["num_med_changes"] = (
        df[DRUG_COLUMNS].isin(["Up", "Down"]).sum(axis=1)
    )

    # --- lab columns: missing literally means "not measured" -> label it ---
    # Use "NotMeasured" (NOT "None"): pandas reads the string "None" back from
    # CSV as NaN, which would silently break the train/serve consistency.
    for col in NOT_MEASURED_COLUMNS:
        df[col] = df[col].fillna("NotMeasured")

    # --- any remaining categorical missing (race, medical_specialty) -> "Unknown" ---
    cat_cols = df.select_dtypes(include="object").columns
    df[cat_cols] = df[cat_cols].fillna("Unknown")

    # --- drop near-constant categorical columns (no information) ---
    # A column where one value covers >99% of rows can't help the model.
    dropped = []
    for col in df.select_dtypes(include="object").columns:
        top_share = df[col].value_counts(normalize=True).iloc[0]
        if top_share > 0.99:
            dropped.append(col)
    if dropped:
        df = df.drop(columns=dropped)
        print(f"Dropped {len(dropped)} near-constant columns "
              f"(>99% one value): {dropped}")

    # Sanity: no missing values should remain anywhere.
    assert df.isna().sum().sum() == 0, "Unexpected missing values remain!"
    print(f"Feature table: {len(df):,} rows x {df.shape[1]} columns "
          f"(incl. target '{tname}').")
    return df


def build() -> pd.DataFrame:
    """Full Step 2: raw -> clean -> engineer -> save to data/processed/."""
    cfg = load_config()
    df = clean(load_raw())
    df = engineer(df)

    out_dir = PROJECT_ROOT / cfg["data"]["processed_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "model_input.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved -> {out_path.relative_to(PROJECT_ROOT)}")
    return df


if __name__ == "__main__":
    out = build()
    print("\nEngineered feature columns:")
    print(list(out.columns))
