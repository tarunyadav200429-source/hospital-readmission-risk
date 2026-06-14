"""
preprocess.py  --  STEP 2a: clean the raw encounters into honest, modellable rows.

This module ONLY does deterministic, rule-based cleaning -- decisions that do NOT
"learn" anything from the data (so they are safe to run before the train/test
split, with zero leakage risk). Anything that learns from data (encoding maps,
scaling, imputation values, SMOTE) is deferred to the model Pipeline in Phase 3.

The cleaning steps (each is a defensible, documented decision):
  1. Drop columns that are unusable or administrative.
  2. Remove encounters that ended in death / hospice  (a dead patient cannot be
     readmitted -> keeping them would CONTAMINATE the target).
  3. Keep only each patient's FIRST encounter  (stops the same patient leaking
     across the train/test split -- methodology of Strack et al. 2014).
  4. Drop the 3 rows with invalid gender.
  5. Build the binary target  readmitted_within_30d  (1 if "<30", else 0).
  6. Convert the age band "[70-80)" into a single number (its midpoint, 75).

Run standalone for a quick check with:
    python -m src.data.preprocess
"""

import pandas as pd

from src.config import load_config, PROJECT_ROOT

# Discharge codes that mean the patient died or went to hospice. From
# IDs_mapping.csv: 11=Expired, 13=Hospice/home, 14=Hospice/facility,
# 19/20/21=Expired (hospice, Medicaid). These encounters cannot be "readmitted".
DEATH_HOSPICE_DISCHARGE_IDS = {11, 13, 14, 19, 20, 21}

# Columns we drop up front, with the reason for each:
DROP_COLUMNS = [
    "weight",        # 96.9% missing -- not enough signal to be useful
    "payer_code",    # 39.6% missing, purely administrative (who pays the bill)
    "encounter_id",  # a random row ID, carries no predictive information
    # patient_nbr is dropped LATER, after we use it to find first encounters.
]


def load_raw() -> pd.DataFrame:
    """Read the raw CSV, treating the literal '?' as a missing value."""
    cfg = load_config()
    path = PROJECT_ROOT / cfg["data"]["raw_dir"] / cfg["data"]["main_file"]
    # low_memory=False silences the mixed-type warning by reading the whole
    # column at once before deciding its type.
    return pd.read_csv(path, na_values="?", low_memory=False)


def _age_band_to_midpoint(band: str) -> int:
    """Turn an age band string like '[70-80)' into its midpoint number, 75."""
    # band looks like "[70-80)". Strip the brackets, split on "-", average.
    low, high = band.strip("[)").split("-")
    return (int(low) + int(high)) // 2


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all deterministic cleaning steps. Returns a fresh, cleaned frame."""
    cfg = load_config()
    start_rows = len(df)
    print(f"Starting rows: {start_rows:,}")

    # --- 2. remove death / hospice encounters (label contamination) ---
    df = df[~df["discharge_disposition_id"].isin(DEATH_HOSPICE_DISCHARGE_IDS)]
    print(f"  after removing death/hospice discharges: {len(df):,} "
          f"(-{start_rows - len(df):,})")

    # --- 3. keep only each patient's FIRST encounter (leakage control) ---
    # The encounter_id increases over time, so the smallest one per patient is
    # their earliest visit. Sort by it, then keep the first row per patient_nbr.
    before = len(df)
    df = df.sort_values("encounter_id").drop_duplicates(
        subset="patient_nbr", keep="first"
    )
    print(f"  after keeping first encounter per patient: {len(df):,} "
          f"(-{before - len(df):,})")

    # --- 4. drop the handful of invalid-gender rows ---
    before = len(df)
    df = df[df["gender"].isin(["Male", "Female"])]
    print(f"  after dropping invalid gender: {len(df):,} (-{before - len(df):,})")

    # --- 5. build the binary target, then drop the original 3-class column ---
    pos = cfg["target"]["positive_label"]          # "<30"
    tname = cfg["target"]["name"]                   # "readmitted_within_30d"
    # (column == "<30") gives True/False; .astype(int) makes it 1/0.
    df[tname] = (df[cfg["target"]["raw_column"]] == pos).astype(int)
    df = df.drop(columns=[cfg["target"]["raw_column"]])

    # --- 6. age band -> numeric midpoint ---
    df["age"] = df["age"].apply(_age_band_to_midpoint)

    # --- 1. drop unusable / administrative columns + the now-finished patient_nbr ---
    df = df.drop(columns=DROP_COLUMNS + ["patient_nbr"])

    # Reset the row index so it runs 0..N-1 again after all the filtering.
    df = df.reset_index(drop=True)

    pos_rate = df[tname].mean() * 100
    print(f"Final cleaned rows: {len(df):,} | columns: {df.shape[1]} | "
          f"positive (readmit<30) rate: {pos_rate:.1f}%")
    return df


if __name__ == "__main__":
    cleaned = clean(load_raw())
    print("\nColumn dtypes after cleaning:")
    print(cleaned.dtypes.value_counts())
