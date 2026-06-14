"""
download.py  --  STEP 1 of the pipeline: fetch the raw public dataset.

What it does, in plain English:
  1. Downloads the dataset ZIP from the UCI Machine Learning Repository.
  2. Unzips it (the zip contains a nested folder with two CSV files).
  3. Copies the two CSVs into data/raw/.
  4. Prints a short, honest audit of the main file (shape, target balance,
     missing-value markers) so we KNOW what we're dealing with before modelling.

Run it from the repo root with:
    python -m src.data.download
('-m' runs it as a module so the `from src...` imports resolve correctly.)
"""

import io                         # lets us treat downloaded bytes like a file
import zipfile                    # read .zip archives
from pathlib import Path

import pandas as pd
import requests                   # downloads files over HTTP

from src.config import load_config, PROJECT_ROOT


def download_and_extract() -> Path:
    """Download the zip and extract the two CSVs into data/raw/. Returns raw dir."""
    cfg = load_config()
    url = cfg["data"]["source_url"]
    raw_dir = PROJECT_ROOT / cfg["data"]["raw_dir"]
    raw_dir.mkdir(parents=True, exist_ok=True)   # make data/raw/ if missing

    print(f"Downloading dataset from:\n  {url}")
    # stream=False -> download the whole file into memory (it's only a few MB).
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()                       # crash loudly if the URL failed
    print(f"  downloaded {len(resp.content) / 1_000_000:.1f} MB")

    # The downloaded bytes ARE a zip file. zipfile can read it straight from
    # memory if we wrap the bytes in io.BytesIO (a file-like object).
    with zipfile.ZipFile(io.BytesIO(resp.content)) as outer:
        names = outer.namelist()
        print(f"  zip contains {len(names)} entries")

        # This particular zip has a NESTED zip inside (dataset_diabetes.zip ->
        # dataset_diabetes/ folder OR an inner zip, depending on mirror). We
        # handle both: find every .csv anywhere in the archive and extract it.
        wanted = {cfg["data"]["main_file"], cfg["data"]["mapping_file"]}
        found = {}

        for name in names:
            # An entry may itself be a zip we must open, or a csv we want.
            if name.lower().endswith(".zip"):
                # nested zip -> open it and look inside for our csvs
                with zipfile.ZipFile(io.BytesIO(outer.read(name))) as inner:
                    for inner_name in inner.namelist():
                        base = Path(inner_name).name      # strip folders
                        if base in wanted:
                            found[base] = inner.read(inner_name)
            else:
                base = Path(name).name
                if base in wanted:
                    found[base] = outer.read(name)

        # Safety check: did we actually get both files we expected?
        missing = wanted - set(found)
        if missing:
            raise RuntimeError(
                f"Could not find {missing} inside the zip. "
                f"Archive entries were: {names}"
            )

        # Write each found CSV to data/raw/
        for base, content in found.items():
            out_path = raw_dir / base
            out_path.write_bytes(content)
            print(f"  saved -> {out_path.relative_to(PROJECT_ROOT)} "
                  f"({len(content) / 1_000_000:.1f} MB)")

    return raw_dir


def audit(raw_dir: Path) -> None:
    """Print an honest first look at the main file -- like a data analyst would."""
    cfg = load_config()
    main_path = raw_dir / cfg["data"]["main_file"]

    # This dataset uses the literal string "?" to mean "missing". We tell pandas
    # to treat "?" as NaN (missing) so our missing-value counts are correct.
    df = pd.read_csv(main_path, na_values="?")

    print("\n================= DATA AUDIT =================")
    print(f"Rows (hospital encounters): {len(df):,}")
    print(f"Columns:                    {df.shape[1]}")

    # ---- Target balance: how rare is "readmitted within 30 days"? ----
    tcol = cfg["target"]["raw_column"]
    print(f"\nTarget column '{tcol}' value counts:")
    counts = df[tcol].value_counts(dropna=False)
    for label, n in counts.items():
        print(f"  {str(label):>6} : {n:>7,}  ({n / len(df) * 100:5.1f}%)")
    pos = cfg["target"]["positive_label"]
    print(f"-> If we predict '{pos}' (readmit <30 days) as the positive class, "
          f"it is the MINORITY -> this is an IMBALANCED problem.")

    # ---- Columns with the most missing values ----
    miss = df.isna().sum()
    miss = miss[miss > 0].sort_values(ascending=False)
    print(f"\nColumns with missing values ({len(miss)} of {df.shape[1]}):")
    for col, n in miss.items():
        print(f"  {col:24} {n:>7,}  ({n / len(df) * 100:5.1f}%)")

    print("=============================================\n")


if __name__ == "__main__":
    raw = download_and_extract()
    audit(raw)
