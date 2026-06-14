"""
pipeline.py  --  shared building blocks for modelling.

Two responsibilities:
  1. Load the model-input table and split it into train/test the SAME way every
     time (stratified, fixed seed) -- so every experiment is comparable.
  2. Build the preprocessing step (encode categoricals, scale numerics) as a
     scikit-learn ColumnTransformer. Crucially this is FITTED INSIDE the pipeline
     on the training fold only -> no information leaks from test to train.

Both train.py and the API reuse these functions, so preprocessing can never
drift between training and serving (a classic production bug).
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import load_config, PROJECT_ROOT

# These columns are stored as numbers but are really CATEGORY CODES (e.g.
# admission_type_id 1=Emergency, 2=Urgent). Treating them as numbers would be
# wrong (the model would think 2 > 1 means "more"), so we cast them to strings.
ID_CODE_COLUMNS = [
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id",
]


def load_model_input() -> pd.DataFrame:
    """Read data/processed/model_input.csv and fix the ID-code dtypes."""
    cfg = load_config()
    path = PROJECT_ROOT / cfg["data"]["processed_dir"] / "model_input.csv"
    # keep_default_na=False: the processed file has NO missing values (we filled
    # them with explicit labels), so we must NOT let pandas re-interpret strings
    # like "None"/"NA" as NaN. This keeps training and serving perfectly aligned.
    df = pd.read_csv(path, keep_default_na=False)
    # Cast the numeric-looking category codes to string so they are one-hot
    # encoded (as categories) rather than scaled (as numbers).
    for col in ID_CODE_COLUMNS:
        df[col] = df[col].astype(str)
    return df


def split_data(df: pd.DataFrame):
    """Stratified train/test split. Returns X_train, X_test, y_train, y_test."""
    cfg = load_config()
    tname = cfg["target"]["name"]
    X = df.drop(columns=[tname])
    y = df[tname]
    # stratify=y keeps the same 9% positive rate in BOTH train and test, which
    # matters a lot for an imbalanced problem.
    return train_test_split(
        X, y,
        test_size=cfg["split"]["test_size"],
        random_state=cfg["split"]["random_state"],
        stratify=y,
    )


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Build the encode-categoricals + scale-numerics transformer for columns in X."""
    # Categorical = text columns; numeric = everything else.
    categorical = X.select_dtypes(include="object").columns.tolist()
    numeric = X.select_dtypes(exclude="object").columns.tolist()

    return ColumnTransformer(
        transformers=[
            # One-hot encode categories. handle_unknown="ignore" means: if the
            # live API ever sees a category not seen in training, encode it as
            # all-zeros instead of crashing -- essential for production.
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
             categorical),
            # Scale numeric features to comparable ranges (helps linear models).
            ("num", StandardScaler(), numeric),
        ],
        remainder="drop",
    )
