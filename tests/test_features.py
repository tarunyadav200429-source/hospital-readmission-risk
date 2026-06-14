"""
test_features.py  --  automated tests for the feature-engineering logic.

These run with `pytest` and check the trickiest custom code: the ICD-9 diagnosis
grouping and the age-band conversion. Automated tests like these are what let a
team change code confidently -- and the CI workflow runs them on every push.
"""

import pandas as pd

from src.data.preprocess import _age_band_to_midpoint
from src.features.build_features import map_icd9_to_group


# ---- age band -> midpoint --------------------------------------------------
def test_age_band_midpoints():
    assert _age_band_to_midpoint("[0-10)") == 5
    assert _age_band_to_midpoint("[70-80)") == 75
    assert _age_band_to_midpoint("[90-100)") == 95


# ---- ICD-9 diagnosis grouping ----------------------------------------------
def test_diabetes_codes_grouped_as_diabetes():
    assert map_icd9_to_group("250") == "Diabetes"
    assert map_icd9_to_group("250.83") == "Diabetes"


def test_numeric_ranges():
    assert map_icd9_to_group("428") == "Circulatory"     # heart failure
    assert map_icd9_to_group("491") == "Respiratory"     # bronchitis
    assert map_icd9_to_group("550") == "Digestive"       # hernia
    assert map_icd9_to_group("820") == "Injury"          # fracture
    assert map_icd9_to_group("715") == "Musculoskeletal" # osteoarthritis
    assert map_icd9_to_group("600") == "Genitourinary"   # prostate
    assert map_icd9_to_group("200") == "Neoplasms"       # lymphoma


def test_special_single_codes():
    assert map_icd9_to_group("785") == "Circulatory"
    assert map_icd9_to_group("786") == "Respiratory"


def test_e_and_v_codes_are_other():
    assert map_icd9_to_group("E909") == "Other"
    assert map_icd9_to_group("V45") == "Other"


def test_missing_code_is_missing():
    assert map_icd9_to_group(pd.NA) == "Missing"
    assert map_icd9_to_group(float("nan")) == "Missing"
