"""
main.py  --  STEP 5: serve the model as a REST API with FastAPI.

This turns the trained model into a live service. Another system (or our Streamlit
app) can POST a patient's details to /predict and get back a readmission-risk
probability in milliseconds. This is "model serving" -- the skill that separates a
notebook from a product.

The API loads three artifacts once at startup:
  * models/model.joblib        -- the trained pipeline (preprocessing + model)
  * models/threshold.json      -- the tuned decision threshold from Step 4
  * models/feature_schema.json -- which features exist (used by /schema)

Run locally with:
    uvicorn src.api.main:app --reload
Then open http://127.0.0.1:8000/docs for an interactive, auto-generated UI.
"""

import json

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.config import load_config, PROJECT_ROOT

# ---- load artifacts once, when the server starts ---------------------------
cfg = load_config()
_model_dir = PROJECT_ROOT / cfg["model"]["output_dir"]
MODEL = joblib.load(_model_dir / "model.joblib")
THRESHOLD = json.loads((_model_dir / "threshold.json").read_text())["threshold"]
SCHEMA = json.loads((_model_dir / "feature_schema.json").read_text())

app = FastAPI(
    title="Hospital Readmission Risk API",
    description="Predicts the probability that a diabetic patient is readmitted "
                "within 30 days of discharge.",
    version="1.0.0",
)


class PatientRecord(BaseModel):
    """The features the model needs for one patient encounter.

    Field(...) marks a field as required; the example values populate the
    interactive /docs form so anyone can try a realistic request.
    """
    race: str = Field("Caucasian", description="Patient race")
    gender: str = Field("Female")
    age: int = Field(65, description="Age (band midpoint, e.g. 65 for [60-70))")
    admission_type_id: str = Field("1", description="Admission type code")
    discharge_disposition_id: str = Field("1", description="Discharge disposition code")
    admission_source_id: str = Field("7", description="Admission source code")
    time_in_hospital: int = Field(5, description="Days in hospital")
    medical_specialty: str = Field("InternalMedicine")
    num_lab_procedures: int = Field(45)
    num_procedures: int = Field(1)
    num_medications: int = Field(16)
    number_outpatient: int = Field(0)
    number_emergency: int = Field(0)
    number_inpatient: int = Field(0)
    number_diagnoses: int = Field(9)
    max_glu_serum: str = Field("NotMeasured", description="NotMeasured / Norm / >200 / >300")
    A1Cresult: str = Field("NotMeasured", description="NotMeasured / Norm / >7 / >8")
    metformin: str = Field("No", description="No / Steady / Up / Down")
    repaglinide: str = Field("No")
    glimepiride: str = Field("No")
    glipizide: str = Field("No")
    glyburide: str = Field("No")
    pioglitazone: str = Field("No")
    rosiglitazone: str = Field("No")
    insulin: str = Field("Steady")
    change: str = Field("No", description="No / Ch (any medication changed)")
    diabetesMed: str = Field("Yes", description="No / Yes")
    diag_1_group: str = Field("Circulatory")
    diag_2_group: str = Field("Diabetes")
    diag_3_group: str = Field("Respiratory")
    total_prior_visits: int = Field(0, description="outpatient+emergency+inpatient")
    num_med_changes: int = Field(0, description="# diabetes meds adjusted this stay")


class Prediction(BaseModel):
    """What the API returns."""
    readmission_probability: float
    will_readmit: bool
    risk_band: str
    threshold_used: float


def _risk_band(p: float) -> str:
    """Human-friendly risk label, relative to the tuned decision threshold.

    Probabilities are calibrated (~9% base rate), so bands are anchored to the
    threshold rather than to a fixed 0.5: below the threshold = Low (not flagged),
    up to ~2x the threshold = Medium, beyond that = High.
    """
    if p < THRESHOLD:
        return "Low"
    if p < 2 * THRESHOLD:
        return "Medium"
    return "High"


@app.get("/")
def health():
    """Simple health check -- monitoring tools ping this."""
    return {"status": "ok", "model": cfg["model"]["registered_name"]}


@app.get("/schema")
def schema():
    """Expose the feature schema (numeric cols + categorical cols & values)."""
    return SCHEMA


@app.post("/predict", response_model=Prediction)
def predict(record: PatientRecord) -> Prediction:
    """Score one patient and return the readmission risk."""
    # Pydantic validated the input; turn it into a 1-row DataFrame the pipeline
    # understands (the pipeline does all encoding/scaling internally).
    row = pd.DataFrame([record.model_dump()])
    proba = float(MODEL.predict_proba(row)[0, 1])
    return Prediction(
        readmission_probability=round(proba, 4),
        will_readmit=proba >= THRESHOLD,
        risk_band=_risk_band(proba),
        threshold_used=THRESHOLD,
    )
