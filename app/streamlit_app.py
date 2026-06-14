"""
streamlit_app.py  --  STEP 7: an interactive web app around the model.

This is the "front door" of the project: a recruiter can open the live URL, enter
a patient's details, and instantly see the predicted 30-day readmission risk --
no code, no setup. It also shows how the model was evaluated, so the demo is
honest about what the model can and cannot do.

Run locally:
    streamlit run app/streamlit_app.py
"""

import json
from pathlib import Path

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ---- locate the repo root (this file is in <root>/app/) --------------------
ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
OUT_DIR = ROOT / "outputs"


@st.cache_resource          # load the model once and reuse across interactions
def load_artifacts():
    model = joblib.load(MODEL_DIR / "model.joblib")
    threshold = json.loads((MODEL_DIR / "threshold.json").read_text())["threshold"]
    schema = json.loads((MODEL_DIR / "feature_schema.json").read_text())
    return model, threshold, schema


# A full default patient record -- the form below overrides the key fields and
# the rest keep these sensible defaults (the model needs all of them).
DEFAULT_PATIENT = {
    "race": "Caucasian", "gender": "Female", "age": 65,
    "admission_type_id": "1", "discharge_disposition_id": "1",
    "admission_source_id": "7", "time_in_hospital": 5,
    "medical_specialty": "InternalMedicine", "num_lab_procedures": 45,
    "num_procedures": 1, "num_medications": 16, "number_outpatient": 0,
    "number_emergency": 0, "number_inpatient": 0, "number_diagnoses": 9,
    "max_glu_serum": "NotMeasured", "A1Cresult": "NotMeasured", "metformin": "No",
    "repaglinide": "No", "glimepiride": "No", "glipizide": "No",
    "glyburide": "No", "pioglitazone": "No", "rosiglitazone": "No",
    "insulin": "Steady", "change": "No", "diabetesMed": "Yes",
    "diag_1_group": "Circulatory", "diag_2_group": "Diabetes",
    "diag_3_group": "Respiratory", "total_prior_visits": 0, "num_med_changes": 0,
}

DIAG_GROUPS = ["Circulatory", "Respiratory", "Digestive", "Diabetes", "Injury",
               "Musculoskeletal", "Genitourinary", "Neoplasms", "Other", "Missing"]


def risk_gauge(prob: float, threshold: float) -> go.Figure:
    """A speedometer-style gauge showing the readmission probability."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        number={"suffix": "%"},
        title={"text": "30-day readmission risk"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "darkblue"},
            # bands anchored to the tuned threshold (probabilities are calibrated)
            "steps": [
                {"range": [0, threshold * 100], "color": "#c8e6c9"},        # low (green)
                {"range": [threshold * 100, 2 * threshold * 100], "color": "#fff9c4"},  # med
                {"range": [2 * threshold * 100, 100], "color": "#ffcdd2"},  # high (red)
            ],
            "threshold": {  # show the model's decision cut-off
                "line": {"color": "black", "width": 3},
                "value": threshold * 100,
            },
        },
    ))
    fig.update_layout(height=300, margin=dict(t=50, b=10))
    return fig


def main():
    st.set_page_config(page_title="Readmission Risk", page_icon="🏥",
                       layout="wide")
    model, threshold, schema = load_artifacts()

    st.title("🏥 Hospital Readmission Risk Predictor")
    st.markdown(
        "Predicts the probability that a **diabetic patient is readmitted within "
        "30 days** of discharge, from a real dataset of 100k+ US hospital "
        "encounters. Built to demonstrate an **end-to-end ML system**: cleaning → "
        "feature engineering → tracked training → tuned threshold → live serving."
    )

    tab_predict, tab_perf = st.tabs(["🔮 Predict", "📊 Model performance"])

    # ===================== PREDICT TAB =====================
    with tab_predict:
        st.subheader("Enter patient details")
        patient = dict(DEFAULT_PATIENT)   # start from defaults, override below

        c1, c2, c3 = st.columns(3)
        with c1:
            patient["age"] = st.slider("Age", 5, 95, 65, step=10)
            patient["time_in_hospital"] = st.slider("Days in hospital", 1, 14, 5)
            patient["num_medications"] = st.slider("Number of medications", 1, 60, 16)
            patient["number_diagnoses"] = st.slider("Number of diagnoses", 1, 16, 9)
        with c2:
            patient["number_inpatient"] = st.number_input(
                "Prior inpatient visits (last yr)", 0, 20, 0)
            patient["number_emergency"] = st.number_input(
                "Prior emergency visits (last yr)", 0, 20, 0)
            patient["number_outpatient"] = st.number_input(
                "Prior outpatient visits (last yr)", 0, 40, 0)
            patient["num_med_changes"] = st.number_input(
                "Diabetes meds changed this stay", 0, 6, 0)
        with c3:
            patient["A1Cresult"] = st.selectbox(
                "A1C result", ["NotMeasured", "Norm", ">7", ">8"])
            patient["insulin"] = st.selectbox(
                "Insulin", ["No", "Steady", "Up", "Down"], index=1)
            patient["change"] = st.selectbox("Any medication changed?", ["No", "Ch"])
            patient["diabetesMed"] = st.selectbox("On diabetes medication?",
                                                  ["Yes", "No"])
            patient["diag_1_group"] = st.selectbox("Primary diagnosis group",
                                                   DIAG_GROUPS)

        # derived feature: total prior visits = sum of the three visit counts
        patient["total_prior_visits"] = (
            patient["number_inpatient"] + patient["number_emergency"]
            + patient["number_outpatient"]
        )

        if st.button("Predict readmission risk", type="primary"):
            row = pd.DataFrame([patient])
            prob = float(model.predict_proba(row)[0, 1])
            colA, colB = st.columns([1, 1])
            with colA:
                st.plotly_chart(risk_gauge(prob, threshold),
                                use_container_width=True)
            with colB:
                band = ("Low" if prob < threshold else
                        "Medium" if prob < 2 * threshold else "High")
                st.metric("Predicted probability", f"{prob*100:.1f}%")
                st.metric("Risk band", band)
                if prob >= threshold:
                    st.error(f"⚠️ Flagged for follow-up "
                             f"(above the {threshold*100:.0f}% decision threshold).")
                else:
                    st.success(f"✓ Below the {threshold*100:.0f}% decision "
                               f"threshold.")
            st.caption("The black line on the gauge is the tuned decision "
                       "threshold (chosen to maximise F1 on held-out data).")

    # ===================== PERFORMANCE TAB =====================
    with tab_perf:
        st.subheader("How good is the model — honestly?")
        st.markdown(
            "Readmission is genuinely hard to predict; published results on this "
            "dataset sit around **0.64–0.68 ROC-AUC**, and this model is in that "
            "range. The value is in *ranking* who is most at risk, not in perfect "
            "prediction."
        )

        # --- the honest, unbiased headline: the held-out TEST set ---
        test_path = OUT_DIR / "test_metrics.json"
        if test_path.exists():
            tm = json.loads(test_path.read_text())
            roc_ci, pr_ci = tm.get("roc_ci"), tm.get("pr_ci")
            c1, c2 = st.columns(2)
            c1.metric("Held-out test ROC-AUC", f"{tm['roc_auc']:.3f}",
                      help=(f"95% CI {roc_ci[0]:.3f}–{roc_ci[1]:.3f}"
                            if roc_ci else None))
            c2.metric("Held-out test PR-AUC", f"{tm['pr_auc']:.3f}",
                      help=(f"95% CI {pr_ci[0]:.3f}–{pr_ci[1]:.3f}; "
                            f"base rate {tm.get('base_rate', 0):.3f}"
                            if pr_ci else None))

        # --- model comparison: tuned on train, SELECTED on validation ---
        st.markdown("**Model comparison** — each model is Bayesian-tuned on the "
                    "training set; the winner is picked on a separate validation "
                    "set, then judged once on the test set above.")
        comp_path = OUT_DIR / "model_comparison.json"
        if comp_path.exists():
            comp = json.loads(comp_path.read_text())
            rows = []
            for name, m in comp["results"].items():
                rows.append({
                    "Model": name,
                    "Validation ROC-AUC": round(m["val_roc_auc"], 3),
                    "Validation PR-AUC": round(m["val_pr_auc"], 3),
                })
            df = pd.DataFrame(rows).sort_values("Validation PR-AUC", ascending=False)
            st.dataframe(df, hide_index=True, use_container_width=True)
            sel = {"val_pr_auc": "validation PR-AUC",
                   "val_roc_auc": "validation ROC-AUC"}.get(
                       comp.get("selection_metric"), "validation score")
            st.caption(f"Best model: **{comp['best']}** (selected by {sel}). The "
                       "models are statistically close — see the README for the "
                       "Friedman + Wilcoxon-Holm comparison.")

        # show the evaluation plots if they were generated
        for img, cap in [
            ("roc_pr_curves.png", "ROC & Precision-Recall curves"),
            ("confusion_matrix.png", "Confusion matrix at the tuned threshold"),
            ("calibration_curve.png", "Calibration: are the probabilities trustworthy?"),
        ]:
            p = OUT_DIR / img
            if p.exists():
                st.image(str(p), caption=cap, width=520)


if __name__ == "__main__":
    main()
