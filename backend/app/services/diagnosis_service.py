"""
COVID-19 ML inference service.
Uses EnsemblePredictor (Mexican comorbidity model + optional Einstein lab model).
Converts PatientInput → feature vector → prediction + SHAP.
"""
import os
import numpy as np
import joblib
from app.services.shap_explainer import get_explainer
from app.services.ensemble_predictor import get_predictor
from app.models.schemas import PatientInput, SHAPFeature

TREATMENT_MAP = {
    "No_COVID": (
        "No COVID-19 detected. Monitor for 48hrs, maintain hygiene, "
        "retest if symptoms develop or worsen."
    ),
    "Mild": (
        "Home isolation 10 days. Rest, hydrate, paracetamol for fever. "
        "Monitor O2 saturation. Seek care if breathing worsens."
    ),
    "Moderate": (
        "Seek medical attention within 24hrs. Antiviral eligibility "
        "assessment recommended. Monitor O2 saturation closely."
    ),
    "Severe": (
        "Go to emergency department immediately. Supplemental oxygen "
        "required. Hospital admission recommended."
    ),
    "Critical": (
        "Call emergency services NOW. ICU admission required. "
        "Do not drive yourself to hospital."
    ),
}

COMORBIDITY_KEYS = [
    "DIABETES", "COPD", "ASTHMA", "INMSUPR",
    "HIPERTENSION", "CARDIOVASCULAR", "OBESITY", "RENAL_CHRONIC", "TOBACCO",
]


def _encode_patient(inp: PatientInput) -> np.ndarray:
    """Convert PatientInput (bool/str) into the 17-feature array the model expects."""
    sex = 1 if inp.sex.lower() == "male" else 2
    pregnant = 1 if inp.pregnant else 2
    patient_type = 2 if inp.hospitalized else 1   # 2=hospitalized, 1=outpatient
    pneumonia = 1 if inp.pneumonia else 2

    def yn(val: bool) -> int:
        return 1 if val else 2

    diabetes       = yn(inp.diabetes)
    copd           = yn(inp.copd)
    asthma         = yn(inp.asthma)
    inmsupr        = yn(inp.immunocompromised)
    hipertension   = yn(inp.hypertension)
    cardiovascular = yn(inp.cardiovascular)
    obesity        = yn(inp.obesity)
    renal_chronic  = yn(inp.renal_chronic)
    tobacco        = yn(inp.tobacco)

    comorbidities = [
        diabetes, copd, asthma, inmsupr, hipertension,
        cardiovascular, obesity, renal_chronic, tobacco,
    ]
    comorbidity_count = sum(1 for c in comorbidities if c == 1)
    age_risk  = 1 if inp.age > 60 else 0
    high_risk = 1 if comorbidity_count >= 2 and age_risk == 1 else 0

    return np.array([[
        inp.age, sex, pregnant,
        diabetes, copd, asthma, inmsupr, hipertension,
        cardiovascular, obesity, renal_chronic, tobacco,
        pneumonia, patient_type,
        comorbidity_count, age_risk, high_risk,
    ]], dtype=np.float32)


class DiagnosisService:
    def __init__(self):
        self.predictor = get_predictor()
        self.shap_explainer = get_explainer()

    def predict(self, inp: PatientInput) -> dict:
        X = _encode_patient(inp)

        severity, confidence, model_mode, final_proba = self.predictor.predict(
            X, lab_values=inp.lab_values
        )

        # Determine class index in Mexican label space for SHAP
        mexican_classes = self.predictor.mexican_classes
        pred_idx = int(np.argmax(final_proba))

        shap_dicts = self.shap_explainer.get_top_features(X, pred_idx, n=5)
        shap_features = [SHAPFeature(**d) for d in shap_dicts]

        treatment = TREATMENT_MAP.get(severity, "Consult a healthcare provider immediately.")

        return {
            "severity": severity,
            "confidence": confidence,
            "shap_features": shap_features,
            "treatment_recommendation": treatment,
            "model_mode": model_mode,
        }


# Singleton
_service_instance = None


def get_service() -> DiagnosisService:
    global _service_instance
    if _service_instance is None:
        _service_instance = DiagnosisService()
    return _service_instance
