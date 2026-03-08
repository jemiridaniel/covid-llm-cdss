from pydantic import BaseModel
from typing import List, Optional, Dict


class PatientInput(BaseModel):
    age: int
    sex: str                # "male" or "female"
    pregnant: bool = False
    diabetes: bool = False
    copd: bool = False
    asthma: bool = False
    immunocompromised: bool = False
    hypertension: bool = False
    cardiovascular: bool = False
    obesity: bool = False
    renal_chronic: bool = False
    tobacco: bool = False
    pneumonia: bool = False
    hospitalized: bool = False  # True = hospitalized, False = outpatient
    patient_name: str = ""
    patient_id: str = ""
    lab_values: Optional[Dict[str, float]] = None


class SHAPFeature(BaseModel):
    feature: str
    value: float
    shap_value: float


class DiagnosisResult(BaseModel):
    model_config = {"protected_namespaces": ()}

    severity: str
    confidence: float
    shap_features: List[SHAPFeature]
    llm_explanation: str
    treatment_recommendation: str
    model_used: str
    model_mode: str = "mexican_only"
    patient_name: str = ""
    patient_id: str = ""
    timestamp: str


class ReportRequest(BaseModel):
    model_config = {"protected_namespaces": ()}

    patient_name: str = ""
    patient_id: str = ""
    timestamp: str = ""
    age: int = 0
    sex: str = ""
    pregnant: bool = False
    hospitalized: bool = False
    pneumonia: bool = False
    diabetes: bool = False
    copd: bool = False
    asthma: bool = False
    immunocompromised: bool = False
    hypertension: bool = False
    cardiovascular: bool = False
    obesity: bool = False
    renal_chronic: bool = False
    tobacco: bool = False
    severity: str = ""
    confidence: float = 0.0
    shap_features: List[SHAPFeature] = []
    llm_explanation: str = ""
    treatment_recommendation: str = ""
    model_used: str = ""
