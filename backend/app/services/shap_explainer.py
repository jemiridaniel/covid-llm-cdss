"""
SHAP PermutationExplainer for COVID-19 Severity Classifier.
Uses PermutationExplainer (compatible with multiclass GBC).
Returns top N features with SHAP values for the predicted class.
"""
import os
import numpy as np
import joblib
import shap

MODEL_PATH = os.path.join(os.path.dirname(__file__), "../models/covid_classifier.pkl")
BG_PATH    = os.path.join(os.path.dirname(__file__), "../models/shap_background.npy")

FEATURE_LABELS = {
    "AGE":             "Age",
    "SEX":             "Sex",
    "PREGNANT":        "Pregnancy",
    "DIABETES":        "Diabetes",
    "COPD":            "COPD",
    "ASTHMA":          "Asthma",
    "INMSUPR":         "Immunosuppression",
    "HIPERTENSION":    "Hypertension",
    "CARDIOVASCULAR":  "Cardiovascular disease",
    "OBESITY":         "Obesity",
    "RENAL_CHRONIC":   "Chronic renal disease",
    "TOBACCO":         "Tobacco use",
    "PNEUMONIA":       "Pneumonia",
    "PATIENT_TYPE":    "Clinical setting",
    "comorbidity_count": "Comorbidity count",
    "age_risk":        "Age risk (>60)",
    "high_risk":       "High-risk composite",
}

FEATURE_ORDER = list(FEATURE_LABELS.keys())


class SHAPExplainer:
    def __init__(self):
        model = joblib.load(MODEL_PATH)
        # Load background dataset saved during training
        if os.path.exists(BG_PATH):
            background = np.load(BG_PATH)
        else:
            # Fallback: synthetic background with median-like values
            background = np.tile(
                [40, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 0, 0, 0],
                (50, 1)
            ).astype(np.float32)

        self.explainer = shap.PermutationExplainer(model.predict_proba, background)
        self.feature_names = FEATURE_ORDER
        self.feature_labels = FEATURE_LABELS

    def get_top_features(self, X: np.ndarray, predicted_class_idx: int, n: int = 5) -> list:
        """
        Return top N features sorted by absolute SHAP value for the predicted class.
        Returns list of dicts: [{"feature": str, "value": float, "shap_value": float}]
        """
        # max_evals=35 is the minimum for 17 features (2*17+1)
        shap_vals = self.explainer(X, max_evals=2 * len(self.feature_names) + 1)

        # shap_vals.values shape: (n_samples, n_features, n_classes)
        class_shap = shap_vals.values[0, :, predicted_class_idx]
        feature_vals = X[0]

        top_indices = np.argsort(np.abs(class_shap))[::-1][:n]

        result = []
        for idx in top_indices:
            name = self.feature_names[idx]
            result.append({
                "feature": self.feature_labels.get(name, name),
                "value": float(feature_vals[idx]),
                "shap_value": float(class_shap[idx]),
            })
        return result


# Singleton — loaded once at startup
_explainer_instance = None


def get_explainer() -> SHAPExplainer:
    global _explainer_instance
    if _explainer_instance is None:
        _explainer_instance = SHAPExplainer()
    return _explainer_instance
