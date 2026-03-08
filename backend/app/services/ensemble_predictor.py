"""
Ensemble predictor combining Mexican comorbidity model + Einstein lab model.
Falls back to Mexican model only when no lab values are provided.
"""
import os
import numpy as np
import joblib
from typing import Optional, Dict

MODEL_DIR = os.path.join(os.path.dirname(__file__), "../models")

MEXICAN_CLASSES = ['Critical', 'Mild', 'Moderate', 'No_COVID', 'Severe']
# Indices:           0           1      2            3           4


class EnsemblePredictor:
    def __init__(self):
        self.mexican_model   = joblib.load(os.path.join(MODEL_DIR, "covid_classifier.pkl"))
        self.mexican_encoder = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))

        einstein_clf_path      = os.path.join(MODEL_DIR, "einstein_classifier.pkl")
        einstein_features_path = os.path.join(MODEL_DIR, "einstein_features.pkl")
        einstein_encoder_path  = os.path.join(MODEL_DIR, "einstein_label_encoder.pkl")

        self.einstein_available = all(
            os.path.exists(p) for p in [
                einstein_clf_path, einstein_features_path, einstein_encoder_path
            ]
        )

        if self.einstein_available:
            self.einstein_model    = joblib.load(einstein_clf_path)
            self.einstein_features = joblib.load(einstein_features_path)
            self.einstein_encoder  = joblib.load(einstein_encoder_path)

        # Mexican class indices (sorted alphabetically by LabelEncoder)
        self.mexican_classes = list(self.mexican_encoder.classes_)
        self.idx_critical = self.mexican_classes.index('Critical')
        self.idx_mild     = self.mexican_classes.index('Mild')
        self.idx_moderate = self.mexican_classes.index('Moderate')
        self.idx_no_covid = self.mexican_classes.index('No_COVID')
        self.idx_severe   = self.mexican_classes.index('Severe')

    def predict(self, mexican_X, lab_values=None):
        """
        mexican_X: shape (1, 17) — always required
        lab_values: dict of {alias: value} — optional

        Returns: (severity: str, confidence: float, model_used: str, final_proba: ndarray)
        """
        mexican_proba = self.mexican_model.predict_proba(mexican_X)[0]  # shape (5,)

        if lab_values and self.einstein_available:
            einstein_X = self._build_einstein_input(lab_values)
            if einstein_X is not None:
                einstein_proba = self.einstein_model.predict_proba(einstein_X)[0]  # shape (3,)
                mapped = self._map_einstein_to_mexican(einstein_proba)             # shape (5,)
                final_proba = 0.6 * mexican_proba + 0.4 * mapped
                model_used = "ensemble"
            else:
                final_proba = mexican_proba
                model_used = "mexican_only"
        else:
            final_proba = mexican_proba
            model_used = "mexican_only"

        pred_idx = int(np.argmax(final_proba))
        severity   = self.mexican_classes[pred_idx]
        confidence = float(final_proba[pred_idx])

        return severity, confidence, model_used, final_proba

    def _build_einstein_input(self, lab_values):
        """Build Einstein feature vector from lab values dict using short aliases."""
        if not self.einstein_available:
            return None

        row = []
        for feat in self.einstein_features:
            val = lab_values.get(feat)
            if val is None or val == '':
                row.append(np.nan)
            else:
                try:
                    row.append(float(val))
                except (ValueError, TypeError):
                    row.append(np.nan)

        arr = np.array([row], dtype=np.float32)
        # Fill NaN with 0 (median was applied during training; 0 is neutral)
        arr = np.nan_to_num(arr, nan=0.0)
        return arr

    def _map_einstein_to_mexican(self, einstein_proba):
        """Map 3-class Einstein probabilities to 5-class Mexican probability space."""
        einstein_classes = list(self.einstein_encoder.classes_)

        no_covid_p   = 0.0
        non_severe_p = 0.0
        severe_p     = 0.0

        for i, cls in enumerate(einstein_classes):
            if cls == 'No_COVID':
                no_covid_p = einstein_proba[i]
            elif cls == 'Non_Severe':
                non_severe_p = einstein_proba[i]
            elif cls == 'Severe':
                severe_p = einstein_proba[i]

        mapped = np.zeros(5, dtype=np.float32)
        mapped[self.idx_no_covid]  = no_covid_p
        mapped[self.idx_mild]      = non_severe_p * 0.5
        mapped[self.idx_moderate]  = non_severe_p * 0.5
        mapped[self.idx_severe]    = severe_p * 0.5
        mapped[self.idx_critical]  = severe_p * 0.5

        return mapped


_predictor_instance = None


def get_predictor():
    """Return a singleton EnsemblePredictor instance."""
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = EnsemblePredictor()
    return _predictor_instance
