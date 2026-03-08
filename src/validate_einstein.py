"""
Cross-Dataset Validation: Einstein Hospital Brazil (Sao Paulo)
Validates the COVID-19 severity model trained on Mexican data
against Einstein hospital data (5,644 patients, 111 features).

Dataset: Hospital Israelita Albert Einstein, Sao Paulo
"""
import os
import sys
import warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EINSTEIN_PATH = os.path.join(BASE_DIR, "data", "raw", "dataset.xlsx")
MODEL_PATH = os.path.join(BASE_DIR, "backend", "app", "models", "covid_classifier.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "backend", "app", "models", "label_encoder.pkl")

FEATURE_ORDER = [
    "AGE", "SEX", "PREGNANT", "DIABETES", "COPD", "ASTHMA",
    "INMSUPR", "HIPERTENSION", "CARDIOVASCULAR", "OBESITY",
    "RENAL_CHRONIC", "TOBACCO", "PNEUMONIA", "PATIENT_TYPE",
    "comorbidity_count", "age_risk", "high_risk",
]

COMORBIDITY_COLS = [
    "DIABETES", "COPD", "ASTHMA", "INMSUPR", "HIPERTENSION",
    "CARDIOVASCULAR", "OBESITY", "RENAL_CHRONIC", "TOBACCO",
]


def _find_col(df, keywords):
    """Find a column containing all given keywords (case-insensitive)."""
    for col in df.columns:
        col_lower = col.lower()
        if all(kw.lower() in col_lower for kw in keywords):
            return col
    return None


def assign_einstein_severity(row, icu_col, semi_col, ward_col) -> str:
    """Map Einstein hospital data to our 5-class severity system."""
    test_result = str(row.get("SARS-Cov-2 exam result", "")).strip().lower()

    if test_result != "positive":
        return "No_COVID"

    # Positive — check admission level
    icu  = pd.to_numeric(row.get(icu_col, 0),  errors="coerce") or 0
    semi = pd.to_numeric(row.get(semi_col, 0), errors="coerce") or 0
    ward = pd.to_numeric(row.get(ward_col, 0), errors="coerce") or 0

    if icu  == 1: return "Critical"
    if semi == 1: return "Severe"
    if ward == 1: return "Moderate"
    return "Mild"   # positive but not admitted anywhere


def map_einstein_to_features(df: pd.DataFrame, ward_col: str) -> pd.DataFrame:
    """Map Einstein columns to the 17 model features."""
    out = pd.DataFrame()

    # AGE: "Patient age quantile" on 0-20 scale → multiply by 4 ≈ years
    age_col = _find_col(df, ["age"])
    if age_col:
        out["AGE"] = pd.to_numeric(df[age_col], errors="coerce").fillna(5) * 4
        out["AGE"] = out["AGE"].clip(0, 100)
    else:
        out["AGE"] = 40

    # SEX — Einstein doesn't have a gender column in the public version
    out["SEX"] = 1  # default male (no sex column in Einstein v1)

    # Comorbidities — not available in Einstein, fill with 2 (No)
    for col in ["PREGNANT", "DIABETES", "COPD", "ASTHMA", "INMSUPR",
                "HIPERTENSION", "CARDIOVASCULAR", "OBESITY", "RENAL_CHRONIC", "TOBACCO"]:
        out[col] = 2  # No

    # PNEUMONIA — not directly available
    out["PNEUMONIA"] = 2  # No

    # PATIENT_TYPE — derive from any admission
    icu_col  = _find_col(df, ["intensive", "care"])
    semi_col = _find_col(df, ["semi"])
    admitted = pd.Series(0, index=df.index)
    for col in [ward_col, icu_col, semi_col]:
        if col:
            admitted += pd.to_numeric(df[col], errors="coerce").fillna(0)
    out["PATIENT_TYPE"] = admitted.apply(lambda x: 2 if x > 0 else 1)

    # Engineered features
    comorbidity_vals = out[COMORBIDITY_COLS]
    out["comorbidity_count"] = comorbidity_vals.apply(
        lambda row: sum(1 for v in row if v == 1), axis=1
    )
    out["age_risk"] = (out["AGE"] > 60).astype(int)
    out["high_risk"] = ((out["comorbidity_count"] >= 2) & (out["age_risk"] == 1)).astype(int)

    return out[FEATURE_ORDER]


def main():
    print("=" * 60)
    print("Cross-Dataset Validation: Einstein Hospital Brazil")
    print("=" * 60)

    if not os.path.exists(MODEL_PATH):
        print("\nERROR: Model not found. Run src/train_model.py first.")
        sys.exit(1)

    if not os.path.exists(EINSTEIN_PATH):
        print(f"\nERROR: Einstein dataset not found at:\n  {EINSTEIN_PATH}")
        sys.exit(1)

    # Load model
    print(f"\n[1/5] Loading model from:\n  {MODEL_PATH}")
    clf = joblib.load(MODEL_PATH)
    le  = joblib.load(ENCODER_PATH)
    print(f"  Classes: {list(le.classes_)}")

    # Load dataset
    print(f"\n[2/5] Loading Einstein dataset from:\n  {EINSTEIN_PATH}")
    df = pd.read_excel(EINSTEIN_PATH)
    print(f"  Loaded {len(df):,} rows, {df.shape[1]} columns")

    # Locate admission columns (handles typos in column names)
    ward_col = _find_col(df, ["ward"])
    semi_col = _find_col(df, ["semi"])
    icu_col  = _find_col(df, ["intensive", "care"])
    print(f"\n  Found columns:")
    print(f"    Ward:  {ward_col}")
    print(f"    Semi:  {semi_col}")
    print(f"    ICU:   {icu_col}")

    if not ward_col:
        print("  WARNING: Could not find ward admission column — using keyword fallback")
        ward_col = next((c for c in df.columns if "ward" in c.lower()), None)

    # Assign severity labels
    print("\n[3/5] Assigning severity labels...")
    df["SEVERITY"] = df.apply(
        lambda row: assign_einstein_severity(row, icu_col, semi_col, ward_col), axis=1
    )
    print("  Distribution:")
    dist = df["SEVERITY"].value_counts()
    for label, count in dist.items():
        print(f"    {label:12s}: {count:>6,} ({count/len(df)*100:.1f}%)")

    # Map features
    print("\n[4/5] Mapping Einstein features to model feature space...")
    X = map_einstein_to_features(df, ward_col).values.astype(np.float32)
    print(f"  Feature matrix shape: {X.shape}")
    print("  NOTE: Comorbidity features unavailable → filled with 2 (No)")
    print("  Model relies on AGE and PATIENT_TYPE for this dataset")

    # Keep only classes present in our model
    valid_labels = set(le.classes_)
    mask = df["SEVERITY"].isin(valid_labels)
    df_valid = df[mask].copy()
    X_valid = X[mask.values]

    # Predict
    print(f"\n[5/5] Running predictions on {len(df_valid):,} valid samples...")
    y_true = le.transform(df_valid["SEVERITY"])
    y_pred = clf.predict(X_valid)

    overall_acc = accuracy_score(y_true, y_pred) * 100
    print(f"\n  Overall accuracy: {overall_acc:.2f}%")

    print("\n  Classification Report:")
    present = sorted(set(y_true) | set(y_pred))
    present_names = [le.classes_[i] for i in present]
    print(classification_report(
        y_true, y_pred, labels=present, target_names=present_names, zero_division=0,
    ))

    print("\n  Confusion Matrix:")
    cm = confusion_matrix(y_true, y_pred, labels=present)
    print(f"  Classes: {present_names}")
    print(cm)

    print("\n  Per-class accuracy:")
    for i, cls in enumerate(le.classes_):
        mask_cls = y_true == i
        if mask_cls.sum() > 0:
            cls_acc = (y_pred[mask_cls] == i).mean() * 100
            print(f"    {cls:12s}: {cls_acc:.1f}%  (n={mask_cls.sum():,})")

    # Binary COVID vs No_COVID
    no_covid_idx = le.transform(["No_COVID"])[0]
    covid_true = (y_true != no_covid_idx).astype(int)
    covid_pred = (y_pred != no_covid_idx).astype(int)
    binary_acc = accuracy_score(covid_true, covid_pred) * 100
    print(f"\n  Binary COVID vs No_COVID accuracy: {binary_acc:.2f}%")

    print("\n" + "=" * 60)
    print("Validation complete.")
    print("=" * 60)
    print("\nInterpretation:")
    print("  • Model trained on Mexican government data, tested on Brazilian hospital data.")
    print("  • Comorbidity features unavailable → model uses AGE + PATIENT_TYPE only.")
    print("  • Binary (COVID/No_COVID) accuracy best reflects cross-country generalization.")
    print("  • Low Severe count (n=8 ICU) limits per-class evaluation.")


if __name__ == "__main__":
    main()
