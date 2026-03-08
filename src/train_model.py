"""
COVID-19 Severity Classifier — Training Script
Dataset: Mexican Government COVID-19 Open Data
Model: XGBClassifier (300 estimators, max_depth=6, lr=0.05) + SMOTE
Target: 5-class severity (No_COVID, Mild, Moderate, Severe, Critical)
"""
import os
import sys
import warnings
import numpy as np
import pandas as pd
import joblib
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "Covid Data.csv")
MODEL_DIR = os.path.join(BASE_DIR, "backend", "app", "models")
MODEL_PATH = os.path.join(MODEL_DIR, "covid_classifier.pkl")
ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")

os.makedirs(MODEL_DIR, exist_ok=True)

# ── Feature definitions ────────────────────────────────────────────────────────
FEATURES_BASE = [
    "AGE", "SEX", "PREGNANT", "DIABETES", "COPD", "ASTHMA",
    "INMSUPR", "HIPERTENSION", "CARDIOVASCULAR", "OBESITY",
    "RENAL_CHRONIC", "TOBACCO", "PNEUMONIA", "PATIENT_TYPE",
]
COMORBIDITY_COLS = [
    "DIABETES", "COPD", "ASTHMA", "INMSUPR", "HIPERTENSION",
    "CARDIOVASCULAR", "OBESITY", "RENAL_CHRONIC", "TOBACCO",
]
FEATURE_ORDER = FEATURES_BASE + ["comorbidity_count", "age_risk", "high_risk"]


# ── Severity label assignment ──────────────────────────────────────────────────
def assign_severity(row) -> str:
    """Priority-based severity assignment from dataset columns."""
    clasif = row["CLASIFFICATION_FINAL"]

    # No COVID — negative test result
    if clasif in [4, 5, 6]:
        return "No_COVID"

    # Confirmed COVID (1, 2, 3) — priority order
    # Critical: intubated OR died
    date_died = str(row["DATE_DIED"]).strip()
    if row["INTUBED"] == 1 or date_died != "9999-99-99":
        return "Critical"

    # Severe: in ICU, not intubated
    if row["ICU"] == 1:
        return "Severe"

    # Moderate: hospitalized, not in ICU
    if row["PATIENT_TYPE"] == 2:
        return "Moderate"

    # Mild: outpatient (remaining confirmed cases)
    return "Mild"


# ── Feature engineering ────────────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # comorbidity_count: count columns where value == 1 (Yes)
    df["comorbidity_count"] = df[COMORBIDITY_COLS].apply(
        lambda row: sum(1 for v in row if v == 1), axis=1
    )
    df["age_risk"] = (df["AGE"] > 60).astype(int)
    df["high_risk"] = ((df["comorbidity_count"] >= 2) & (df["age_risk"] == 1)).astype(int)
    return df


# ── Main training pipeline ─────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("COVID-19 Severity Classifier — Training")
    print("=" * 60)

    # 1. Load data
    print(f"\n[1/8] Loading dataset from:\n  {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    print(f"  Loaded {len(df):,} rows, {df.shape[1]} columns")

    # 2. Filter pending classification (7=Pending)
    print("\n[2/8] Filtering pending cases (CLASIFFICATION_FINAL == 7)...")
    df = df[df["CLASIFFICATION_FINAL"].isin([1, 2, 3, 4, 5, 6])].copy()
    print(f"  Remaining: {len(df):,} rows")

    # 3. Assign severity labels
    print("\n[3/8] Assigning severity labels...")
    df["SEVERITY"] = df.apply(assign_severity, axis=1)
    print("  Distribution:")
    dist = df["SEVERITY"].value_counts()
    for label, count in dist.items():
        print(f"    {label:12s}: {count:>8,} ({count/len(df)*100:.1f}%)")

    # 4. Replace missing encodings (97=N/A, 98=Ignored, 99=Not specified) with NaN
    print("\n[4/8] Cleaning data (97/98/99 → NaN, fill with mode)...")
    for col in FEATURES_BASE:
        if col in df.columns:
            df[col] = df[col].replace([97, 98, 99], np.nan)
    # Fill NaN with column mode
    for col in FEATURES_BASE:
        if col in df.columns and df[col].isna().any():
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)
    df["AGE"] = df["AGE"].fillna(df["AGE"].median())

    # 5. Feature engineering
    print("\n[5/8] Engineering features (comorbidity_count, age_risk, high_risk)...")
    df = engineer_features(df)

    # 6. Balanced sampling — 10,000 per class (50,000 total)
    print("\n[6/8] Balanced sampling (10,000 per class → 50,000 total)...")
    per_class = 10_000
    samples = []
    for label in df["SEVERITY"].unique():
        subset = df[df["SEVERITY"] == label]
        n = min(per_class, len(subset))
        samples.append(subset.sample(n=n, random_state=42))
        print(f"  {label:12s}: {n:,} samples")
    df_balanced = pd.concat(samples, ignore_index=True).sample(frac=1, random_state=42)
    print(f"  Total: {len(df_balanced):,} samples")

    # 7. Prepare X and y
    X = df_balanced[FEATURE_ORDER].values.astype(np.float32)
    le = LabelEncoder()
    y = le.fit_transform(df_balanced["SEVERITY"])
    print(f"\n  Classes: {list(le.classes_)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"  Train: {len(X_train):,} | Test: {len(X_test):,}")

    # 7b. Apply SMOTE to oversample minority classes (Severe) in training set
    print("\n  Applying SMOTE to balance training classes...")
    sm = SMOTE(random_state=42)
    X_train, y_train = sm.fit_resample(X_train, y_train)
    unique, counts = np.unique(y_train, return_counts=True)
    for cls_idx, cnt in zip(unique, counts):
        print(f"    {le.classes_[cls_idx]:12s}: {cnt:,} samples after SMOTE")
    print(f"  Train after SMOTE: {len(X_train):,}")

    # 8. Train XGBClassifier
    print("\n[7/8] Training XGBClassifier...")
    print("  n_estimators=300, max_depth=6, learning_rate=0.05")
    clf = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        random_state=42,
        use_label_encoder=False,
    )
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    print(f"\n  Overall accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")
    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    print("\n  Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"  Classes: {list(le.classes_)}")
    print(cm)

    # Per-class accuracy
    print("\n  Per-class accuracy:")
    for i, cls in enumerate(le.classes_):
        cls_mask = y_test == i
        if cls_mask.sum() > 0:
            cls_acc = (y_pred[cls_mask] == i).mean() * 100
            print(f"    {cls:12s}: {cls_acc:.1f}%")

    # 8b. Save background dataset for SHAP PermutationExplainer
    bg_path = os.path.join(MODEL_DIR, "shap_background.npy")
    np.save(bg_path, X_train[:100])
    print(f"  Saved SHAP background {X_train[:100].shape} to: {bg_path}")

    # 9. Save model and encoder
    print(f"\n[8/8] Saving model to:\n  {MODEL_PATH}")
    joblib.dump(clf, MODEL_PATH)
    joblib.dump(le, ENCODER_PATH)
    print(f"  Saved label encoder to:\n  {ENCODER_PATH}")

    # ── Sanity checks ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SANITY CHECKS")
    print("=" * 60)

    def make_input(age, sex, hospitalized, pneumonia,
                   diabetes=2, copd=2, asthma=2, inmsupr=2,
                   hipertension=2, cardiovascular=2, obesity=2,
                   renal_chronic=2, tobacco=2, pregnant=2) -> np.ndarray:
        """Build a feature vector. sex: 1=male, 2=female."""
        patient_type = 2 if hospitalized else 1
        comorbidities = [diabetes, copd, asthma, inmsupr, hipertension, cardiovascular, obesity, renal_chronic, tobacco]
        comorbidity_count = sum(1 for c in comorbidities if c == 1)
        age_risk = 1 if age > 60 else 0
        high_risk = 1 if comorbidity_count >= 2 and age_risk == 1 else 0
        pneumonia_val = 1 if pneumonia else 2
        return np.array([[
            age, sex, pregnant, diabetes, copd, asthma, inmsupr,
            hipertension, cardiovascular, obesity, renal_chronic, tobacco,
            pneumonia_val, patient_type,
            comorbidity_count, age_risk, high_risk
        ]], dtype=np.float32)

    cases = [
        {
            "name": "Case 1 — Mild",
            "desc": "Age 28, male, no comorbidities, outpatient, no pneumonia",
            "X": make_input(age=28, sex=1, hospitalized=False, pneumonia=False),
            "expected": ["Mild", "No_COVID"],
        },
        {
            "name": "Case 2 — Severe/Critical",
            "desc": "Age 75, male, diabetes+hypertension+obesity+cardiovascular, hospitalized, pneumonia",
            "X": make_input(
                age=75, sex=1, hospitalized=True, pneumonia=True,
                diabetes=1, hipertension=1, obesity=1, cardiovascular=1
            ),
            "expected": ["Severe", "Critical"],
        },
    ]

    # Case 3 — No_COVID: take from test set
    no_covid_idx = np.where(y_test == le.transform(["No_COVID"])[0])[0]
    if len(no_covid_idx) > 0:
        sample_idx = no_covid_idx[0]
        cases.append({
            "name": "Case 3 — No_COVID",
            "desc": "Test-set patient with No_COVID label",
            "X": X_test[sample_idx:sample_idx+1],
            "expected": ["No_COVID"],
        })

    all_passed = True
    for case in cases:
        proba = clf.predict_proba(case["X"])[0]
        pred_idx = np.argmax(proba)
        predicted = le.classes_[pred_idx]
        confidence = proba[pred_idx] * 100
        passed = predicted in case["expected"]
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False
        print(f"\n  [{status}] {case['name']}")
        print(f"    {case['desc']}")
        print(f"    Predicted: {predicted} (confidence: {confidence:.1f}%)")
        print(f"    Expected:  {case['expected']}")
        sorted_idx = np.argsort(proba)[::-1][:3]
        for i in sorted_idx:
            print(f"      {le.classes_[i]:12s}: {proba[i]*100:.1f}%")

    print("\n" + "=" * 60)
    if all_passed:
        print("All sanity checks PASSED!")
    else:
        print("WARNING: Some sanity checks FAILED — review model.")
    print("=" * 60)
    print("\nTraining complete. Models saved successfully.")
    print(f"  Model: {MODEL_PATH}")
    print(f"  Encoder: {ENCODER_PATH}")


if __name__ == "__main__":
    main()
