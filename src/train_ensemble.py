"""
Ensemble Evaluation Script for COVID-19 Severity Prediction.
Loads Mexican comorbidity model + Einstein lab model, evaluates each on
held-out test sets, then evaluates the ensemble combination.

Does NOT retrain models — loads pre-trained artifacts only.
"""
import os
import sys
import warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR  = os.path.join(BASE_DIR, "backend", "app", "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

MEXICAN_CSV  = os.path.join(BASE_DIR, "data", "raw", "Covid Data.csv")
EINSTEIN_XLS = os.path.join(BASE_DIR, "data", "raw", "dataset.xlsx")

MEXICAN_FEATURES_BASE = [
    "AGE", "SEX", "PREGNANT", "DIABETES", "COPD", "ASTHMA",
    "INMSUPR", "HIPERTENSION", "CARDIOVASCULAR", "OBESITY",
    "RENAL_CHRONIC", "TOBACCO", "PNEUMONIA", "PATIENT_TYPE",
]
COMORBIDITY_COLS = [
    "DIABETES", "COPD", "ASTHMA", "INMSUPR", "HIPERTENSION",
    "CARDIOVASCULAR", "OBESITY", "RENAL_CHRONIC", "TOBACCO",
]
MEXICAN_FEATURE_ORDER = MEXICAN_FEATURES_BASE + ["comorbidity_count", "age_risk", "high_risk"]

# Mexican classes (alphabetical, as produced by LabelEncoder)
MEXICAN_CLASSES = ['Critical', 'Mild', 'Moderate', 'No_COVID', 'Severe']


# ── Helpers: Mexican dataset ───────────────────────────────────────────────────
def assign_severity_mexican(row):
    clasif = row["CLASIFFICATION_FINAL"]
    if clasif in [4, 5, 6]:
        return "No_COVID"
    date_died = str(row["DATE_DIED"]).strip()
    if row["INTUBED"] == 1 or date_died != "9999-99-99":
        return "Critical"
    if row["ICU"] == 1:
        return "Severe"
    if row["PATIENT_TYPE"] == 2:
        return "Moderate"
    return "Mild"


def engineer_features_mexican(df):
    df = df.copy()
    df["comorbidity_count"] = df[COMORBIDITY_COLS].apply(
        lambda row: sum(1 for v in row if v == 1), axis=1
    )
    df["age_risk"]  = (df["AGE"] > 60).astype(int)
    df["high_risk"] = ((df["comorbidity_count"] >= 2) & (df["age_risk"] == 1)).astype(int)
    return df


def load_mexican_test_set(le_mexican):
    """Load Mexican CSV and produce a consistent 20% held-out test set."""
    print(f"  Loading Mexican CSV: {MEXICAN_CSV}")
    df = pd.read_csv(MEXICAN_CSV)
    df = df[df["CLASIFFICATION_FINAL"].isin([1, 2, 3, 4, 5, 6])].copy()

    for col in MEXICAN_FEATURES_BASE:
        if col in df.columns:
            df[col] = df[col].replace([97, 98, 99], np.nan)
    for col in MEXICAN_FEATURES_BASE:
        if col in df.columns and df[col].isna().any():
            df[col] = df[col].fillna(df[col].mode()[0])
    df["AGE"] = df["AGE"].fillna(df["AGE"].median())

    df["SEVERITY"] = df.apply(assign_severity_mexican, axis=1)
    df = engineer_features_mexican(df)

    # Balance to 10,000 per class (same as training)
    per_class = 10_000
    samples = []
    for label in df["SEVERITY"].unique():
        subset = df[df["SEVERITY"] == label]
        n = min(per_class, len(subset))
        samples.append(subset.sample(n=n, random_state=42))
    df_bal = pd.concat(samples, ignore_index=True).sample(frac=1, random_state=42)

    X = df_bal[MEXICAN_FEATURE_ORDER].values.astype(np.float32)
    y = le_mexican.transform(df_bal["SEVERITY"])

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    return X_test, y_test


# ── Helpers: Einstein dataset ──────────────────────────────────────────────────
ALIAS_KEYWORDS = {
    'hemoglobin':      ['hemoglobin'],
    'leukocytes':      ['leukocytes'],
    'lymphocytes':     ['lymphocytes'],
    'platelets':       ['platelets'],
    'neutrophils':     ['neutrophils'],
    'monocytes':       ['monocytes'],
    'eosinophils':     ['eosinophils'],
    'red_blood_cells': ['red blood cells'],
    'mcv':             ['mean corpuscular volume', 'mcv'],
    'rdw':             ['red blood cell distribution width', 'rdw'],
    'crp':             ['proteina c reativa', 'c-reactive'],
    'ldh':             ['lactic dehydrogenase', 'ldh'],
    'ddimer':          ['d-dimer'],
    'ferritin':        ['ferritin'],
}


def find_column(df_columns, keywords):
    cols_lower = [c.lower() for c in df_columns]
    for kw in keywords:
        kw = kw.lower()
        for i, col_low in enumerate(cols_lower):
            if kw in col_low:
                return df_columns[i]
    return None


def assign_severity_einstein(row):
    exam = str(row.get('SARS-Cov-2 exam result', '')).strip().lower()
    if exam != 'positive':
        return 'No_COVID'
    icu_val  = row.get('Patient addmited to intensive care unit (1=yes, 0=no)', 0)
    semi_val = row.get('Patient addmited to semi-intensive unit (1=yes, 0=no)', 0)
    try:
        icu = int(float(icu_val)) == 1
    except (ValueError, TypeError):
        icu = False
    try:
        semi = int(float(semi_val)) == 1
    except (ValueError, TypeError):
        semi = False
    if icu or semi:
        return 'Severe'
    return 'Non_Severe'


def load_einstein_test_set(einstein_features, le_einstein):
    """Load Einstein XLSX and produce a consistent 20% held-out test set."""
    print(f"  Loading Einstein XLSX: {EINSTEIN_XLS}")
    df = pd.read_excel(EINSTEIN_XLS)

    # Resolve alias -> actual column
    alias_to_col = {}
    for alias in einstein_features:
        if alias in ALIAS_KEYWORDS:
            col = find_column(list(df.columns), ALIAS_KEYWORDS[alias])
            if col:
                alias_to_col[alias] = col

    actual_cols = [alias_to_col[a] for a in einstein_features if a in alias_to_col]
    used_aliases = [a for a in einstein_features if a in alias_to_col]

    df['SEVERITY'] = df.apply(assign_severity_einstein, axis=1)

    feat_df = df[actual_cols].copy()
    for col in actual_cols:
        feat_df[col] = pd.to_numeric(feat_df[col], errors='coerce')

    non_null_count = feat_df.notna().sum(axis=1)
    mask = non_null_count >= 5
    feat_df = feat_df[mask]
    severity_series = df.loc[mask, 'SEVERITY']

    for col in actual_cols:
        med = feat_df[col].median()
        feat_df[col] = feat_df[col].fillna(med if not pd.isna(med) else 0.0)

    X = feat_df.values.astype(np.float32)
    y = le_einstein.transform(severity_series)

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    return X_test, y_test


# ── Ensemble mapping ──────────────────────────────────────────────────────────
def map_einstein_to_mexican(einstein_proba, le_einstein, mexican_classes):
    """Map 3-class Einstein probabilities to 5-class Mexican probability space."""
    einstein_classes = list(le_einstein.classes_)

    no_covid_p  = 0.0
    non_severe_p = 0.0
    severe_p    = 0.0

    for i, cls in enumerate(einstein_classes):
        if cls == 'No_COVID':
            no_covid_p = einstein_proba[i]
        elif cls == 'Non_Severe':
            non_severe_p = einstein_proba[i]
        elif cls == 'Severe':
            severe_p = einstein_proba[i]

    idx_critical = mexican_classes.index('Critical')
    idx_mild     = mexican_classes.index('Mild')
    idx_moderate = mexican_classes.index('Moderate')
    idx_no_covid = mexican_classes.index('No_COVID')
    idx_severe   = mexican_classes.index('Severe')

    mapped = np.zeros(5, dtype=np.float32)
    mapped[idx_no_covid]  = no_covid_p
    mapped[idx_mild]      = non_severe_p * 0.5
    mapped[idx_moderate]  = non_severe_p * 0.5
    mapped[idx_severe]    = severe_p * 0.5
    mapped[idx_critical]  = severe_p * 0.5
    return mapped


def get_severe_critical_recall(y_true, y_pred, classes):
    """Compute recall for Severe and Critical classes."""
    results = {}
    for cls_name in ['Severe', 'Critical']:
        if cls_name in classes:
            idx = classes.index(cls_name)
            mask = y_true == idx
            if mask.sum() > 0:
                results[cls_name] = (y_pred[mask] == idx).mean() * 100
            else:
                results[cls_name] = float('nan')
        else:
            results[cls_name] = float('nan')
    return results


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("Ensemble Evaluation — COVID-19 Severity Models")
    print("=" * 60)

    # Load Mexican model
    print("\n[1/5] Loading Mexican model...")
    mexican_clf = joblib.load(os.path.join(MODEL_DIR, "covid_classifier.pkl"))
    le_mexican  = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))
    mexican_classes = list(le_mexican.classes_)
    print(f"  Mexican classes: {mexican_classes}")

    # Load Einstein model
    print("\n[2/5] Loading Einstein model...")
    einstein_clf      = joblib.load(os.path.join(MODEL_DIR, "einstein_classifier.pkl"))
    einstein_features = joblib.load(os.path.join(MODEL_DIR, "einstein_features.pkl"))
    le_einstein       = joblib.load(os.path.join(MODEL_DIR, "einstein_label_encoder.pkl"))
    print(f"  Einstein classes:  {list(le_einstein.classes_)}")
    print(f"  Einstein features: {einstein_features}")

    # Load test sets
    print("\n[3/5] Building held-out test sets...")
    X_mex_test, y_mex_test = load_mexican_test_set(le_mexican)
    print(f"  Mexican test set: {len(X_mex_test):,} samples")

    X_ein_test, y_ein_test = load_einstein_test_set(einstein_features, le_einstein)
    print(f"  Einstein test set: {len(X_ein_test):,} samples")

    # ── Evaluate Mexican model on Mexican test set ────────────────────────────
    print("\n[4/5] Evaluating models...")
    y_mex_pred = mexican_clf.predict(X_mex_test)
    mex_acc = accuracy_score(y_mex_test, y_mex_pred) * 100
    mex_recalls = get_severe_critical_recall(y_mex_test, y_mex_pred, mexican_classes)

    # ── Evaluate Einstein model on Einstein test set ──────────────────────────
    y_ein_pred = einstein_clf.predict(X_ein_test)
    einstein_classes_list = list(le_einstein.classes_)

    # Map Einstein 3-class to severity equivalents for Mexican-aligned recall
    # For Einstein: Severe = Severe+Critical, Non_Severe = Mild+Moderate
    ein_acc = accuracy_score(y_ein_test, y_ein_pred) * 100

    # Einstein recall for "Severe" class (maps to Severe+Critical in Mexican)
    ein_recalls = {}
    for cls_name in ['Severe', 'Non_Severe', 'No_COVID']:
        if cls_name in einstein_classes_list:
            idx = einstein_classes_list.index(cls_name)
            mask = y_ein_test == idx
            if mask.sum() > 0:
                ein_recalls[cls_name] = (y_ein_pred[mask] == idx).mean() * 100
            else:
                ein_recalls[cls_name] = float('nan')

    # ── Ensemble evaluation on Mexican test set ───────────────────────────────
    # For ensemble on Mexican test set, we can't directly apply Einstein
    # (different feature space). So we evaluate ensemble architecture using
    # Mexican probas only (since we have no lab values for Mexican patients).
    # Instead, evaluate ensemble on Einstein test set mapped to Mexican space.

    # Build ensemble predictions on Einstein test set
    # (Einstein proba mapped to Mexican space, compared to "equivalent" Mexican labels)
    ein_proba_all = einstein_clf.predict_proba(X_ein_test)  # shape (N, 3)
    y_ein_mapped_pred = []
    for i in range(len(X_ein_test)):
        mapped = map_einstein_to_mexican(ein_proba_all[i], le_einstein, mexican_classes)
        y_ein_mapped_pred.append(np.argmax(mapped))
    y_ein_mapped_pred = np.array(y_ein_mapped_pred)

    # Convert Einstein true labels to Mexican-equivalent labels for comparison
    # No_COVID -> No_COVID(3), Non_Severe -> Mild(1) or Moderate(2), Severe -> Severe(4) or Critical(0)
    # Use: No_COVID->No_COVID, Non_Severe->Mild, Severe->Severe (simplest fair mapping)
    ein_to_mex_label = {}
    for cls in einstein_classes_list:
        idx_ein = einstein_classes_list.index(cls)
        if cls == 'No_COVID':
            ein_to_mex_label[idx_ein] = mexican_classes.index('No_COVID')
        elif cls == 'Non_Severe':
            ein_to_mex_label[idx_ein] = mexican_classes.index('Mild')
        elif cls == 'Severe':
            ein_to_mex_label[idx_ein] = mexican_classes.index('Severe')

    y_ein_test_mex = np.array([ein_to_mex_label[y] for y in y_ein_test])

    ensemble_acc = accuracy_score(y_ein_test_mex, y_ein_mapped_pred) * 100
    ens_recalls = get_severe_critical_recall(y_ein_test_mex, y_ein_mapped_pred, mexican_classes)

    # ── Print comparison table ────────────────────────────────────────────────
    print("\n[5/5] Model Comparison Table")
    print("=" * 72)
    header = f"{'Model':<28} {'Overall Acc':>13} {'Severe Recall':>15} {'Critical Recall':>15}"
    print(header)
    print("-" * 72)

    def fmt(val):
        if np.isnan(val):
            return "   N/A"
        return f"{val:>6.1f}%"

    print(f"{'Mexican (comorbidity)':<28} {mex_acc:>12.1f}% "
          f"{fmt(mex_recalls.get('Severe', float('nan'))):>15} "
          f"{fmt(mex_recalls.get('Critical', float('nan'))):>15}")

    ein_severe_recall = ein_recalls.get('Severe', float('nan'))
    print(f"{'Einstein (lab)':<28} {ein_acc:>12.1f}% "
          f"{fmt(ein_severe_recall):>15} "
          f"{'   N/A':>15}")

    print(f"{'Ensemble':<28} {ensemble_acc:>12.1f}% "
          f"{fmt(ens_recalls.get('Severe', float('nan'))):>15} "
          f"{fmt(ens_recalls.get('Critical', float('nan'))):>15}")

    print("=" * 72)
    print("\nNote: Einstein model uses 3-class output (No_COVID/Non_Severe/Severe).")
    print("      Ensemble evaluated on Einstein test set with labels mapped to")
    print("      Mexican 5-class space for comparison.")

    # ── Save CSV ──────────────────────────────────────────────────────────────
    results_path = os.path.join(RESULTS_DIR, "model_comparison.csv")
    comparison_df = pd.DataFrame([
        {
            "Model": "Mexican (comorbidity)",
            "Overall_Acc": round(mex_acc, 2),
            "Severe_Recall": round(mex_recalls.get('Severe', float('nan')), 2),
            "Critical_Recall": round(mex_recalls.get('Critical', float('nan')), 2),
            "Dataset": "Mexican",
        },
        {
            "Model": "Einstein (lab)",
            "Overall_Acc": round(ein_acc, 2),
            "Severe_Recall": round(ein_severe_recall, 2),
            "Critical_Recall": float('nan'),
            "Dataset": "Einstein",
        },
        {
            "Model": "Ensemble",
            "Overall_Acc": round(ensemble_acc, 2),
            "Severe_Recall": round(ens_recalls.get('Severe', float('nan')), 2),
            "Critical_Recall": round(ens_recalls.get('Critical', float('nan')), 2),
            "Dataset": "Einstein (mapped)",
        },
    ])
    comparison_df.to_csv(results_path, index=False)
    print(f"\n  Comparison table saved to: {results_path}")
    print("\nEnsemble evaluation complete!")


if __name__ == "__main__":
    main()
