"""
Einstein Hospital Lab-Based COVID-19 Severity Classifier — Training Script
Dataset: Einstein Hospital Brazil (dataset.xlsx, 5,644 rows, 111 columns)
Model: XGBClassifier (300 estimators, max_depth=5, lr=0.05) + SMOTE
Target: 3-class severity (No_COVID, Non_Severe, Severe)

NOTE on missingness: The Einstein dataset stores lab results sparsely — the
target blood-work columns are ~89-100% missing at the dataset level, but the
missingness is CORRELATED: 602 patients have nearly complete CBC panels.
We filter to those rows (>=5 non-null among target features), drop columns
that are STILL >70% missing within that subset (e.g. D-Dimer, Ferritin,
LDH), then train on the viable feature set.
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
from sklearn.metrics import classification_report, accuracy_score

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "dataset.xlsx")
MODEL_DIR = os.path.join(BASE_DIR, "backend", "app", "models")
EINSTEIN_CLF_PATH  = os.path.join(MODEL_DIR, "einstein_classifier.pkl")
EINSTEIN_FEAT_PATH = os.path.join(MODEL_DIR, "einstein_features.pkl")
EINSTEIN_ENC_PATH  = os.path.join(MODEL_DIR, "einstein_label_encoder.pkl")

os.makedirs(MODEL_DIR, exist_ok=True)

# ── Feature alias → keyword mapping ───────────────────────────────────────────
# SHORT ALIASES are saved in einstein_features.pkl so the frontend/backend
# can reference them by short name. Column names are resolved at load time.
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
    """Find first column whose lowercase name contains any of the keywords."""
    cols_lower = [c.lower() for c in df_columns]
    for kw in keywords:
        kw = kw.lower()
        for i, col_low in enumerate(cols_lower):
            if kw in col_low:
                return df_columns[i]
    return None


def assign_severity(row):
    """3-class severity from Einstein columns."""
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


def main():
    print("=" * 60)
    print("Einstein Lab-Based COVID Classifier — Training")
    print("=" * 60)

    # 1. Load data
    print(f"\n[1/7] Loading dataset from:\n  {DATA_PATH}")
    df = pd.read_excel(DATA_PATH)
    print(f"  Loaded {len(df):,} rows, {df.shape[1]} columns")

    # 2. Assign severity labels
    print("\n[2/7] Assigning 3-class severity labels...")
    df['SEVERITY'] = df.apply(assign_severity, axis=1)
    dist = df['SEVERITY'].value_counts()
    for label, count in dist.items():
        print(f"  {label:12s}: {count:>6,} ({count/len(df)*100:.1f}%)")

    # 3. Resolve actual column names from keywords
    print("\n[3/7] Resolving feature columns by keyword matching...")
    alias_to_col = {}
    for alias, keywords in ALIAS_KEYWORDS.items():
        col = find_column(list(df.columns), keywords)
        if col is not None:
            alias_to_col[alias] = col
            print(f"  {alias:20s} -> {repr(col)}")
        else:
            print(f"  {alias:20s} -> NOT FOUND (skipping)")

    all_aliases   = list(alias_to_col.keys())
    actual_cols   = [alias_to_col[a] for a in all_aliases]

    # 4. FIRST filter rows: keep only those with >=5 non-null target features
    #    (missingness in this dataset is correlated — the 602 rows with lab
    #     data have nearly complete CBC panels)
    print("\n[4/7] Filtering rows with >=5 non-null target features...")
    feat_df_full = df[actual_cols].copy()
    for col in actual_cols:
        feat_df_full[col] = pd.to_numeric(feat_df_full[col], errors='coerce')

    non_null_count = feat_df_full.notna().sum(axis=1)
    row_mask = non_null_count >= 5
    feat_subset = feat_df_full[row_mask].copy()
    severity_subset = df.loc[row_mask, 'SEVERITY']
    print(f"  Rows with >=5 non-null features: {row_mask.sum():,} "
          f"(dropped {(~row_mask).sum():,} data-poor rows)")

    # 5. THEN drop columns with >70% missing WITHIN the usable subset
    print("\n[5/7] Dropping columns with >70% missing within usable subset...")
    valid_aliases = []
    for alias, col in zip(all_aliases, actual_cols):
        miss_rate = feat_subset[col].isna().mean()
        if miss_rate <= 0.70:
            valid_aliases.append(alias)
            print(f"  KEEP  {alias:20s}  missing={miss_rate*100:.1f}%")
        else:
            print(f"  DROP  {alias:20s}  missing={miss_rate*100:.1f}%")

    if not valid_aliases:
        print("ERROR: No features survived the missingness filter. Aborting.")
        sys.exit(1)

    valid_actual_cols = [alias_to_col[a] for a in valid_aliases]
    feat_df = feat_subset[valid_actual_cols].copy()
    print(f"  Kept {len(valid_aliases)} features: {valid_aliases}")

    # Fill remaining NaN with column median
    for col in valid_actual_cols:
        med = feat_df[col].median()
        feat_df[col] = feat_df[col].fillna(med if not pd.isna(med) else 0.0)

    # 6. Prepare X and y
    print("\n[6/7] Preparing training data...")
    X = feat_df.values.astype(np.float32)
    le = LabelEncoder()
    y = le.fit_transform(severity_subset)
    print(f"  Classes: {list(le.classes_)}")
    print(f"  Total usable samples: {len(X):,}")

    unique, counts = np.unique(y, return_counts=True)
    for cls_idx, cnt in zip(unique, counts):
        print(f"  {le.classes_[cls_idx]:12s}: {cnt:,}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"  Train: {len(X_train):,} | Test: {len(X_test):,}")

    # Apply SMOTE (k_neighbors capped to min class size - 1)
    print("\n  Applying SMOTE to balance training classes...")
    min_class_train = np.min(np.bincount(y_train))
    k_neighbors = min(5, min_class_train - 1)
    if k_neighbors < 1:
        print(f"  WARNING: min class in train has only {min_class_train} samples. "
              "Skipping SMOTE, training on imbalanced data.")
        X_train_sm, y_train_sm = X_train, y_train
    else:
        sm = SMOTE(random_state=42, k_neighbors=k_neighbors)
        X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)
    unique2, counts2 = np.unique(y_train_sm, return_counts=True)
    for cls_idx, cnt in zip(unique2, counts2):
        print(f"    {le.classes_[cls_idx]:12s}: {cnt:,} samples after SMOTE")
    print(f"  Train after SMOTE: {len(X_train_sm):,}")

    # 7. Train
    print("\n[7/7] Training XGBClassifier...")
    clf = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='mlogloss',
        random_state=42,
        use_label_encoder=False,
    )
    clf.fit(X_train_sm, y_train_sm)

    # Evaluate
    y_pred = clf.predict(X_test)
    overall_acc = accuracy_score(y_test, y_pred) * 100
    print(f"\n  Overall accuracy: {overall_acc:.2f}%")
    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    print("\n  Per-class accuracy:")
    for i, cls in enumerate(le.classes_):
        cls_mask = y_test == i
        if cls_mask.sum() > 0:
            cls_acc = (y_pred[cls_mask] == i).mean() * 100
            print(f"    {cls:12s}: {cls_acc:.1f}%")

    # Save models — save SHORT ALIASES as feature list
    joblib.dump(clf, EINSTEIN_CLF_PATH)
    joblib.dump(valid_aliases, EINSTEIN_FEAT_PATH)
    joblib.dump(le, EINSTEIN_ENC_PATH)

    print(f"\n  Saved classifier to:    {EINSTEIN_CLF_PATH}")
    print(f"  Saved feature aliases:  {EINSTEIN_FEAT_PATH}")
    print(f"  Saved label encoder:    {EINSTEIN_ENC_PATH}")
    print(f"\n  Feature aliases saved: {valid_aliases}")
    print("\nEinstein training complete!")


if __name__ == "__main__":
    main()
