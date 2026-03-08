# Datasets

This directory contains the raw datasets used for training the COVID-19 severity models.
The actual data files are excluded from git (too large). Follow the instructions below to download them.

## Dataset 1: Mexican Government COVID-19 Dataset

- **Source**: Mexico Ministry of Health (Secretaría de Salud)
- **Kaggle**: https://www.kaggle.com/datasets/meirnizri/covid19-dataset
- **File**: `data/raw/Covid Data.csv`
- **Size**: ~1M records, 21 columns
- **Features**: Patient demographics, comorbidities (diabetes, COPD, hypertension, etc.), COVID test result, hospitalisation type, ICU/intubation status, date of death
- **License**: Open Government Data License (Mexico)

**Download**:
```bash
kaggle datasets download -d meirnizri/covid19-dataset
unzip covid19-dataset.zip -d data/raw/
```

## Dataset 2: Einstein Hospital Brazil (Sao Paulo)

- **Source**: Hospital Israelita Albert Einstein, São Paulo, Brazil
- **Kaggle**: https://www.kaggle.com/datasets/einsteindata4u/covid19
- **File**: `data/raw/dataset.xlsx`
- **Size**: 5,644 patients, 111 lab feature columns
- **Features**: SARS-CoV-2 test result, CBC panel (hemoglobin, leukocytes, lymphocytes, platelets, etc.), hospital admission type
- **License**: CC BY-SA 4.0
- **Note**: Only 602 patients have complete CBC panels — these are used for Einstein model training

**Download**:
```bash
kaggle datasets download -d einsteindata4u/covid19
unzip covid19.zip -d data/raw/
```

## After downloading

Run training scripts:
```bash
python src/train_model.py           # Trains Mexican comorbidity model (~5 min)
python src/train_einstein_model.py  # Trains Einstein lab model (~30 sec)
python src/validate_einstein.py     # Cross-dataset validation
```

Trained model files are saved to `backend/app/models/`.
