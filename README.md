---
title: COVID-19 CDSS
emoji: 🦠
colorFrom: red
colorTo: blue
sdk: docker
pinned: false
license: apache-2.0
short_description: Ensemble ML COVID-19 severity assessment
---

# 🦠 COVID-19 Clinical Decision Support System

[![Hugging Face Spaces](https://img.shields.io/badge/🤗%20HF%20Spaces-Live%20Demo-blue)](https://huggingface.co/spaces/jemiridaniel/covid-llm-cdss)
[![GitHub](https://img.shields.io/badge/GitHub-covid--llm--cdss-181717?logo=github)](https://github.com/jemiridaniel/covid-llm-cdss)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python)](https://python.org)
[![License](https://img.shields.io/badge/License-Apache%202.0-green)](LICENSE)
[![Accuracy](https://img.shields.io/badge/Ensemble%20Accuracy-86.0%25-brightgreen)](publication/stats_report.md)
[![Kappa](https://img.shields.io/badge/Cohen's%20Kappa-0.798-blueviolet)](publication/stats_report.md)

**ML-powered COVID-19 severity assessment combining comorbidity and lab-based models with SHAP explainability and LLM clinical reasoning.**

---

## Key Results

| Model | Accuracy | Cohen's Kappa | Training Data | Country |
|-------|----------|---------------|---------------|---------|
| Mexican (comorbidity) | 51.6% | 0.386 (fair) | 500K+ patients | 🇲🇽 Mexico |
| Einstein (lab-based) | 85.1% | 0.555 (moderate) | 602 patients | 🇧🇷 Brazil |
| **Ensemble** | **86.0%** | **0.798 (substantial)** | Multi-national | 🌍 |

> McNemar test: ensemble vs Mexican-only p≈0.000 (χ²=2798). Bootstrap 95% CI: [83.4–85.0%].

---

## Features

- **Dual-model ensemble** — comorbidity signals (500K Mexican records) + lab signals (Einstein Hospital CBC panel)
- **Graceful degradation** — works without lab values (falls back to comorbidity model only)
- **SHAP explainability** — top 5 contributing features per prediction with direction and magnitude
- **LLM clinical reasoning** — Groq (Llama 3.1) → Anthropic → OpenAI fallback chain
- **PDF reports** — downloadable clinical report with SHAP section
- **5-class severity** — No COVID / Mild / Moderate / Severe / Critical
- **Web interface** — React frontend with Recharts visualisations

---

## Architecture

```
Patient Input (Demographics + Comorbidities)
        │
        ▼
┌───────────────────┐        ┌──────────────────────┐ (optional)
│  Mexican XGBoost  │        │  Einstein XGBoost    │
│  17 features      │        │  11 CBC lab features │
│  5-class output   │        │  3-class output      │
└────────┬──────────┘        └──────────┬───────────┘
         │                              │ mapped to 5-class
         │         60%           40%    │
         └──────────────┬───────────────┘
                        ▼
              ┌──────────────────┐
              │ Weighted Ensemble│
              └────────┬─────────┘
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
    SHAP Explainer  Groq LLM    Severity
    (top 5 feats)  Reasoning    Output
          └────────────┴────────────┘
                        │
                   PDF Report
```

Lab values are **optional** — the system degrades gracefully to comorbidity-only mode when no CBC panel is available.

---

## Quick Start

```bash
# Clone
git clone https://github.com/jemiridaniel/covid-llm-cdss.git
cd covid-llm-cdss

# Python environment
python -m venv venv && source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r backend/requirements.txt

# Environment variables
cp .env.example backend/.env
# Edit backend/.env and add your Groq API key

# Download datasets (see data/README.md)
# Then train models:
python src/train_model.py          # Mexican model (~5 min)
python src/train_einstein_model.py # Einstein model (~1 min)

# Start backend
cd backend && uvicorn app.main:app --reload --port 8000

# Start frontend (new terminal)
cd frontend && npm install && npm start
```

Open [http://localhost:3000](http://localhost:3000)

---

## Datasets

| Dataset | Source | Size | License |
|---------|--------|------|---------|
| Mexican Government COVID-19 | [Kaggle](https://www.kaggle.com/datasets/meirnizri/covid19-dataset) | 1M+ records | Open Government |
| Einstein Hospital Brazil | [Kaggle](https://www.kaggle.com/datasets/einsteindata4u/covid19) | 5,644 patients | CC BY-SA 4.0 |

See [`data/README.md`](data/README.md) for setup instructions.

---

## Severity Classification

| Severity | Definition | Recommendation |
|----------|-----------|----------------|
| 🔴 Critical | Intubated or deceased | Immediate ICU / emergency services |
| 🟠 Severe | ICU admission, not intubated | Emergency department — urgent |
| 🟡 Moderate | Hospitalised, not ICU | Medical evaluation within 24h |
| 🟢 Mild | Outpatient, confirmed COVID | Home isolation, monitor SpO₂ |
| ⚪ No COVID | Negative test result | Monitor, retest if symptoms persist |

---

## Statistical Highlights

- **McNemar test** (ensemble vs comorbidity-only): χ²=2798, **p≈0.000** — highly significant improvement
- **Bootstrap 95% CI** (1,000 iterations): Ensemble [83.4–85.0%], Mexican [50.6–52.6%]
- **Cohen's Kappa**: Ensemble 0.798 (approaching "almost perfect" threshold of 0.80)
- **Severe class recall**: 5.7% (GBC baseline) → 31.0% (XGBoost + SMOTE) — 5.4× improvement
- **Cross-dataset validation**: Mexican model on Brazilian data → 38.25% (expected, missing lab features)

---

## Research Context

Part of a **generalizable infectious disease CDSS framework**.

> See also: **[MalariaLLM](https://github.com/jemiridaniel/malaria-llm-cdss)** — same architecture applied to malaria severity prediction.

Both systems share: XGBoost + SHAP + Groq LLM + FastAPI + React. The modular design enables rapid adaptation to new infectious diseases with domain-specific datasets.

---

## Clinical Safety Disclaimer

> ⚠️ **This system is a research prototype for educational purposes only.**
> It is NOT a certified medical device and must NOT be used for actual clinical diagnosis or treatment decisions.
> Always consult a qualified healthcare professional for medical advice.
> Model predictions may be incorrect — clinical judgment must take precedence.

---

## Authors

**Jemiri Daniel Taiwo** — Lead Developer & ML Engineer
Department of Computer Science, Federal University of Technology Owerri (FUTO), Nigeria
[GitHub](https://github.com/jemiridaniel) · [LinkedIn](https://linkedin.com/in/jemiridaniel)

**Dr. Olaoluwa Aladetola** — Clinical Advisor & Co-Author
*[Institution placeholder]*

---

## Roadmap

- [ ] Prospective clinical validation study
- [ ] EHR integration (FHIR API)
- [ ] Multi-language support (Spanish, Portuguese, French)
- [ ] LDH / D-Dimer / Ferritin model (larger dataset needed)
- [ ] Mobile-optimised interface
- [ ] Expansion to dengue fever and tuberculosis

---

## Citation

```bibtex
@software{jemiri2024covidcdss,
  author    = {Taiwo, Jemiri Daniel and Aladetola, Olaoluwa},
  title     = {COVID-19 Clinical Decision Support System:
               A Multi-Dataset Ensemble Approach with Explainable AI},
  year      = {2025},
  url       = {https://github.com/jemiridaniel/covid-llm-cdss},
  license   = {Apache-2.0}
}
```

---

## License

Apache License 2.0 — see [LICENSE](LICENSE) for details.
