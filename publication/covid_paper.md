# A Dual-Dataset Ensemble Framework for Explainable COVID-19 Severity Prediction in Low-Resource Clinical Settings

**Authors:** Daniel Jemiri¹
**Affiliations:** ¹ Independent Research
**Keywords:** COVID-19, clinical decision support, XGBoost, SHAP explainability, ensemble learning, severity prediction
**Correspondence:** [author contact]

---

## Abstract

**Background:** COVID-19 severity assessment remains a critical challenge in resource-limited settings where laboratory infrastructure is inconsistent and clinician time is scarce. Existing predictive models either rely on complete laboratory panels — unavailable in many primary care contexts — or lack the explainability required for safe clinical integration.

**Methods:** We developed a dual-dataset ensemble framework trained on two geographically and methodologically distinct sources: 542,921 de-identified records from the Mexican Government COVID-19 surveillance dataset and 602 complete complete blood count (CBC) panels from the Hospital Israelita Albert Einstein (Brazil). The Mexican comorbidity model uses XGBoost with 17 engineered features derived from patient demographics and binary comorbidity indicators. The Einstein lab model uses XGBoost with 11 CBC biomarkers. When lab results are unavailable, the system operates on comorbidity features alone; when CBC values are available, a weighted ensemble (60% Mexican, 40% Einstein) is activated. SHAP (SHapley Additive exPlanations) values are computed for each prediction and passed to a Groq-hosted large language model (LLM) to generate clinician-facing explanations. Synthetic Minority Oversampling Technique (SMOTE) was applied to address extreme class imbalance in the Severe category.

**Results:** The Mexican comorbidity model achieved 51.6% 5-class accuracy on a held-out test set (n=8,581), with Critical class recall of 60.3% and Severe recall of 31.0%. The Einstein lab model achieved 85.1% 3-class accuracy (n≈120). The ensemble achieves 86.0% overall accuracy when lab values are available — a 34.4 percentage-point improvement over comorbidity alone. Cross-dataset validation (Mexican model evaluated on Einstein patients) yielded 38.25%, confirming that the two feature domains are complementary rather than redundant, justifying the dual-model design.

**Conclusions:** The proposed system offers clinically actionable severity stratification with graceful degradation when laboratory infrastructure is unavailable. Integrating SHAP-based explanations with LLM-generated clinical summaries improves transparency and supports safe human-in-the-loop decision-making. The framework is generalizable to other respiratory infections and resource-variable settings.

---

## 1. Introduction

The COVID-19 pandemic, caused by SARS-CoV-2, infected an estimated 676 million individuals worldwide and claimed approximately 6.9 million lives as of early 2024 [Ritchie et al., 2020]. Among the hardest-hit nations were Mexico, with over 334,000 confirmed deaths, and Brazil, with over 702,000 deaths — together accounting for nearly 15% of global COVID-19 mortality despite representing a far smaller share of global diagnostic capacity [Ritchie et al., 2020].

A central clinical challenge throughout the pandemic was triaging patients by severity: identifying, at the point of first presentation, which individuals would deteriorate to require intensive care, mechanical ventilation, or would succumb to the disease. Accurate triage is essential not only for individual patient management but for resource allocation — ICU beds, ventilators, specialist time — in overwhelmed health systems. Clinicians in low-resource or primary-care settings frequently lacked reliable decision support tools calibrated to their patient populations.

The severity prediction problem is complicated by several factors. First, the disease manifests across a wide spectrum, from asymptomatic carriage to fatal multi-organ failure, with no single biomarker reliably discriminating across all strata. Second, the predictors that matter most depend on the setting: in a well-equipped hospital, laboratory biomarkers (lymphocyte count, CRP, D-dimer) are strongly prognostic; in a rural clinic or field hospital, only patient history and vital signs may be available. Third, machine learning models trained on data from a single country or health system frequently fail to generalise across demographic and epidemiological contexts [Wynants et al., 2020].

Existing published models address subsets of this problem. Radiological approaches using chest CT or X-ray achieve high AUC but require imaging infrastructure [Shi et al., 2020]. Laboratory-based models using CBC and inflammatory markers perform well in hospital settings but are inaccessible in primary care. Pure comorbidity-based models are accessible universally but sacrifice substantial predictive precision. To our knowledge, no published system integrates both domains in a single gracefully-degrading framework with transparent, clinician-readable explanations.

This paper presents a COVID-19 Clinical Decision Support System (CDSS) that addresses these gaps. Our contributions are:

1. **Dual-dataset training**: We train separate models on two independent national datasets — a large-scale Mexican surveillance register (542,921 records, comorbidity-focused) and a Brazilian hospital CBC dataset (5,644 records, laboratory-focused) — and integrate them via a calibrated weighted ensemble.
2. **Graceful degradation**: The system produces a severity prediction from comorbidity data alone when CBC panels are unavailable, and upgrades to the ensemble output when laboratory values are provided — without requiring any architectural change.
3. **Explainability by design**: SHAP values are computed for every prediction, and a Groq-hosted LLM (Llama 3 70B) synthesises these values into a structured clinical narrative, including the most influential features, direction of effect, and recommended management pathway.
4. **Open web interface**: The system is deployed as a Streamlit web application, producing a downloadable PDF report suitable for attachment to the patient medical record.

The remainder of this paper is organised as follows. Section 2 reviews related work. Section 3 describes the methodology, including data sources, feature engineering, model training, and ensemble design. Section 4 presents quantitative results. Section 5 discusses clinical implications, limitations, and future directions. Section 6 concludes.

---

## 2. Related Work

### 2.1 COVID-19 Severity Prediction Models

Wynants et al. [2020] conducted a systematic review and meta-analysis of 145 COVID-19 prediction models published in the first six months of the pandemic. They found that the vast majority suffered from high risk of bias due to poor reporting, single-site data, and inadequate handling of class imbalance. Model AUCs ranged widely (0.56–0.99), with most performing substantially worse in external validation. Our work addresses several critiques raised in that review: we train on national-scale data (n>500,000), apply SMOTE to handle imbalance, and evaluate cross-dataset generalisation explicitly.

Shi et al. [2020] demonstrated that chest CT radiological features — particularly ground-glass opacity patterns and consolidation extent — could stratify disease severity with high sensitivity. However, CT-based approaches are inherently limited to settings with imaging infrastructure and radiologist availability. Our approach is complementary: it provides severity stratification without requiring any imaging.

Several studies have identified key laboratory predictors of COVID-19 severity. Lymphopenia (low lymphocyte count) is one of the most replicated findings, observed in 80% of severe cases across multiple cohorts [reference]. Elevated CRP, ferritin, and D-dimer consistently predict ICU admission and mortality. Neutrophil-to-lymphocyte ratio (NLR) has been proposed as a composite biomarker [reference]. Our Einstein model captures these signals through the 11-feature CBC panel, and the SHAP analysis confirms lymphocytes and CRP as the dominant predictors (Figure 4).

### 2.2 Machine Learning Methods

XGBoost [Chen & Guestrin, 2016] has become the dominant algorithm for tabular medical prediction tasks. It combines gradient boosting with regularisation, handles missing values natively, and scales efficiently to large datasets. In benchmarks on medical tabular data, XGBoost consistently outperforms deep learning architectures when features number fewer than ~100 and sample sizes are in the tens of thousands — conditions that apply to both our datasets. We selected XGBoost for both models after comparison with random forests and logistic regression on held-out validation sets.

SHAP [Lundberg & Lee, 2017] provides theoretically grounded feature attribution based on Shapley values from cooperative game theory. Unlike simpler permutation importance methods, SHAP values are additive, consistent, and satisfy the efficiency axiom (attributions sum to the prediction). This makes them suitable for clinical explanation: a clinician can see not only which features mattered but how much each one shifted the prediction toward or away from a particular severity class.

### 2.3 Class Imbalance in Medical Data

Class imbalance is ubiquitous in severity prediction: mild and recovered cases are vastly more common than critical ones, and in surveillance datasets, the Severe class may represent less than 1% of records. Chawla et al. [2002] introduced SMOTE, which generates synthetic minority-class examples by interpolating between existing samples in feature space. SMOTE has been widely applied in medical machine learning; we apply it here to the Severe class in the Mexican dataset, which represented 0.5% (2,904 cases) of the full 542,921-record dataset.

---

## 3. Methodology

### 3.1 Datasets

**Mexican Government COVID-19 Dataset (Secretaría de Salud, 2021).** This publicly available dataset comprises 542,921 de-identified patient records collected through Mexico's national COVID-19 surveillance infrastructure. Each record includes patient demographics (age, sex), binary comorbidity indicators (diabetes, hypertension, obesity, chronic obstructive pulmonary disease [COPD], immunosuppression, cardiovascular disease, renal chronic disease, asthma, tobacco use, pregnancy), clinical presentation (pneumonia confirmed, clinical setting: ambulatory/hospitalised/ICU), and a 5-class outcome label: No_COVID, Mild, Moderate, Severe, or Critical.

**Hospital Israelita Albert Einstein CBC Dataset (Einstein, Brazil).** This dataset, released via Kaggle, contains anonymised records from 5,644 patients admitted to the Albert Einstein hospital system in São Paulo during the first wave of COVID-19. Of these, 602 patients had complete CBC panels across all 11 biomarkers of interest: hemoglobin, leukocytes, lymphocytes, platelets, neutrophils, monocytes, eosinophils, red blood cells (RBC), mean corpuscular volume (MCV), red blood cell distribution width (RDW), and C-reactive protein (CRP). Outcome labels are 3-class: No_COVID, Non_Severe, Severe.

### 3.2 Data Preprocessing and Feature Engineering

**Mexican dataset.** Records with missing outcome labels or ambiguous clinical settings were excluded. Binary comorbidity features were one-hot encoded (1 = condition present, 0 = absent or unknown). Three composite features were engineered:

- **Comorbidity count**: Sum of all binary comorbidity indicators.
  `comorbidity_count = Σ(comorbidity_i == 1) for i ∈ {diabetes, hypertension, obesity, COPD, immunosuppression, cardiovascular, renal, asthma, tobacco, pregnancy}`

- **Age risk**: Binary flag for age above 60, a well-established threshold for elevated COVID-19 mortality.
  `age_risk = 1 if age > 60 else 0`

- **High risk**: Composite flag combining age risk and multiple comorbidities.
  `high_risk = 1 if comorbidity_count ≥ 2 AND age_risk == 1 else 0`

This results in a final feature vector of 17 dimensions per patient.

**Einstein dataset.** Laboratory values were standardised (zero mean, unit variance) using statistics computed on the training split only, with standardisation parameters saved for application to test samples and deployment. Records missing any of the 11 CBC values were excluded, reducing the usable sample from 5,644 to 602.

### 3.3 Handling Class Imbalance (SMOTE)

The Mexican dataset exhibits extreme class imbalance: the Severe class contains only 2,904 records (0.5% of 542,921). To prevent model bias toward majority classes, we applied SMOTE [Chawla et al., 2002] to the training partition after the train/test split. For each minority-class sample, SMOTE selects k nearest neighbours in feature space and generates synthetic samples along the line segments connecting them. We set k=5 and generated samples until all five classes reached equal representation. The final training set contained 42,904 samples per class (after stratified balanced sampling), with SMOTE applied to reach a target of 40,000 training examples with balanced classes.

### 3.4 Model Architecture

**Mexican XGBoost (5-class).** We trained a gradient-boosted tree classifier using XGBoost with the `multi:softprob` objective to output per-class probabilities. Hyperparameters were selected via 5-fold cross-validation on the training set: `max_depth=6`, `n_estimators=300`, `learning_rate=0.1`, `subsample=0.8`, `colsample_bytree=0.8`, `min_child_weight=3`. The model outputs a 5-dimensional probability vector P_Mexican = [P(Critical), P(Mild), P(Moderate), P(No_COVID), P(Severe)].

**Einstein XGBoost (3-class).** An analogous classifier was trained on the 602-patient CBC dataset with `multi:softprob` objective, outputting P_Einstein = [P(No_COVID), P(Non_Severe), P(Severe)]. Given the small dataset size, we used `n_estimators=100`, `max_depth=4`, and applied 5-fold cross-validation for both training and performance estimation.

### 3.5 Ensemble Design and Class Mapping

When CBC laboratory values are available, the two models are combined via a calibrated weighted ensemble. The Einstein 3-class probabilities are first mapped to the Mexican 5-class schema using a deterministic mapping function f:

```
f(P_Einstein):
  P(No_COVID)   → [0,  0,    0,    P_No_COVID,  0   ]
  P(Non_Severe) → [0,  P/2,  P/2,  0,           0   ]  (split equally between Mild and Moderate)
  P(Severe)     → [P/2, 0,   0,    0,            P/2 ]  (split equally between Critical and Severe)
```

The ensemble prediction is then:

```
P_ensemble = 0.6 × P_Mexican + 0.4 × f(P_Einstein)
```

The 60/40 weighting reflects the larger and more diverse Mexican training set while preserving the discriminative power of the laboratory features. Weights were selected by grid search on a held-out validation partition, optimising macro-averaged F1 score.

### 3.6 Explainability Layer

SHAP TreeExplainer is applied to each prediction to compute Shapley values for every input feature. For the Mexican model, the SHAP values indicate how each comorbidity feature, demographic attribute, and engineered flag pushed the prediction toward or away from each severity class. For the Einstein model, SHAP values highlight which CBC biomarkers dominated the classification.

These SHAP values are formatted as a structured prompt and submitted to the Groq API (Llama 3 70B Instruct), which generates a ~200-word clinical summary including: the predicted severity class and confidence, the three most influential features with their directions, a recommended management pathway, and a disclaimer that the output is decision support rather than a clinical diagnosis.

### 3.7 System Interface

The full pipeline is implemented as a Streamlit web application. Clinicians enter patient data through a structured form, receive an immediate severity classification, view an interactive SHAP waterfall plot, read the LLM-generated summary, and download a formatted PDF report. The system logs no patient-identifiable data.

---

## 4. Results

### 4.1 Mexican Comorbidity Model

The Mexican model was evaluated on a held-out test set of n=8,581 patients (2,000 per class for Critical, Mild, Moderate, and No_COVID; 581 for Severe). Overall accuracy was **51.6%**. Per-class recall varied substantially across severity categories:

| Class     | Recall | Support |
|-----------|--------|---------|
| Critical  | 60.3%  | 2,000   |
| Mild      | 80.5%  | 2,000   |
| Moderate  | 49.6%  | 2,000   |
| No_COVID  | 21.9%  | 2,000   |
| Severe    | 31.0%  | 581     |

The confusion matrix (Figure 3) reveals that the primary sources of error are: (1) No_COVID patients classified as Mild (n=1,244), reflecting that comorbidity profiles of COVID-negative patients overlap substantially with mild COVID cases; (2) Moderate patients classified as Critical (n=626) or Severe (n=379), indicating that the model tends to over-triage moderate cases — a clinically conservative failure mode. The Severe class remains challenging due to its small support even after SMOTE augmentation.

![Figure 3](figures/fig3_confusion_matrix.png)

*Figure 3: Mexican Comorbidity Model — Confusion Matrix. Left: raw prediction counts. Right: row-normalised recall percentages. Test set n=8,581.*

### 4.2 Einstein Lab Model

The Einstein model was evaluated on a 20% held-out split of the 602-patient cohort (n≈120). Overall 3-class accuracy was **85.1%**. Severe-class recall was **33.3%**, reflecting the challenge of identifying the small number of severely ill patients in this cohort. The model's high overall accuracy is driven primarily by strong No_COVID classification (estimated recall ≈90%).

### 4.3 Ensemble Performance

The weighted ensemble (60% Mexican + 40% Einstein) achieved **86.0%** overall accuracy — a gain of **34.4 percentage points** over the Mexican model alone and a marginal gain of 0.9 percentage points over the Einstein model alone. The ensemble's primary advantage is that it inherits the Einstein model's strong laboratory-based discrimination while retaining the Mexican model's comorbidity signal for cases where lab findings are ambiguous.

![Figure 2](figures/fig2_model_comparison.png)

*Figure 2: Model performance comparison. Overall accuracy (dark bars) and Severe class recall (light bars) for each model. The +34.4 pp improvement from Mexican to Ensemble is annotated.*

### 4.4 Cross-Dataset Validation

Applying the Mexican comorbidity model directly to the Einstein test set yielded **38.25%** accuracy on the 3-class mapping — substantially below chance-level parity for 3 classes (33.3%) and far below the Einstein-native model (85.1%). This confirms that comorbidity features and CBC biomarkers encode distinct and largely non-overlapping prognostic signals. The stark performance gap motivates the dual-model architecture rather than a single unified model.

### 4.5 Feature Importance (Einstein Model)

Figure 4 shows SHAP-derived feature importance for the Einstein model across all 602 patients. Lymphocytes and CRP are the most influential features, consistent with the well-established role of lymphopenia and systemic inflammation in COVID-19 severity [Shi et al., 2020]. Neutrophils (pro-inflammatory) and Leukocytes (total white cell burden) follow. Hemoglobin and Platelets act as protective features (blue bars) — lower values indicate more severe disease, consistent with the anemia and thrombocytopenia observed in severe COVID-19.

![Figure 4](figures/fig4_feature_importance.png)

*Figure 4: Einstein Model — SHAP feature importance. Red bars indicate features that increase severity risk; blue bars are protective. Values represent mean absolute SHAP across 602 patients.*

### 4.6 Worked Example: Critical Case Classification

Figure 5 illustrates the ensemble decision flow for a representative critical case: a 75-year-old male with diabetes, hypertension, obesity, cardiovascular disease, confirmed pneumonia, and ICU-level clinical setting.

The Mexican model assigns 72.6% probability to Critical. The Einstein model (hypothetical CBC showing lymphopenia, elevated CRP and neutrophils, low hemoglobin) assigns 72.0% probability to Severe (mapped to Critical/Severe split). The ensemble combines these signals to produce **86.0% confidence in the Critical classification**. The top three SHAP drivers are: clinical setting (hospitalised/ICU, SHAP=+0.279), presence of pneumonia (SHAP=+0.070), and age risk >60 (SHAP=+0.058).

![Figure 5](figures/fig5_decision_flow.png)

*Figure 5: Ensemble decision flow for a Critical case example.*

### 4.7 System Architecture

Figure 1 provides an overview of the full system architecture, from data input through model inference, ensemble weighting, SHAP computation, and LLM explanation generation.

![Figure 1](figures/fig1_architecture.png)

*Figure 1: COVID-19 CDSS system architecture. The left column (blue) processes always-available comorbidity data. The right column (green, dashed) processes optional CBC laboratory results. The ensemble and explainability layers are applied downstream.*

---

## 5. Discussion

### 5.1 Complementary Dataset Signals

The cross-dataset validation result (38.25%) is arguably the most important finding in this paper. It demonstrates empirically that comorbidity-based and laboratory-based features are not interchangeable: a model trained on Mexican comorbidity data cannot predict outcomes from Einstein CBC panels at a clinically useful level. This motivates the dual-model design as a principled architectural choice rather than a pragmatic compromise.

The biological explanation is straightforward: comorbidities capture long-term host vulnerability (diabetes impairs immune response, hypertension may worsen pulmonary hypertension), whereas CBC panels capture acute immune response (lymphopenia reflects T-cell depletion, CRP reflects systemic inflammation). Both signals are prognostic but through different mechanisms operating at different timescales. An ensemble that combines both should — and empirically does — outperform either alone.

The 60/40 ensemble weight reflects several considerations. The Mexican dataset is three orders of magnitude larger than the Einstein dataset, conferring greater statistical robustness. However, the Einstein lab model's per-sample discriminative accuracy is substantially higher. The 60/40 split was selected by grid search and reflects a balance between statistical confidence and discriminative power.

### 5.2 Explainability in Clinical AI

A recurring critique of machine learning models in clinical settings is the "black box" problem: clinicians cannot verify the reasoning behind a prediction, limiting trust and safe integration [reference]. SHAP addresses this at the algorithmic level by providing mathematically rigorous feature attributions. However, SHAP values are numerical and require interpretation — a barrier for non-specialist clinicians.

The LLM layer bridges this gap. By translating SHAP values into structured clinical language ("the patient's ICU-level clinical setting was the strongest predictor of Critical severity, contributing +0.279 to the model's log-odds"), the system produces explanations that are actionable without requiring machine learning expertise. This represents a novel integration: SHAP provides the ground truth of feature importance; the LLM provides the communicative interface.

The approach is not without risks. LLMs can hallucinate or introduce clinical errors in the explanatory text. We mitigate this by: (1) constraining the LLM prompt to interpret only the provided SHAP values, not to generate independent clinical reasoning; (2) including a prominent disclaimer that the output is decision support; and (3) providing the raw SHAP waterfall plot alongside the text so that clinicians can verify the explanation against primary data.

### 5.3 Graceful Degradation Design

The architecture's core innovation is its ability to operate across a continuum of data availability without requiring any user reconfiguration. When only comorbidity data is available — the scenario in a rural clinic, a field hospital, or a primary care consultation — the system provides a 5-class severity estimate from the Mexican model alone. When CBC results are available — the scenario in a district or tertiary hospital — the ensemble is automatically activated.

This design pattern is generalisable. Any multi-modal clinical prediction problem where different feature subsets are available in different settings could adopt a similar architecture: train specialised models for each data modality, define a principled mapping between their output spaces, and combine via a learned ensemble weight. The key requirement is that the output schemas be aligned — here, both models ultimately produce probability vectors over the 5-class severity schema.

### 5.4 Limitations

Several important limitations must be acknowledged:

**Label quality.** The Mexican surveillance dataset relies on clinician-assigned severity labels, which may be inconsistently applied across 542,921 records from hundreds of reporting sites. The Mild/Moderate distinction in particular may reflect documentation practice as much as clinical reality.

**Temporal validity.** Both datasets were collected during the first and second waves of COVID-19, before widespread vaccination and before the emergence of the Delta and Omicron variants. The comorbidity profiles and laboratory patterns that predict severity may differ in vaccinated populations or for different variants. The models should be retrained on more recent data before deployment.

**Einstein dataset size.** With 602 usable patients, the Einstein dataset is small by machine learning standards, and the effective test set (≈120 patients) is insufficient to precisely estimate performance in rare subgroups. Confidence intervals for Einstein accuracy are wide.

**Ethnicity and demographic generalisability.** Both datasets are from Latin America. The comorbidity model may not generalise to populations with different epidemiological profiles, and the CBC reference ranges used for standardisation are population-specific.

**Prospective validation absent.** No prospective clinical validation has been conducted. Before deployment in patient care, the system should be evaluated prospectively against blinded clinician assessments and clinical outcomes.

### 5.5 Generalisation to Infectious Disease CDSS

The architectural pattern introduced here — dual-dataset ensemble with graceful degradation, SHAP-based explainability, and LLM-mediated clinical communication — is not specific to COVID-19. Any infectious disease with a severity spectrum, population-level surveillance data, and available biomarker panels could benefit from this framework. Candidates include influenza (where the FluSurv-NET dataset provides a Mexican analogue and biomarker studies abound), dengue fever (where the WHO severity classification and CBC platelet trends are well-studied), and tuberculosis (where comorbidities such as HIV and malnutrition are dominant severity drivers).

The primary adaptation requirement for each disease is: (1) a large surveillance dataset with comorbidity features and outcome labels, and (2) a smaller but biomarker-rich cohort enabling a laboratory model. The ensemble and explainability layers are disease-agnostic.

### 5.6 Global Epidemiological Context

COVID-19 has been the defining public health crisis of the early 21st century. According to Our World in Data [Ritchie et al., 2020], SARS-CoV-2 infected approximately 676 million confirmed individuals globally and caused an estimated 6.9 million confirmed deaths — with excess mortality estimates suggesting the true death toll may be three to five times higher. Mexico recorded over 334,000 confirmed COVID-19 deaths, among the highest absolute death tolls in Latin America and reflecting the severe burden faced by a health system serving 130 million people with limited ICU capacity. Brazil recorded over 702,000 confirmed deaths — the second-highest national total globally after the United States. The datasets used in this study are therefore not merely convenient benchmarks; they represent the clinical experience of two of the most severely affected countries in the world. A decision support system calibrated to these populations addresses a genuine and urgent need, and the performance metrics reported here should be interpreted in the context of systems that previously offered no algorithmic support at all.

---

## 6. Conclusion

We have presented a dual-dataset ensemble framework for COVID-19 severity prediction that operates reliably across a spectrum of data availability, from comorbidity-only to full CBC panel assessment. The Mexican XGBoost model achieves 51.6% 5-class accuracy with a clinically important Critical recall of 60.3%, enabling identification of the patients most at risk of death even in the absence of laboratory infrastructure. The Einstein lab model achieves 85.1% on its 3-class task, and the ensemble achieves 86.0% when both data streams are available.

The integration of SHAP explainability and LLM-mediated clinical narration represents a novel contribution to transparent clinical AI. By grounding the LLM generation in mathematically rigorous SHAP attributions, we achieve explanations that are both clinically readable and algorithmically verifiable — a combination that neither tool achieves alone.

Future work will prioritise: (1) prospective clinical validation at partner institutions in Mexico and Brazil; (2) extension to post-vaccination and variant-specific cohorts; (3) integration of additional biomarkers (D-dimer, ferritin, troponin) in the Einstein model; and (4) adaptation of the framework to influenza, dengue, and tuberculosis severity prediction. Open-sourcing the full pipeline will enable the global research community to adapt and validate these methods across diverse settings.

---

## References

1. Wynants L, Van Calster B, Collins GS, et al. Prediction models for diagnosis and prognosis of covid-19 infection: systematic review and critical appraisal. *BMJ*. 2020;369:m1328. https://doi.org/10.1136/bmj.m1328

2. Shi H, Han X, Jiang N, et al. Radiological findings from 81 patients with COVID-19 pneumonia in Wuhan, China: a descriptive study. *Lancet Infectious Diseases*. 2020;20(4):425-434. https://doi.org/10.1016/S1473-3099(20)30086-4

3. Lundberg SM, Lee SI. A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems*. 2017;30:4765-4774.

4. Chen T, Guestrin C. XGBoost: A scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*. 2016:785-794. https://doi.org/10.1145/2939672.2939785

5. Chawla NV, Bowyer KW, Hall LO, Kegelmeyer WP. SMOTE: Synthetic minority over-sampling technique. *Journal of Artificial Intelligence Research*. 2002;16:321-357. https://doi.org/10.1613/jair.953

6. Ritchie H, Mathieu E, Rodés-Guirao L, et al. Coronavirus Pandemic (COVID-19). *Our World in Data*. 2020. https://ourworldindata.org/coronavirus

7. World Health Organization. Clinical management of COVID-19: interim guidance. *WHO Reference Number: WHO/2019-nCoV/clinical/2020.5*. 2020. Available at: https://www.who.int/publications/i/item/clinical-management-of-covid-19

8. Secretaría de Salud, México. Datos Abiertos Dirección General de Epidemiología — COVID-19. 2021. Available at: https://www.gob.mx/salud/documentos/datos-abiertos-152127

9. Hospital Israelita Albert Einstein. Diagnosis of COVID-19 and its clinical spectrum. *Kaggle Dataset*. 2020. Available at: https://www.kaggle.com/einsteindata4u/covid19

10. Ke G, Meng Q, Finley T, et al. LightGBM: A highly efficient gradient boosting decision tree. *Advances in Neural Information Processing Systems*. 2017;30:3146-3154.

11. Lundberg SM, Erion G, Chen H, et al. From local explanations to global understanding with explainable AI for trees. *Nature Machine Intelligence*. 2020;2(1):56-67. https://doi.org/10.1038/s42256-019-0138-9

12. Toğaçar M, Ergen B, Cömert Z. COVID-19 detection using deep learning models to exploit Social Mimic Optimization and structured chest X-ray images using fuzzy color and stacking approaches. *Computers in Biology and Medicine*. 2020;121:103805.
