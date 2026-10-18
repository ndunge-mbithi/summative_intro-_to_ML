# Predicting Diabetes Status from Population Health Indicators

**Traditional machine learning vs. deep learning on the CDC BRFSS 2015 survey**

Author: Ndunge Mutheu Mbithi (BSc Software Engineering, African Leadership University)


## Overview

**Mission alignment.** My mission is to revolutionise healthcare in sub-Saharan Africa through technology, creating accessible, equitable and scalable solutions for underserved rural communities. In Africa, 54% of diabetes is undiagnosed (IDF Africa, 2021), and in rural areas laboratory testing is scarce. This project measures how much cheap, questionnaire-style indicators can reveal about diabetes status, which models extract that information best, and what each decision rule costs in missed cases and false alarms. It uses the large, well-documented US BRFSS as a testbed whose lessons must be revalidated on local African data.


I compare logistic regression, random forests, gradient boosting and TensorFlow/Keras neural networks (Sequential and Functional API) for classifying self-reported diabetes status into **no diabetes / prediabetes / diabetes**. The data are the raw 2015 Behavioral Risk Factor Surveillance System survey (441,456 respondents), not the cleaned 253,680-row Kaggle/UCI derivative.

**Research question.** How effectively can traditional machine learning and deep learning models classify diabetes status from population health indicators, and how do their predictive performance, interpretability, computational efficiency and error patterns differ?

**Objectives.**
1. Verify a large, authoritative open dataset and define a faithful categorical target.
2. Build a reproducible, leakage-safe preprocessing and feature-engineering pipeline.
3. Compare Scikit-learn classifiers and Keras networks (Sequential, Functional, `tf.data`).
4. Run and track more than seven systematic experiments: hyper-parameters, features, class imbalance and regularisation.
5. Evaluate with imbalance-aware metrics, decision-rule tuning, error analysis, explainability and subgroup checks.

## Results summary

All numbers come from the executed notebook. A second run from a clean state reproduced every experiment metric exactly.

| Test set (66,099 respondents) | Logistic regression (E01) | Gradient boosting, tuned (E07) | Functional wide-and-deep + engineered Keras (E17) |
|---|---|---|---|
| Macro ROC-AUC | 0.794 | **0.801** | 0.800 |
| Macro average precision | 0.469 | **0.479** | 0.478 |
| Macro F1 (tuned decision weights) | 0.462 | **0.469** | 0.468 |
| Diabetes recall / precision | 0.502 / 0.394 | **0.546 / 0.401** | 0.481 / 0.421 |
| Prediabetes recall / precision | 0.072 / 0.044 | 0.100 / 0.042 | 0.056 / 0.053 |

- **17 tracked experiments** were run (`reports/experiment_results.csv`). Every non-reweighted model lands within about 0.011 validation macro average precision, so the data, not the algorithm, set the ceiling.
- **Gradient boosting** (most regularised configuration from the randomized search) had the best threshold-free scores, the **wide-and-deep network** tied on ROC-AUC, and **logistic regression** trailed by about 0.01 AP while training in about 5 s.
- **The decision rule matters most.** Diabetes sensitivity is 0.17 at the default threshold and 0.85 at a validation-selected F2 threshold, which flags 39% of respondents (PPV 0.28).
- **Prediabetes is essentially not identifiable** from these self-reported indicators (average precision 0.036 vs 0.017 prevalence).
- **Explainability and subgroups.** Explanations point to BMI, general health, age and hypertension, plus reverse-causation signals such as check-up recency. Detection varies widely by age, income and race/ethnicity under a single threshold.


## Dataset

| | |
|---|---|
| Source | CDC BRFSS 2015 — https://www.cdc.gov/brfss/annual_data/annual_2015.html |
| File | `LLCP2015.XPT` (SAS transport, 1.17 GB unzipped; CDC zip `LLCP2015XPT.zip`, 96.5 MB) |
| Records | 441,456 (330 variables); 440,658 with a usable target |
| Target | `DIABETE3` → No diabetes (codes 3, 2) / Prediabetes (4) / Diabetes (1); codes 7, 9 and blank removed |
| Design | survey weight `_LLCPWT`, strata `_STSTR`, PSU `_PSU` |
| Licence | US federal public-use data without personal identifiers. Raw data are **not** committed to this repository. |

### Data access

```bash
pip install -r requirements.txt
python scripts/download_data.py                 # downloads from CDC, verifies against the codebook, caches Parquet
# or, if you already have the file:
python scripts/download_data.py --xpt /path/to/LLCP2015.XPT
```

The loader (`src/data_loading.py`) checks the file against the published 2015 codebook: 441,456 records, 330 variables, and every `DIABETE3` frequency (57,256 / 3,608 / 372,104 / 7,690 / 598 / 193 / 7). It raises an error on any mismatch. The file used for the reported results has SHA-256 `bfe9e62977cfc5183e51c3e8bdb5193510995cc3c21b225e568f537ad300b1b9`.

## Repository structure

```text
healthcare-classification-project/
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   └── diabetes_classification.ipynb   # executed notebook: EDA → experiments → evaluation → export
├── src/
│   ├── data_loading.py                 # CDC download, chunked XPT reading, codebook verification, cache
│   ├── preprocessing.py                # codebook recodes, target mapping, ColumnTransformer, splits
│   ├── feature_engineering.py          # stateless engineered risk features
│   ├── traditional_models.py           # LR / RF / gradient boosting pipelines, randomized search
│   ├── deep_learning_models.py         # tf.data pipelines, Sequential and Functional (wide-and-deep) models
│   ├── evaluation.py                   # metrics, decision-weight tuning, screening threshold, plots, subgroups
│   ├── experiment_tracking.py          # JSONL + CSV + TensorBoard experiment tracker
│   └── utils.py                        # seeds, environment report
├── scripts/
│   ├── download_data.py                # official data access script
│   ├── build_notebook.py               # regenerates the notebook from source cells
│   └── notebook_interpretations.py     # interpretation text written after reading executed outputs
├── reports/
│   ├── figures/                        # all figures used in the report
│   ├── experiment_results.csv          # one row per executed experiment (validation metrics)
│   ├── experiment_log.jsonl            # full configs with run IDs and timestamps
│   ├── test_results.csv, screening_results.csv, subgroup_metrics.csv, ...
│   ├── report.pdf                      # IEEE-style written report
│   └── presentation_script.md          # video outline and demonstration script
├── report/                             # LaTeX source of the written report
├── models/model_metadata.json          # model artifacts themselves are distributed via Google Drive
├── logs/tensorboard/                   # TensorBoard event files for every run
└── tests/test_preprocessing.py         # unit tests for recodes, engineering and leakage safeguards
```

## Environment setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -q                     # 9 tests
```

The results were produced with Python 3.13 on a 2-core CPU (no GPU). Seeds are fixed (`SEED = 42`) for Python, NumPy, TensorFlow and every estimator, and TensorFlow deterministic ops are enabled.

## Running the notebook

```bash
jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=7200 notebooks/diabetes_classification.ipynb
```

On **Google Colab**, set `REPO_URL` to this repository's URL in the first code cell, or upload the repository. The loader then downloads the data directly from CDC. A full run takes roughly 45–60 minutes on 2 CPU cores, most of it in the randomized searches.

## Experiment tracking

Every run writes one row to `reports/experiment_results.csv` and a full configuration record (run ID, UTC timestamp, hyper-parameters, preprocessing, imbalance strategy, training time, validation metrics) to `reports/experiment_log.jsonl`. TensorBoard event files are written to `logs/tensorboard/<run_id>`, and per-epoch Keras curves to `logs/keras/<experiment>/`.

```bash
tensorboard --logdir logs
```

## Evaluation methods

- Primary selection metric: **validation macro average precision** (one-vs-rest PR-AUC), which is threshold-free and imbalance-aware.
- Also reported: accuracy, balanced accuracy, macro and per-class precision/recall/F1, macro ROC-AUC, confusion matrices, ROC and PR curves.
- Decision rule: per-class probability weights tuned on validation (macro-F1). Screening view: diabetes-vs-not threshold chosen on validation by F2.
- Error analysis, permutation importance, TreeSHAP, logistic coefficients, and subgroup metrics by sex, age, race/ethnicity and income.
- Survey-weighted (`_LLCPWT`) test metrics as an approximate population view.

## Model export

The notebook saves, after model selection and test evaluation:

- `models/best_traditional_pipeline.joblib` — the full scikit-learn pipeline (feature engineering → preprocessing → model);
- `models/best_deep_model.keras` plus `models/deep_model_preprocessor.joblib`;
- `models/model_metadata.json` — features, classes, decision weights, screening threshold, test metrics, versions and data hash.

```python
import joblib, pandas as pd
from src.preprocessing import clean_brfss
pipe = joblib.load("models/best_traditional_pipeline.joblib")
X, y, meta = clean_brfss(raw_brfss_frame)        # raw BRFSS columns -> cleaned features
proba = pipe.predict_proba(X)                     # columns: no diabetes, prediabetes, diabetes
```

## Limitations

- Labels are **self-reported** "ever told" diagnoses. Undiagnosed diabetes is labelled "no diabetes", and prediabetes is heavily under-reported.
- The data are cross-sectional, so predictors may be consequences of diagnosis (reverse causation). Feature importance is not causal.
- The split and metrics are unweighted (sample-level). Weighted metrics are approximate and ignore strata.
- The data are from 2015 and US adults only, so transfer to other periods or countries (e.g. East Africa) is untested.
- Hyper-parameter searches used a 100k-row training subsample because of the CPU budget.

## Submission links

- GitHub repository: https://github.com/ndunge-mbithi/summative_intro-_to_ML
- Google Drive deliverables: **pending — add the actual folder URL after uploading and checking access**
- Video presentation: **pending — record, upload, and add the actual playable URL**
