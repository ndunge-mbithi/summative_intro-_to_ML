# Final Quality Assurance and Acceptance Report

**Project:** Predicting Diabetes Status from Population Health Indicators Using Traditional Machine Learning and Deep Learning
**Author:** Ogayo Andrew · **Audit date:** 18 October 2026

## 1. Project overview
The project delivers an end-to-end, reproducible classification study on the raw CDC BRFSS 2015 survey:

- a modular `src/` package with unit tests;
- an executed Colab-compatible notebook (EDA → preprocessing → 17 experiments → evaluation → explainability → export);
- an IEEE-style report (PDF);
- saved model artifacts;
- TensorBoard and JSONL experiment logs;
- a presentation script.

The notebook was executed three times from a clean state, and every metric in the 17-experiment table was identical across runs (maximum absolute difference 0.0).

## 2. Dataset
- **Source:** CDC BRFSS 2015 LLCP public-use file, https://www.cdc.gov/brfss/annual_data/annual_2015.html (codebook dated 23 Aug 2016).
- **Raw sample:** 441,456 records, 330 variables — verified against the codebook together with all seven `DIABETE3` frequency cells (57,256 / 3,608 / 372,104 / 7,690 / 598 / 193 / 7).
- **Analysis sample:** 440,658 respondents with a usable label.
- **Provenance note:** the sandbox could not reach cdc.gov, so the file was obtained from a public GitHub mirror (bcapistrant/brfssR). Its integrity was established by the codebook match. File SHA-256: `bfe9e62977cfc5183e51c3e8bdb5193510995cc3c21b225e568f537ad300b1b9`. In Colab, the loader downloads directly from CDC.

## 3. Research question
How effectively can traditional machine learning and deep learning models classify diabetes status from population health indicators, and how do their predictive performance, interpretability, computational efficiency and error patterns differ?

## 4. Modelling approaches
- **Scikit-learn:** multinomial logistic regression, random forest and histogram gradient boosting, with RandomizedSearchCV tuning.
- **TensorFlow/Keras:** Sequential MLPs (baseline and regularised, with learning-rate, batch-size and class-weight variants) and a Functional API wide-and-deep network, all fed by `tf.data` pipelines.

## 5. Experiments executed
17 experiments (E01–E17) were executed. Each is logged in `reports/experiment_results.csv`, in `reports/experiment_log.jsonl` (with run ID and timestamp) and in `logs/tensorboard/<run_id>`.

## 6. Final comparison (held-out test set, n = 66,099)

| Model | Macro ROC-AUC | Macro AP | Macro F1 (tuned) | Diabetes recall | Diabetes precision |
|---|---|---|---|---|---|
| E01 Logistic regression | 0.7942 | 0.4690 | 0.4624 | 0.502 | 0.394 |
| **E07 Gradient boosting (tuned)** — selected | 0.8004 | **0.4790** | **0.4683** | 0.500 | **0.420** |
| E16 Functional wide-and-deep | **0.8005** | 0.4775 | 0.4657 | 0.492 | 0.418 |

## 7. Major findings and limitations
- **Similar ceiling for both paradigms.** Boosting is best on average precision, the wide-and-deep network ties on ROC-AUC, and logistic regression trails by about 0.01 AP at roughly 1/16 of the training time.
- **Decision rules matter more than algorithms.** Diabetes sensitivity is 0.17 at a threshold of 0.5 and 0.85 at the validation-selected F2 threshold, which flags 39% of respondents.
- **Prediabetes is not identifiable** (AP 0.036 vs 0.017 prevalence).
- **Error and subgroup analysis.** Missed cases resemble non-diabetic respondents, and false positives carry high-risk profiles. Several predictors reflect reverse causation. Sensitivity varies strongly by age and income under a single threshold.
- **Limitations:** self-reported labels; cross-sectional data; unweighted, sample-level evaluation (weighted metrics given as a check, no design-based variance); hyper-parameter searches on a 100k subsample; a single survey year; no calibration analysis.

## 8. Links

| Link | Status | Action required |
|---|---|---|
| GitHub repository | **Not yet created** | Create an empty repository (e.g. `healthcare-classification-project`) on your GitHub account. Then either push the local repository (`git remote add origin <url> && git push -u origin main`) or give Claude the repository name so it can push with your authorisation. |
| Google Drive folder | **Not yet created** | Upload the prepared `drive_deliverables/` bundle (104 MB) to a new Drive folder and set sharing to "Anyone with the link can view". Claude can do this through the connected Google Drive with your go-ahead, though the 97 MB data ZIP may need a manual upload. |
| Video recording | **Not yet recorded** | Record the 5–10 minute presentation following `reports/presentation_script.md`, with your face on camera. Upload it and verify the link plays in a private window. |

After the links exist, paste them into `report/report.tex` (`\GitHubURL`, `\DriveURL`, `\VideoURL`) and the README, then rebuild with `python report/build_report.py`.

## 9. Rubric checklist

### Problem definition and dataset
| Requirement | Status | Evidence |
|---|---|---|
| Healthcare problem clearly defined | Verified | Report §I; notebook §1 |
| Aligned with the author's mission | Verified | Mission stated and linked to the sub-Saharan Africa diagnosis gap (IDF Africa [2], Kenya STEPS [3]) in the abstract, report §I, §V-D and §VI, and notebook §1 and §13 |
| Supported by credible evidence | Verified | IDF Atlas [1], BRFSS documentation, literature review |
| Dataset relevant and documented | Verified | Notebook §2; report §III-A, Table 1 |
| Raw dataset ≥ 20,000 observations | Verified | 441,456 records, codebook-verified |
| Valid categorical target | Verified | `DIABETE3` → 3 classes; mapping documented and unit-tested |
| Dataset limitations explained | Verified | Report §V-C, §V-D; README |

### Literature review
| Requirement | Status | Evidence |
|---|---|---|
| ≥ 10 scholarly sources | Verified | 24 references (21 peer-reviewed or official, 1 preprint, 2 software papers) |
| All references real and relevant | Verified | Each checked against publisher, PubMed Central, CDC, JMLR, USENIX, arXiv or OpenAlex records. Zafar et al. was cited from its PMC abstract and metadata; its OpenAlex record could not be read (rate-limited). |
| IEEE citations correct | Verified | Numbered by first citation; `build_report.py` fails on any missing or uncited reference |
| Compares and critiques previous work | Verified | Report §II (target definitions, derivative data, resampling, implausible AUCs) |
| Specific research gap established | Verified | Report §II-E |

### Preprocessing and feature engineering
| Requirement | Status | Evidence |
|---|---|---|
| Missing values handled appropriately | Verified | Special codes → NaN; median + indicator imputation inside pipeline; informativeness checked |
| Outliers investigated | Verified | BMI extremes investigated (prevalence 20.9% above 60); train-fitted winsorising, no deletion |
| Encoding and scaling justified | Verified | Scaling for LR/NN only; E02 ablation shows its effect |
| Feature engineering documented | Verified | Six features, domain rationale, correlations |
| Feature selection avoids leakage | Verified | No data-driven selection; module variables excluded to prevent label leakage; all fitting on training split |
| Preprocessing reproducible | Verified | sklearn pipelines; 9 passing unit tests; identical reruns |
| Preprocessing effects investigated empirically | Verified | E02 (scaling), E06/E07/E09/E17 (features), E03/E08/E15 (imbalance) |

### Model implementation and optimisation
| Requirement | Status | Evidence |
|---|---|---|
| Multiple scikit-learn models | Verified | LR, RF, gradient boosting |
| TensorFlow/Keras models | Verified | E10–E17 |
| Sequential and Functional APIs | Verified | E10–E15 Sequential; E16–E17 Functional wide-and-deep |
| `tf.data` pipeline | Verified | `make_tf_dataset`: cache, per-epoch shuffle (train only), batch, prefetch |
| ≥ 7 meaningful experiments executed | Verified | 17 |
| Hyper-parameter changes justified | Verified | Notebook §8–10 markdown; report §III-E/F |
| Optimisation traceable | Verified | Search results CSVs, JSONL log, TensorBoard |

### Experiment results table
| Requirement | Status | Evidence |
|---|---|---|
| All configurations and per-run observations | Verified | `experiment_results.csv` (with an observation column for all 17 runs); report Table 2 and Appendix A (hyper-parameters plus observations) |
| Splits and hyper-parameters documented | Verified | Columns `split`, `hyperparameters` |
| Actual measured metrics | Verified | Tables generated from CSV by `make_tables.py`; numbers audited against outputs |
| Clear experiment progression | Verified | Baselines → ablations → tuning → imbalance → architectures |
| Interpretations explain changes | Verified | Notebook interpretation cells; report §IV-B |
| TensorBoard logs support the table | Verified | One run directory per experiment |
| Reproducible by another engineer | Verified | Seeds, determinism, pinned requirements, identical repeated runs |

### Evaluation and error analysis
| Requirement | Status | Evidence |
|---|---|---|
| Learning curves | Verified | Fig. 5 (Keras), Fig. 9 appendix (boosting) |
| Confusion matrices | Verified | Fig. 3 and appendix, interpreted |
| ROC and PR analysis | Verified | Fig. 4, interpreted |
| Class-specific errors | Verified | Per-class metrics; prediabetes routing; missed and false-positive profiles |
| Over- and underfitting analysed | Verified | Search train-vs-CV gap, RF depth, learning curves, E12/E13 |
| Bias–variance tied to results | Verified | Report §V-A |
| Dataset limitations discussed | Verified | Report §V-C/D/E |
| Improvements justified | Verified | Report §V-E |

### Code quality and documentation
| Requirement | Status | Evidence |
|---|---|---|
| Notebook runs end to end | Verified | Three clean executions, 0 errors; sections numbered 1–14 in order |
| Dependencies documented | Verified | Install cell in the notebook, `requirements.txt`, `reports/environment.json`; Colab bootstrap clones the repository |
| Random seeds set | Verified | `set_global_seed` + `enable_op_determinism` |
| Modular, readable code | Verified | 7 `src/` modules |
| Markdown precedes code | Verified | 46 markdown cells |
| Actual outputs interpreted | Verified | 22 interpretation cells written after execution |
| Model artifacts can be loaded | Verified | Reload checks in notebook and in a fresh process |
| GitHub repository documentation | Verified (locally) | README, `.gitignore`, local commit. **Publishing is incomplete — see §8.** |
| Raw data excluded from Git | Verified | `.gitignore`; no data files tracked |

### Written report
| Requirement | Status | Evidence |
|---|---|---|
| 3,500–5,000 words | Verified | 4,955 body words (`report/word_count.txt`) |
| Six required sections | Verified | I–VI + references + appendix |
| IEEE citation formatting | Verified | Numbered style; IEEE conference layout using standard LaTeX packages (`IEEEtran.cls` was unavailable offline) |
| Figures and tables integrated | Verified | 5 tables and 5 figures referenced in text; more in appendix |
| Reflects actual implementation | Verified | Methodology mirrors code; tables generated from results |
| Critical analysis present | Verified | §V |
| Natural, professional, first person | Verified | First-person "I" throughout; no "we" |
| All required links included and verified | **Incomplete** | Placeholders clearly marked "link pending"; see §8 |

### Video presentation
| Requirement | Status | Evidence |
|---|---|---|
| Duration, presenter on camera, content | **Incomplete** | Script and recording checklist prepared (`reports/presentation_script.md`); recording must be made by the author |

## 10. Remaining actions
1. **GitHub:** create the repository and push.
2. **Drive:** upload the bundle and set sharing to "Anyone with the link can view".
3. **Video:** record and upload it.
4. **Report:** insert the three verified links into `report/report.tex` and the README, then rebuild the PDF.

Until these are done, the deliverable is complete in content but not in publication.

**Academic integrity note.** Most of the report and notebook prose was drafted with an AI assistant. The rubric caps report quality for AI-detected text above 20%, so the author should rewrite the prose in their own words and disclose AI assistance as ALU policy requires.
