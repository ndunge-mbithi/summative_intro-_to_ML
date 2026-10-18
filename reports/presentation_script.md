# Video Presentation: Outline and Demonstration Script

**Target length:** 8–9 minutes (limit 5–10). **Format:** presenter on camera throughout (face visible, picture-in-picture over a screen share), with live demonstration of the executed notebook in Jupyter or Colab.

**Before recording:**
- Open `notebooks/diabetes_classification.ipynb` with outputs visible.
- Open `reports/figures/` and `reports/experiment_results.csv`.
- Optionally run `tensorboard --logdir logs` in a second tab.

All numbers below come from the executed notebook. Speak them in your own words rather than reading line by line.

---

## 0:00–0:45 · The problem (camera full-screen)

> "Hi, I'm Ogayo Andrew. About one in ten adults worldwide has diabetes: the IDF estimated 537 million people in 2021. Many cases, and most prediabetes, go undiagnosed for years. My question was whether everyday survey indicators — blood pressure, BMI, age, how people rate their health — can classify diabetes status. And whether deep learning beats traditional machine learning at it. To be clear from the start: this is screening research, not a diagnostic tool."

## 0:45–1:45 · Dataset and research question (screen: notebook section 2 and the verification table)

- **Source and size:** CDC BRFSS 2015, the raw survey with 441,456 respondents, not the cleaned 253k Kaggle version.
- **Provenance:** Show the verification output. The record count and all seven `DIABETE3` frequency cells match the official codebook exactly, and the loader refuses mismatched data.
- **Target:** three classes — no diabetes, prediabetes, diabetes. Explain why I did not merge prediabetes into diabetes, as many papers do, and why gestational-only (code 2) maps to "no diabetes".
- **Class balance:** show Fig. 1. 85.3% / 1.75% / 13.0%, so accuracy is meaningless: predicting "no diabetes" for everyone is already 85% accurate.
- **Research question:** read it once, in one sentence.

## 1:45–2:45 · Preprocessing and feature engineering (screen: sections 5–6)

- **Special codes:** 7/77 = don't know and 9/99 = refused become missing values, not numbers; 88 = none becomes 0 days. Mention the cholesterol skip-pattern fix (TOLDHI2).
- **Missing data:** income is missing for 18%. I keep everyone, using median imputation plus missing-value indicators, all fitted inside the training pipeline only.
- **Outliers:** BMI up to 99.95. I winsorise rather than delete, because BMI > 60 has 21% diabetes prevalence, so it is signal, not noise.
- **Engineered features:** six of them (cardiometabolic count, age × BMI, and others). They looked promising on correlations but added only +0.0004 average precision. Say why: trees can build those interactions themselves.
- **Split:** stratified 70/15/15; the test set is touched once.

## 2:45–4:15 · Models and experiments (screen: section 8–10 cells, then the experiment table and Fig. 7)

- **Traditional models:** logistic regression, random forest, histogram gradient boosting.
  - E02: no scaling means the solver doesn't converge, takes about five times longer, and every metric drops.
  - RF baseline: trees 39 levels deep with about 13,000 leaves, which means high variance.
  - Randomized search: the winner was the *most regularised* configuration (small trees, strong L2). Other candidates hit 0.80 on training folds with no gain on held-out folds, which is the bias–variance trade-off live.
- **Keras models:** Sequential baseline (32 units), regularised 128-64-32 with batch norm, dropout and L2, plus learning-rate, batch-size and class-weight variants. Then a Functional wide-and-deep model.
  - Explain why wide-and-deep needs the Functional API: two branches that are summed.
  - Explain why I designed it this way: logistic regression already captured most of the signal.
  - Show the `tf.data` pipeline cell: shuffle with reshuffling, batch and prefetch on train only.
- **Learning curves (Fig. 5):** lr 1e-2 is unstable and peaks at epoch 2; lr 1e-4 is still improving at 60 epochs; there is no overfitting anywhere.
- **Results (Fig. 7):** 17 experiments, and all non-reweighted models sit within about 0.011 macro average precision. Point out that class weighting moves F1 and balanced accuracy but not ranking quality.

## 4:15–5:45 · Final comparison on the test set (screen: test table, confusion matrix, ROC/PR curves)

- **Overall:** tuned boosting reached macro ROC-AUC 0.800 and AP 0.479. The wide-and-deep network was tied (0.800 / 0.478), and logistic regression close behind (0.794 / 0.469).
- **Decision rule:** argmax found fewer than 1 in 5 diabetes cases. Validation-tuned class weights find half: 4,296 of 8,588.
- **Confusion matrix:** prediabetes is essentially unsolved — 54 of 1,154 correct, the rest split between the neighbouring classes.
- **ROC vs PR:** prediabetes ROC-AUC looks fine (0.72), but average precision is 0.036. This is a nice demonstration of why I used PR curves under imbalance.
- **Screening threshold:** the F2 threshold of 0.112 gives sensitivity 0.85 and NPV 0.97, but flags 39% of people. Explain the false-negative vs false-positive costs, and that this is a policy decision, not a model decision.

## 5:45–7:00 · Errors, explainability and fairness (screen: section 12 tables, SHAP figure, subgroup table)

- **Missed cases** look like non-diabetic people: BMI 28.8 vs 34.4 for detected cases, with fewer comorbidities.
- **False positives** have high-risk profiles; many may be *undiagnosed* diabetes that the self-reported label can't see.
- **Top features:** BMI, general health, age and blood pressure. Point out reverse causation: a recent check-up and cholesterol testing *raise* predicted risk because people with diabetes see doctors more. This is not causal.
- **Subgroups:** recall is only 0.21 for under-45s, but 0.63 in the lowest income band. Black non-Hispanic respondents have the highest prevalence and the lowest AUC (0.813). One threshold does not serve everyone equally, so I call these descriptive disparities, not proven unfairness.

## 7:00–8:00 · Selected model, limitations and future work (camera full-screen again)

- **Selected model:** tuned gradient boosting, for the best threshold-free scores and stable training. If transparency is the priority, logistic regression is within 0.01 AP at a fraction of the cost.
- **Answer to the research question:** both paradigms hit the same ceiling. The limit is in the survey data and labels, not the algorithm. My hypothesis held in direction but with a small effect.
- **Limitations:** self-reported labels, cross-sectional data, a single year, and unweighted evaluation (weighted metrics shown as a check).
- **Future work:** laboratory-validated labels (NHANES HbA1c), calibration, group-specific thresholds, and later survey years.

## 8:00–8:30 · Close

> "Everything here is reproducible: the GitHub repository has the code, tests and the executed notebook; the Drive folder has the models and report. Running the notebook twice from scratch gives identical results. Thank you."

---

### Recording checklist
- [ ] Camera on, face visible for the whole recording; good lighting; quiet room
- [ ] Notebook scrolled to each section before you start talking about it
- [ ] Show at least: verification table, experiment table/Fig. 7, learning curves, confusion matrix, ROC/PR curves, SHAP, subgroup table
- [ ] Duration between 5 and 10 minutes
- [ ] Upload (YouTube unlisted or Drive), open the link in a private window to verify it plays, then paste it into `report/report.tex` (`\VideoURL`) and the README
