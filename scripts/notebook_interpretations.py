"""Interpretation text for the notebook, written after reading the executed outputs.

Every number quoted here was read from an executed run of the notebook; the
final build re-executes the notebook and audit_interpretations.py checks the
quoted figures against the saved results files.
"""

INTERP: dict[str, str] = {
    "env": """
All experiments ran on a 2-core x86-64 CPU with 8.4 GB of RAM and no GPU. That constraint shapes two later decisions: I use histogram-based gradient boosting rather than the classic implementation, and I run the hyper-parameter searches on a 100,000-row training subsample. The training times in the experiment table are only comparable within this hardware.
""",
    "loading": """
The file matches the published 2015 codebook exactly: 441,456 records, 330 variables, and all seven `DIABETE3` frequency cells (including the 7 blank responses). Its SHA-256 also matches the reference copy I used, so anyone re-running this notebook can tell whether they are using byte-identical data. There are no duplicate respondents and no fully duplicated rows. Because `_PSU` never repeats within a state, every respondent is their own primary sampling unit, so a random split cannot place members of one cluster on both sides of the train/test boundary.
""",
    "target": """
After removing the 798 unusable labels, 440,658 respondents remain: 85.26% report no diabetes, 1.75% prediabetes and 12.99% diabetes. The survey-weighted diabetes share (10.5%) reproduces the codebook's weighted 10.48% for code 1. It is lower than the unweighted share because older adults, who have more diabetes, are over-represented in the raw sample relative to the population.

Two consequences follow for evaluation. First, a model that predicts "no diabetes" for everyone would already be 85% accurate, so accuracy cannot be the selection metric. Second, prediabetes is a 1-in-57 class, and self-reported prediabetes captures only people who were tested and told, so I expect it to be the weakest class for every model.
""",
    "missing": """
After converting special codes to missing, income has the most missing values (17.97%, mostly refusals). Next come the calculated diet and activity variables (vegetables 11.56%, fruit 9.88%, physical activity 8.59%) and BMI (8.21%). The diet and activity gaps arise largely from partially completed interviews.

The association between missingness and the outcome is weak: diabetes prevalence is 13.5% when income is missing versus 12.9% when it is reported, and 12.5% versus 13.0% for BMI. So missingness is unlikely to be a strong predictor. Still, deleting incomplete rows would remove about a fifth of the sample and shift its composition, which is why I keep everyone and impute with missingness indicators.
""",
    "distributions": """
Three patterns stand out:

* **BMI:** the BMI distribution of people with diabetes is shifted about three units to the right of those without. The prediabetes and diabetes curves almost coincide, an early sign that BMI cannot separate those two classes.
* **Age:** diabetes prevalence rises steeply with age, from about 1% at 18–29 to about 20% at 70–80. Prediabetes rises only gently, to about 2.5%.
* **Self-rated health:** this is the strongest single gradient. Diabetes prevalence is about 2% among those rating their health "excellent" and about 36% among those rating it "poor".

The self-rated health gradient is an association that runs both ways: living with diabetes is itself a reason to rate one's health poorly, so this variable partly reflects the outcome rather than predicting it.
""",
    "associations": """
The largest prevalence ratios belong to variables that measure contact with the health system or existing disease:

* Having had a cholesterol check in the past five years (14.9% vs 2.8% diabetes prevalence, ratio 5.3);
* High blood pressure (24.1% vs 5.4%);
* Difficulty walking (29.3% vs 9.6%);
* Kidney disease (36.4% vs 12.1%).

The cholesterol-check effect is unlikely to be causal; people with diabetes are monitored more often. Some protective-looking associations also need care. Heavy drinkers show lower prevalence (5.3% vs 13.5%), plausibly because people with chronic illness drink less ("sick quitters") rather than because alcohol protects.

The correlation heatmap shows clusters among general health, difficulty walking and physically unhealthy days, and between education and income. The tree-based permutation importances later will therefore share credit within these clusters.
""",
    "outliers": """
BMI ranges from 12.02 to 99.95. There are 313 values below 15, 1,457 above 60 and 509 above 80; the 0.1st and 99.9th percentiles are 15.22 and 81.67. BMIs above 80 are implausible and probably reflect height or weight reporting errors. However, respondents with BMI above 60 have a diabetes prevalence of 20.9%, against 13.0% overall, so the extreme tail carries real signal and is not just noise.

I therefore do not delete these respondents. Winsorising at the training-set 0.1st/99.9th percentiles caps the few implausible values without discarding anyone, and tree models are unaffected by it.
""",
    "engineering": """
On a 50,000-row preview, the engineered features with the strongest rank correlation to diabetes are the cardiometabolic comorbidity count (ρ = 0.316) and the age × BMI interaction (ρ = 0.307), followed by BMI category (0.214) and unhealthy days (0.112). The lifestyle score is only weakly related (0.063), and the access-barrier flag is essentially unrelated (−0.007).

Strong marginal correlations do not guarantee added value: each engineered feature is a deterministic function of inputs the models already receive. Experiments E06, E07, E09 and E17 test whether they help in practice.
""",
    "lr": """
* **E01 (baseline):** standardised logistic regression converges in 36 L-BFGS iterations. It reaches validation macro average precision of 0.4704 and macro ROC-AUC of 0.7888. With argmax decisions, accuracy is 0.858, but recall is 0 for prediabetes and only 0.18 for diabetes, which is exactly the failure mode the high accuracy hides.
* **E02 (unscaled):** removing scaling does not change the model class, but it changes the optimisation. The solver hit its 200-iteration limit and raised a convergence warning. Training took about five times longer (25.7 s vs 5.4 s), and the result is worse on every metric (macro AP 0.4632, ROC-AUC 0.7694). This is why I standardise inputs for gradient-based learners.
* **E03 (balanced class weights):** reweighting moves the decision boundary dramatically. Prediabetes recall rises to 0.31 and diabetes recall to 0.59, at the cost of accuracy falling to 0.645 and very low precision (0.028 for prediabetes, 0.346 for diabetes). Macro-F1 improves (0.397 → 0.423), but macro AP falls slightly (0.4704 → 0.4658). Reweighting mostly relocates the threshold rather than improving the ranking. A similar effect can be obtained post hoc by tuning decision weights on validation, without distorting the probability estimates.
""",
    "trees": """
* **E04 (random forest baseline):** fully grown trees have a mean depth of 38.9 and about 12,955 leaves, a sign that each tree memorises much of its bootstrap sample. Averaging contains the variance: macro AP (0.4780) beats logistic regression, but macro ROC-AUC (0.7862) is slightly lower. The forest ranks the extreme cases well but is less smooth in the middle of the distribution.
* **E05 (gradient boosting baseline):** boosting improves both threshold-free metrics (macro AP 0.4794, ROC-AUC 0.7918).
* **E06 (engineered features):** adding the six engineered features changes macro AP by only +0.0004 (0.4798). Trees can already form interactions such as age × BMI through successive splits, so hand-made combinations add little for them.

All three models still predict prediabetes for nobody under argmax decisions.
""",
    "tuning": """
The 16 gradient boosting candidates are tightly clustered: the top six cross-validated macro AP scores differ by only 0.003 (0.4720–0.4753). The objective is therefore flat around its optimum, and further search would yield little.

The winning configuration is the most conservative one: a low learning rate (0.036) with many iterations (587), small trees (16 leaves), large leaves (73) and strong L2 regularisation (4.34). Its training-fold score (0.625) is the closest of all candidates to its validation-fold score (0.475). Candidates with larger trees score up to 0.80 on the training folds without scoring better on held-out folds. This is a direct observation of the bias–variance trade-off: extra capacity is spent fitting noise.

* **E07 (tuned, refitted on all training data):** improves validation macro AP to 0.4814 and macro ROC-AUC to 0.7953, the best traditional result.
* **E08 (tuned + balanced weights):** repeats the E03 pattern. Diabetes recall rises to 0.656 and prediabetes recall to 0.284, but macro AP falls to 0.4686.
* **E09 (tuned random forest):** the search prefers shallow trees (depth 14) with few features per split (0.157 of the features). This raises ROC-AUC from 0.7862 to 0.7921 at the same macro AP (0.4782 vs 0.4780), because regularising the trees mainly smooths the ranking of mid-risk respondents.
""",
    "keras": """
* **E10 (Sequential baseline):** the 32-unit network (1,763 trainable parameters) early-stops at epoch 9 with almost no generalisation gap (training loss 0.3691 vs validation 0.3721). It reaches macro AP 0.4772, above logistic regression but below boosting.
* **E11 (regularised Sequential):** the 128-64-32 network trains for 29 epochs, with the best epoch at 23, and reaches macro AP 0.4791. Its training loss (0.3764) sits *above* its validation loss (0.3742) because dropout and L2 penalties are active only during training. The curves show no overfitting, so the extra capacity is not being wasted, but it also buys little.
* **E12 (learning rate 1e-2):** optimisation becomes unstable. Validation loss is lowest at epoch 2 and then oscillates, so early stopping halts at epoch 8, and macro AP drops to 0.4728.
* **E13 (learning rate 1e-4):** too slow for the epoch budget. Validation loss is still falling at epoch 60, the cap (best epoch 60), and training took about twice as long as E11 (216 s vs 112 s) for a similar macro AP (0.4793). Learning rate 1e-3 is the efficient middle ground.
* **E14 (batch size 4096):** with 8 times fewer updates per epoch, the model needs almost the full 60 epochs (best epoch 58) to reach the same macro AP (0.4791). Larger batches do not improve the solution here; they only change how quickly it is reached.
* **E15 (class weights):** the softened weights (0.625 / 4.37 / 1.60) raise diabetes recall from 0.155 to 0.500 and macro-F1 to 0.454, while macro AP stays essentially unchanged (0.4785). Even so, prediabetes is still never the argmax class.
* **E16/E17 (Functional wide-and-deep):** this is the best deep model (macro AP 0.4799, ROC-AUC 0.7956), essentially tied with E11 and E13. Engineered features (E17, 0.4797) again add nothing.

Across all architectures, validation macro AP stays within a narrow 0.473–0.480 band. That suggests the limit lies in the information the survey variables carry, not in network capacity.
""",
    "boost_curve": """
For the tuned boosting model, validation log-loss decreases monotonically and reaches its minimum (0.3696) at essentially the final iteration, while training loss ends at 0.3571. The small, stable gap (0.013) and the absence of a rising validation curve indicate mild variance but no overfitting: the strong regularisation selected by the search keeps 587 trees from memorising the data.
""",
    "comparison": """
The panels make the central result visible. Excluding the reweighted runs, every model lands within a macro average precision band of about 0.011 (0.4704 for logistic regression to 0.4814 for tuned boosting). ROC-AUC follows the same order: unscaled logistic regression and the reweighted models are lowest, and tuned boosting and the regularised networks are highest. Threshold-dependent metrics tell a different story. Macro-F1 and balanced accuracy are driven mainly by the imbalance strategy (E03, E08, E15), not by the model family, which shows that reweighting the loss shifts the decision threshold rather than improving the ranking.

The best traditional model (E07) edges out the best deep model (E16) by 0.0015 macro AP. The differences between the strong models are small, but they are consistent across the boosting variants. The ranking of model families is clear even where the absolute gaps are modest.
""",
    "selection": """
On validation macro average precision, the selected models are tuned gradient boosting (E07, 0.4814) and the Functional wide-and-deep network (E16, 0.4799). Tuning the decision weights on validation raises macro-F1 substantially for all three final candidates: E07 from 0.406 to 0.469, E16 from 0.399 to 0.469, and logistic regression from 0.397 to 0.465.

The chosen multipliers are similar across models: prediabetes 15–20 and diabetes 2.5–3, relative to 1 for no diabetes. In other words, a respondent should be called "prediabetes" or "diabetes" well before that class becomes the single most probable one. With the first, coarser grid, the prediabetes weight landed on the grid's upper edge (20). I widened the grid to 80 so that the optimum (15 for E07 and E16, 20 for E01) is now interior. The validation surface near the optimum is flat: the next-best weight pairs score within 0.002 macro-F1.
""",
    "test": """
On the untouched test set, the three final models are very close:

| | Macro ROC-AUC | Macro AP | Macro-F1 |
|---|---|---|---|
| Gradient boosting (E07) | 0.8004 | 0.4790 | 0.468 |
| Wide-and-deep (E16) | 0.8005 | 0.4775 | 0.466 |
| Logistic regression (E01) | 0.7942 | 0.4690 | 0.462 |

The test values agree with validation (E07 validation macro AP was 0.4814), so selecting on validation did not overfit it.

For the **diabetes** class, the tuned rule recovers half of the cases. Boosting finds 4,296 of 8,588 (recall 0.500, precision 0.420); logistic regression has a similar recall (0.502) but lower precision (0.394). The confusion matrix shows where the remaining cases go: 3,987 diabetes cases are labelled "no diabetes" and 305 "prediabetes". In the other direction, 5,549 people without diabetes are flagged as diabetic.

**Prediabetes** remains essentially unsolved. Boosting labels 54 of 1,154 cases correctly (recall 0.047, precision 0.048). It assigns 714 to "no diabetes" and 386 to "diabetes", so prediabetes cases are spread across both neighbouring classes. The survey-weighted metrics tell the same story (weighted macro ROC-AUC 0.828 for boosting), so the conclusions are not an artefact of the sample's age structure.
""",
    "curves": """
The ROC curves of the three models almost overlap. Diabetes-versus-rest AUC is 0.843 for boosting, 0.842 for wide-and-deep and 0.833 for logistic regression, and the curves for "no diabetes" are similar. The precision–recall curves are more revealing:

* **Diabetes:** average precision is 0.434 (boosting) against a prevalence baseline of 0.130. The models concentrate cases about three-fold, but precision falls below 0.5 once recall passes about 0.3.
* **Prediabetes:** average precision is only 0.036–0.037 against a prevalence of 0.017, even though its ROC-AUC (0.72) looks respectable.

The prediabetes result illustrates why ROC curves can flatter performance on rare classes: a moderate ROC-AUC co-exists with precision that never meaningfully leaves the floor.
""",
    "screening": """
At the default 0.5 threshold, the model flags only 3.8% of respondents and misses 83% of diabetes cases (sensitivity 0.168), although its flags are usually right (PPV 0.581). The validation-selected F2 threshold of 0.112 reverses the balance. On the test set it detects 7,273 of 8,588 cases (sensitivity 0.847), with specificity 0.680 and NPV 0.967, but it flags 38.9% of respondents. Fewer than three in ten flags are true cases (PPV 0.283), and there are 18,432 false positives for 1,315 missed cases.

Under survey weighting, sensitivity is 0.809 and 29.6% of the weighted population is flagged. The sample over-represents older adults, among whom flags are more common.

Whether this trade-off is acceptable is a policy question, not a modelling one. A follow-up HbA1c test is cheap, so an F2-style rule is defensible for outreach. Even so, a rule that flags a third of adults is only useful if a confirmatory test is affordable at that scale.
""",
    "errors": """
Missed diabetes cases look much more like people without diabetes than like detected cases:

| Mean value | Detected cases | Missed cases | No diabetes |
|---|---|---|---|
| BMI | 34.4 | 28.8 | 27.5 |
| Diagnosed cardiometabolic conditions | 2.34 | 1.27 | 0.79 |
| Self-rated health (5 = poor) | 3.80 | 2.87 | 2.44 |
| Age | 66.1 | 62.9 | 54.0 |

The false positives mirror this. People flagged as diabetic who report no diabetes have an average of 2.15 cardiometabolic conditions, versus 0.63 for true negatives. Their profile resembles diabetes, and some may well be undiagnosed cases, which the self-reported label cannot reveal. So most errors are not random noise. They arise where the survey indicators genuinely overlap: people with diabetes and few other conditions, and people without a diabetes diagnosis but with a high-risk profile.

Prediabetes illustrates the same point. Its cases are predicted as "no diabetes" 61.9% of the time and "diabetes" 33.4%, and only 4.7% are recognised. With these indicators, prediabetes is not a distinct region of feature space but an intermediate zone between its neighbours.
""",
    "explain": """
The three explanation methods agree on the core signals but rank them differently:

* **Permutation importance** (raw input features, including their effect through engineered features) ranks BMI first (drop in macro AP 0.0347), then general health (0.0287), age (0.0165), high blood pressure (0.0109) and high cholesterol (0.0088). All other features contribute less than 0.006 each. The model relies on a handful of variables.
* **SHAP**, computed on the transformed inputs, spreads that credit onto the engineered features the boosting model actually splits on. General health (mean |SHAP| 0.30) comes first, followed by the cardiometabolic count (0.25), age × BMI (0.21) and check-up recency (0.16). BMI's own SHAP value is modest (0.08) because the age × BMI interaction carries much of its effect. This shows how correlated or derived features share credit, so no single ranking should be read as definitive.
* **Logistic regression** gives the clearest directions. Per standard deviation, the odds of diabetes versus none rise by about 1.73 for poorer general health, 1.56 for age, 1.50 for BMI and 1.45 for high blood pressure.

Several signals reflect reverse causation rather than risk. A more recent check-up is associated with *higher* odds (odds ratio 0.79 per SD of time since the last check-up), and SHAP shows the same direction. A cholesterol check carries positive weight. Heavy drinking pushes predictions *down*. The smoking dummies all have negative coefficients only because they are measured relative to the intercept; their differences from each other are small. These are features of how a diagnosis changes behaviour and health-care contact, not of what causes diabetes. They are useful for classification, but any causal reading would be wrong.
""",
    "subgroups": """
Diabetes detection is broadly similar by sex: recall is 0.486 for women and 0.517 for men, with ROC-AUC 0.846 and 0.839. The differences by age and income are larger:

* **Age.** Among adults aged 18–44, ROC-AUC is the highest of any group (0.877), but recall is only 0.209. With a prevalence of 3%, few young people cross the single, population-wide decision rule. Among those aged 65+, recall is 0.544 but ROC-AUC is the lowest (0.767), because diabetes is common (20.5%) and harder to separate from the comorbidities of ageing.
* **Income.** Recall rises from 0.370 in the $50k+ group to 0.631 below $20k, mostly tracking prevalence (9.0% vs 19.9%), while ROC-AUC falls from 0.856 to 0.816.
* **Race/ethnicity.** Black non-Hispanic respondents have the highest prevalence (20.5%) and the lowest ROC-AUC (0.813) among the race/ethnicity groups, against 0.840 for White and 0.866 for Hispanic respondents. Race was never an input, so this reflects how the other indicators relate to diabetes in each group.

These are descriptive disparities, not proof of unfairness. They show that a single threshold yields very different sensitivities across groups, and that groups with more diabetes are, in ranking terms, the hardest to classify. Any screening use would need group-aware threshold review and calibration checks.
""",
    "discussion": """
**Answer to the research question.** With 25 survey indicators, all model families reach a similar ceiling. On the held-out test set, macro ROC-AUC is about 0.80 and macro average precision about 0.48. The diabetes class is reasonably well ranked (ROC-AUC up to 0.843, AP 0.434 against a 13% baseline), while prediabetes is barely distinguishable (AP 0.036 against a 1.7% baseline).

**Traditional ML versus deep learning.** The hypothesis is supported in direction but with a small effect size. Tuned gradient boosting has the best threshold-free scores (validation macro AP 0.4814, test 0.4790). The best network, the wide-and-deep model, essentially ties it on ROC-AUC (0.8005 vs 0.8004) and trails slightly on AP (0.4775). Logistic regression trails both by about 0.01 AP. Notably, the network that worked best was the one that explicitly embedded a logistic regression, and engineered interactions added nothing to either paradigm. Most of the signal in these data is additive.

**Efficiency and interpretability.** Logistic regression trains in about 5 seconds and has directly readable coefficients. Boosting needs about 85 seconds plus an 18-minute search, and the neural networks 28–216 seconds per run plus careful learning-rate and epoch choices. The gains in exchange are about 0.01 macro AP and 0.006 macro ROC-AUC.

**What matters more than the algorithm.** The decision rule matters most. Default argmax decisions find fewer than one in five diabetes cases; validation-tuned weights find half; an F2 screening threshold finds 85%, at the cost of flagging 39% of respondents. Class weighting changes where the threshold sits rather than how well cases are ranked.

**Limits.** The main limits sit in the data, not the models. The labels are self-reported diagnoses, so undiagnosed diabetes and untested prediabetes are counted as negatives, and the false positives look like plausible undiagnosed cases. The data are cross-sectional, and several strong predictors (general health, check-up recency, cholesterol testing) partly reflect a diagnosis that has already happened. Detection also varies widely across age, income and racial groups under one threshold.

**What this means for rural sub-Saharan Africa.** Three findings carry over to the setting my mission targets.

* **A cheap model is enough.** Logistic regression is within about 0.01 AP of the best model, runs in milliseconds on any phone, and its coefficients can be explained to health workers. That suits community-health-worker tools far better than a model that needs a GPU or careful tuning.
* **The threshold is a resource decision.** Detecting 85% of cases meant flagging 39% of people, which only makes sense if confirmatory HbA1c or glucose tests are available for that many. Thresholds must be set with local clinic capacity in mind.
* **Labels and equity cannot be taken for granted.** Sensitivity differed sharply between groups, and "ever told" labels undercount people with poor access to care. That is exactly the situation of rural communities where most diabetes is undiagnosed. A model trained on such labels would learn to miss them.

So the responsible next step for Africa is to collect laboratory-confirmed labels locally, then recalibrate and revalidate before deploying anything.

**Next steps.** I would add laboratory-linked data (for example NHANES HbA1c) to test against true glycaemic status, use design-based survey estimation, check calibration and set group-specific thresholds, and validate on later BRFSS years.
""",
}
