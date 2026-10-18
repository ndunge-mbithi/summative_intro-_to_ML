"""Generate notebooks/diabetes_classification.ipynb from cell definitions.

Interpretation cells are written only after the notebook has been executed
once and the actual outputs have been read; their text lives in
scripts/notebook_interpretations.py and is keyed by section.
"""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
import sys  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from notebook_interpretations import INTERP  # noqa: E402

cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip("\n")))


def code(text):
    cells.append(nbf.v4.new_code_cell(text.strip("\n")))


def interp(key):
    text = INTERP.get(key)
    if text:
        md("**Interpretation.** " + text.strip())


# --------------------------------------------------------------------- 1
md(r"""
# Predicting Diabetes Status from Population Health Indicators Using Traditional Machine Learning and Deep Learning

**Author:** Ndunge Mutheu Mbithi · BSc Software Engineering, African Leadership University

## 1. Project Introduction and Research Question

Diabetes is one of the largest chronic-disease burdens worldwide: the IDF Diabetes Atlas estimates that 10.5% of adults aged 20–79 (536.6 million people) were living with diabetes in 2021, rising to a projected 12.2% by 2045. A large share of cases and most prediabetes remain undiagnosed, which is why population-health researchers are interested in whether routinely collected survey indicators can flag people who may benefit from formal screening.

**Why this matters for my mission.** My mission is *to revolutionize the healthcare industry in sub-Saharan Africa through technology, creating accessible, equitable, and scalable healthcare solutions for underserved rural communities.* The diagnosis gap is widest where I want to work. IDF Africa reports 24 million adults with diabetes in Africa in 2021, and 54% of them undiagnosed. In Kenya's 2015 national STEPS survey, more than half of adults with diabetes did not know they had it, and prevalence was lower in rural than in urban areas — a pattern that may partly reflect less testing. In rural clinics, laboratory glucose testing is often scarce, but a short questionnaire administered by a community health worker costs almost nothing. This project asks the question that has to come first: how much can questionnaire-style indicators alone actually tell us, which models extract that information best, and what does that cost in false alarms and missed cases? I use the BRFSS because no openly available African survey yet offers comparable scale (Kenya's STEPS analysis covered 4,069 adults), richly documented predictors, and a three-level diabetes label. Any model built here would have to be revalidated on local data before use in Kenya or elsewhere in the region. I treat equity as a measurable outcome, which is why the evaluation includes subgroup analysis by age, income and race/ethnicity.

In this notebook I investigate the question:

> **How effectively can traditional machine learning and deep learning models classify diabetes status from population health indicators, and how do their predictive performance, interpretability, computational efficiency, and error patterns differ?**

I treat this strictly as a population-health and screening-research exercise. None of the models here is a clinically validated diagnostic tool, and the outputs are self-reported survey labels, not laboratory diagnoses.

**Hypothesis.** Based on benchmark evidence that tree ensembles tend to match or beat neural networks on structured tabular data, I expect gradient boosting to achieve the strongest threshold-free discrimination, with neural networks performing comparably but not better, and logistic regression close behind because most survey indicators are binary or ordinal with largely additive effects. I also expect the prediabetes class to be very hard to separate from both neighbours. I treat this as a hypothesis to test, not a conclusion.
""")

# --------------------------------------------------------------------- 2
md(r"""
## 2. Dataset Source and Verification

I use the **CDC Behavioral Risk Factor Surveillance System (BRFSS) 2015** combined landline and cell-phone public-use file (`LLCP2015.XPT`), a state-based telephone survey of non-institutionalised US adults.

| Item | Value |
|---|---|
| Official page | https://www.cdc.gov/brfss/annual_data/annual_2015.html |
| Codebook | https://www.cdc.gov/brfss/annual_data/2015/pdf/codebook15_llcp.pdf |
| File | SAS transport `LLCP2015.XPT` (released August 2016) |
| Published size | 441,456 records |
| Target | `DIABETE3` — "(Ever told) you have diabetes" |
| Survey design | weight `_LLCPWT`, strata `_STSTR`, PSU `_PSU` (raking weights) |

**Target categories (codebook section 6.12):** 1 = Yes; 2 = Yes, but female told only during pregnancy; 3 = No; 4 = No, pre-diabetes or borderline diabetes; 7 = Don't know; 9 = Refused; blank = not asked.

I use the original raw survey rather than the widely circulated 253,680-row "Diabetes Health Indicators" derivative. That derivative removes rows with any missing value (which shifts the population) and, in its binary version, merges prediabetes into diabetes.

**Provenance note.** My execution environment could not reach cdc.gov, so I obtained a byte-level copy of `LLCP2015.XPT` from a public research repository mirror. I then verified it against the official codebook: the record count and every `DIABETE3` frequency cell must match exactly, and the loader refuses to proceed otherwise. In Google Colab, the loader downloads the file directly from the CDC URL above. **Licensing:** BRFSS public-use files are US federal government data released for public use without individual identifiers; I still exclude raw data from Git.
""")

code(r"""
# Locate the project. In Colab, clone the repository first (set REPO_URL to its GitHub URL).
import os, sys, subprocess
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules
REPO_URL = os.environ.get("REPO_URL", "https://github.com/ndunge-mbithi/summative_intro-_to_ML")
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
if not (ROOT / "src").exists():
    target = Path("/content/healthcare-classification-project")
    if not target.exists():
        if not REPO_URL:
            raise FileNotFoundError("src/ not found. Run this notebook from the repository, or set "
                                    "os.environ['REPO_URL'] to the repository URL so it can be cloned.")
        subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(target)], check=True)
    ROOT = target
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
print("Project root:", ROOT, "| running in Colab:", IN_COLAB)
""")

# --------------------------------------------------------------------- 3
md(r"""
## 3. Environment Setup and Reproducibility

I fix random seeds for Python, NumPy, TensorFlow and every scikit-learn estimator (`random_state=42`), and I enable TensorFlow's deterministic operations so that re-running the notebook reproduces the experiment table. I also record the package versions and the hardware, because training times reported later depend on them. Dependencies are pinned in `requirements.txt`. The next cell installs anything that is missing: in a fresh Colab runtime, that is usually only SHAP and PyArrow. Locally it does nothing if the environment already matches.
""")

code(r"""
# Install missing dependencies (no-op when they are already present).
import importlib.util
REQUIRED = {"numpy": "numpy", "pandas": "pandas", "sklearn": "scikit-learn", "tensorflow": "tensorflow",
            "matplotlib": "matplotlib", "seaborn": "seaborn", "shap": "shap", "pyarrow": "pyarrow", "scipy": "scipy"}
missing = [pip_name for module, pip_name in REQUIRED.items() if importlib.util.find_spec(module) is None]
if missing:
    print("Installing:", missing)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *missing], check=True)
else:
    print("All dependencies present; full pinned versions are listed in requirements.txt")
""")

code(r"""
import json, time, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*shuffle=True.*")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from src.utils import set_global_seed, environment_report, SEED
from src import data_loading, preprocessing as prep, feature_engineering as fe
from src import traditional_models as tm, deep_learning_models as dl, evaluation as ev
from src.experiment_tracking import ExperimentTracker

set_global_seed(SEED)
FIG = ROOT / "reports" / "figures"; FIG.mkdir(parents=True, exist_ok=True)
MODELS = ROOT / "models"; MODELS.mkdir(exist_ok=True)
LOGS = ROOT / "logs"
sns.set_theme(style="whitegrid", context="notebook")
pd.set_option("display.width", 160, "display.max_columns", 40)

env = environment_report()
(ROOT / "reports" / "environment.json").write_text(json.dumps(env, indent=2))
pd.Series(env, name="value").to_frame()
""")
interp("env")

# --------------------------------------------------------------------- 4
md(r"""
## 4. Data Loading and Initial Inspection

The loader reads the 1.17 GB transport file in 100,000-row chunks and keeps only the variables listed in `src/data_loading.RAW_COLUMNS`: design variables, the target, candidate predictors, and race/ethnicity, which I use **only for subgroup evaluation, never as a model input**. It verifies the full file against the codebook (record count, variable count and every target frequency) and caches the subset as Parquet.
""")

code(r"""
raw, report = data_loading.load_brfss_2015()
print("Verification passed:", report.passed)
print("Records:", f"{report.n_records:,}", "| variables in file:", report.n_variables,
      "| SHA-256 matches reference copy:", report.sha256_matches_reference)
verif = pd.DataFrame({
    "codebook_count": pd.Series(data_loading.EXPECTED_DIABETE3),
    "observed_count": pd.Series(report.diabete3_counts)})
verif["match"] = verif.codebook_count == verif.observed_count
display(verif)
print("Subset shape:", raw.shape)
raw.head()
""")

code(r"""
print(raw.dtypes.value_counts())
print("Duplicate respondent IDs (state + SEQNO):", raw.duplicated(["_STATE", "SEQNO"]).sum())
print("PSU repeated within a state:", raw.duplicated(["_STATE", "_PSU"]).sum())
print("Fully duplicated rows:", raw.duplicated().sum())
""")
interp("loading")

# --------------------------------------------------------------------- 5
md(r"""
## 5. Exploratory Data Analysis

I first recode the raw variables into analysis features (Section 6 documents every rule), because raw BRFSS codes such as 88 ("none") or 77 ("don't know") would distort any summary statistic. I then examine the target distribution, both unweighted and survey-weighted, followed by missingness, distributions and associations.

**Target definition.** I map `DIABETE3` to three classes: **No diabetes** (codes 3 and 2), **Prediabetes** (code 4) and **Diabetes** (code 1). Code 2 (gestational diabetes only) is mapped to "No diabetes" because it does not describe current diabetes status, which is the convention CDC uses for its diabetes indicator. Codes 7, 9 and blank are unusable labels and are removed (798 respondents). I do not merge prediabetes with diabetes, because that would hide the hardest part of the problem.
""")

code(r"""
X_all, y_all, meta_all = prep.clean_brfss(raw)
print("Analysis sample:", X_all.shape, "| removed unusable labels:", len(raw) - len(X_all))
tgt = pd.DataFrame({
    "n": y_all.value_counts().sort_index(),
    "unweighted_%": (100 * y_all.value_counts(normalize=True).sort_index()).round(2),
    "survey_weighted_%": (100 * meta_all.groupby(y_all)["_LLCPWT"].sum() / meta_all["_LLCPWT"].sum()).round(2),
})
tgt.index = prep.CLASS_NAMES
display(tgt)

fig, ax = plt.subplots(figsize=(7, 3.5))
tgt[["unweighted_%", "survey_weighted_%"]].plot.bar(ax=ax, rot=0, color=["#4C72B0", "#55A868"])
ax.set_ylabel("% of respondents"); ax.set_title("Diabetes status distribution (BRFSS 2015)")
for c in ax.containers: ax.bar_label(c, fmt="%.1f", fontsize=8)
fig.tight_layout(); fig.savefig(FIG / "fig01_target_distribution.png", dpi=150)
""")
interp("target")

code(r"""
miss = prep.missingness_table(X_all)
display(miss[miss.missing > 0])
fig, ax = plt.subplots(figsize=(8, 5))
miss[miss.missing > 0].percent.sort_values().plot.barh(ax=ax, color="#8172B2")
ax.set_xlabel("% missing after recoding special codes"); ax.set_title("Missingness by feature")
fig.tight_layout(); fig.savefig(FIG / "fig02_missingness.png", dpi=150)

# Is missingness itself related to the outcome? (informative missingness check)
inc_missing = X_all["income"].isna()
print("Diabetes prevalence | income missing: %.3f vs observed: %.3f"
      % ((y_all[inc_missing] == 2).mean(), (y_all[~inc_missing] == 2).mean()))
bmi_missing = X_all["bmi"].isna()
print("Diabetes prevalence | BMI missing: %.3f vs observed: %.3f"
      % ((y_all[bmi_missing] == 2).mean(), (y_all[~bmi_missing] == 2).mean()))
""")
interp("missing")

code(r"""
display(X_all[prep.CONTINUOUS_FEATURES + prep.ORDINAL_FEATURES].describe().T.round(2))

fig, axes = plt.subplots(1, 3, figsize=(16, 4))
for k, name in enumerate(prep.CLASS_NAMES):
    sns.kdeplot(X_all.loc[y_all == k, "bmi"].dropna(), ax=axes[0], label=name, common_norm=False, bw_adjust=1.5)
axes[0].set(xlim=(12, 60), title="BMI distribution by class", xlabel="BMI (kg/m²)"); axes[0].legend()
age_prev = pd.crosstab(pd.cut(X_all["age"], [17, 29, 39, 49, 59, 69, 80]), y_all, normalize="index") * 100
age_prev.columns = prep.CLASS_NAMES
age_prev[["Prediabetes", "Diabetes"]].plot(ax=axes[1], marker="o")
axes[1].set(title="Prevalence by age band", ylabel="% of age band", xlabel="age")
axes[1].tick_params(axis="x", rotation=30)
gh_prev = pd.crosstab(X_all["gen_health"], y_all, normalize="index") * 100
gh_prev.columns = prep.CLASS_NAMES
gh_prev[["Prediabetes", "Diabetes"]].plot.bar(ax=axes[2], rot=0)
axes[2].set(title="Prevalence by self-rated health (1=excellent, 5=poor)", ylabel="% of group")
fig.tight_layout(); fig.savefig(FIG / "fig03_distributions.png", dpi=150)
""")
interp("distributions")

code(r"""
binary_rates = pd.DataFrame({
    f: {"prevalence_if_1": (y_all[X_all[f] == 1] == 2).mean() * 100,
        "prevalence_if_0": (y_all[X_all[f] == 0] == 2).mean() * 100}
    for f in prep.BINARY_FEATURES}).T
binary_rates["ratio"] = binary_rates.prevalence_if_1 / binary_rates.prevalence_if_0
binary_rates = binary_rates.sort_values("ratio", ascending=False).round(2)
display(binary_rates)

fig, axes = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={"width_ratios": [1, 1.25]})
binary_rates[["prevalence_if_0", "prevalence_if_1"]].plot.barh(ax=axes[0], color=["#BBBBBB", "#C44E52"])
axes[0].set(xlabel="% with diabetes", title="Diabetes prevalence by binary indicator"); axes[0].invert_yaxis()
corr = pd.concat([X_all[prep.NUMERIC_FEATURES], y_all.rename("target")], axis=1).corr(method="spearman")
sns.heatmap(corr, cmap="vlag", center=0, ax=axes[1], cbar_kws={"shrink": 0.7}, xticklabels=True, yticklabels=True)
axes[1].set_title("Spearman correlation (features and ordinal target)")
axes[1].tick_params(labelsize=7)
fig.tight_layout(); fig.savefig(FIG / "fig04_associations.png", dpi=150)
""")
interp("associations")

# --------------------------------------------------------------------- 6
md(r"""
## 6. Data Preprocessing and Feature Engineering

**Recoding (implemented in `src/preprocessing.clean_brfss`, unit-tested in `tests/`).** Every special code from the codebook is mapped to `NaN` instead of being read as a number: 7/77 ("don't know"), 9/99 ("refused"), calculated-variable code 9, `_AGEG5YR`=14, and blanks. "Number of days" items map 88 ("none") to 0. `_BMI5` has two implied decimals, so I divide it by 100. `TOLDHI2` (told high cholesterol) has a skip pattern: it is only asked when cholesterol was ever checked, so respondents never checked are coded 0 ("not told") rather than missing. I use CDC's imputed, top-coded age `_AGE80`.

**Duplicates.** Each respondent has a unique state + sequence number, and `_PSU` equals the sequence number. So there are no repeated respondents and no clusters to keep together when splitting. Rows that look identical on the 25 coarse survey features are different people, so I do **not** delete them.

**Missing values.** Missingness is concentrated in income (refusals) and in calculated diet/activity variables from partially completed interviews. Dropping incomplete rows, as the popular 253k derivative does, would discard about 18% of respondents and over-represent people willing to report income. I therefore impute numeric features with the training-set median **plus a binary missing-indicator column**, so the models can use missingness if it carries signal (the EDA shows the association is weak, but it costs little to keep). I also give the nominal smoking variable an explicit "missing" category. Imputation is fitted on the training split only, inside the pipeline.

**Outliers.** I investigate extreme BMI values below, but I do not delete them. Instead, continuous variables are winsorised at the 0.1st/99.9th percentiles learned on the training data (`QuantileClipper`). This limits their leverage on linear models and neural networks while keeping every respondent.

**Encoding and scaling.** Binary and ordinal items stay numeric (their order is meaningful); smoking status is one-hot encoded. I standardise features for logistic regression and neural networks, because both are optimised with gradient-based methods whose conditioning depends on feature scale. Tree ensembles split on thresholds and are invariant to monotone rescaling, so I do not scale for them.
""")

code(r"""
bmi = X_all["bmi"].dropna()
print("BMI range: %.2f - %.2f" % (bmi.min(), bmi.max()))
print("BMI < 15:", (bmi < 15).sum(), "| BMI > 60:", (bmi > 60).sum(), "| BMI > 80:", (bmi > 80).sum())
print("0.1% / 99.9% quantiles:", bmi.quantile([0.001, 0.999]).round(2).tolist())
print("Diabetes prevalence for BMI > 60: %.3f (overall %.3f)" % ((y_all[X_all.bmi > 60] == 2).mean(), (y_all == 2).mean()))
""")
interp("outliers")

md(r"""
**Feature engineering (`src/feature_engineering.ClinicalFeatureEngineer`).** I add six features motivated by diabetes epidemiology. The transformer is stateless (it learns nothing from data), so it cannot leak information between splits:

1. `bmi_category` — WHO obesity class, because risk rises in steps across obesity classes;
2. `cardiometabolic_count` — number of diagnosed high BP, high cholesterol, heart disease, stroke and kidney disease, reflecting metabolic-syndrome clustering;
3. `lifestyle_risk` — current smoking, inactivity, low fruit, low vegetable intake and heavy drinking;
4. `unhealthy_days` — physical + mental unhealthy days capped at 30 (CDC's "healthy days" construct);
5. `age_bmi` — an age × BMI interaction, because adiposity matters more with age;
6. `access_barrier` — no health plan or cost-related care barrier, since diagnosis requires contact with care.

I compare base and engineered feature sets empirically in experiments E06, E16 and E17.
""")

code(r"""
X_eng_preview = fe.ClinicalFeatureEngineer().fit_transform(X_all.head(50_000))
summary = pd.concat([X_eng_preview[fe.ENGINEERED_FEATURES].describe().T.round(2),
                     X_eng_preview[fe.ENGINEERED_FEATURES].corrwith((y_all.head(50_000) == 2).astype(float), method="spearman")
                     .rename("spearman_with_diabetes").round(3)], axis=1)
summary
""")
interp("engineering")

# --------------------------------------------------------------------- 7
md(r"""
## 7. Dataset Splitting

I use a stratified 70/15/15 split (seed 42):

* **training** (70%) fits every model, imputer, clipper and scaler;
* **validation** (15%) compares experiments, early-stops neural networks, and tunes decision weights and the screening threshold;
* **test** (15%) is touched once, at the end, for the final comparison of the selected models.

Stratification preserves the 1.7% prediabetes share in every split. Because each PSU is a single respondent, a respondent-level random split does not separate members of the same cluster. Hyper-parameter searches use stratified cross-validation inside the training set only.

The split itself is unweighted, so the performance metrics describe the BRFSS *sample*. To see how the picture changes for the weighted *population*, I additionally report survey-weighted test metrics using `_LLCPWT`. This is an approximation: proper design-based variance estimation (using strata) is outside the scope of this notebook.
""")

code(r"""
splits = prep.stratified_split(X_all, y_all, meta_all, sizes=(0.70, 0.15, 0.15), seed=SEED)
X_tr, y_tr, m_tr = splits["train"]; X_va, y_va, m_va = splits["val"]; X_te, y_te, m_te = splits["test"]
split_table = pd.DataFrame({name: {"n": len(s[1]), **{c: f"{100*(s[1]==k).mean():.2f}%" for k, c in enumerate(prep.CLASS_NAMES)}}
                            for name, s in splits.items()}).T
split_table
""")

# --------------------------------------------------------------------- 8
md(r"""
## 8. Traditional Machine Learning Models

I compare three families, each wrapped in the same leakage-safe pipeline (`feature engineering → ColumnTransformer → estimator`):

* **Multinomial logistic regression** — a linear model of the log-odds of each class versus the reference. It is the standard clinical-prediction baseline: well calibrated, fully interpretable, and cheap. L2 regularisation (`C`) controls coefficient shrinkage.
* **Random forest** — bagged, decorrelated decision trees that average away variance. Depth, leaf size and the number of features per split govern the bias–variance trade-off.
* **Gradient boosting** — trees fitted sequentially to the gradient of the log-loss, which reduces bias. I use scikit-learn's histogram-based implementation (`HistGradientBoostingClassifier`), which bins features into ≤255 buckets. That makes boosting on 300k rows feasible on a 2-core CPU, where the classic `GradientBoostingClassifier` would take far longer for the same model class.

I did not include an SVM: kernel SVMs scale roughly quadratically to cubically with the 308k training rows and do not produce native probabilities, which I need for threshold tuning.

**Selection metric.** Accuracy is misleading here: predicting "No diabetes" for everyone already scores about 85%. I rank experiments on validation **macro average precision** (mean one-vs-rest area under the precision–recall curve), which is threshold-free and gives the rare prediabetes class equal weight. I also report macro ROC-AUC, macro-F1 at the default argmax rule, and per-class recall.

Every run is logged by `ExperimentTracker` to `reports/experiment_results.csv`, `reports/experiment_log.jsonl` and TensorBoard (`logs/tensorboard/<run_id>`).
""")

code(r"""
tracker = ExperimentTracker(reset=True)
val_probas, fitted = {}, {}

def evaluate_and_log(exp_id, model_name, paradigm, proba, seconds, config, observation=""):
    metrics = ev.compute_metrics(y_va, proba)
    val_probas[exp_id] = proba
    tracker.log_run(exp_id, model_name, paradigm, config, metrics, seconds, observation)
    keys = ["accuracy", "f1_macro", "roc_auc_ovr_macro", "avg_precision_macro", "recall_prediabetes", "recall_diabetes"]
    print(f"{exp_id} {model_name}: " + ", ".join(f"{k}={metrics[k]:.4f}" for k in keys) + f" | {seconds:.1f}s")
    return metrics

def run_sklearn(exp_id, name, estimator, scale, engineered, imbalance, hyper, preprocessing_desc):
    pipe, sec = tm.fit_timed(tm.build_pipeline(estimator, scale=scale, engineered=engineered), X_tr, y_tr)
    fitted[exp_id] = pipe
    cfg = {"preprocessing": preprocessing_desc, "features": "base+engineered (31)" if engineered else "base (25)",
           "imbalance_strategy": imbalance, "hyperparameters": hyper}
    return evaluate_and_log(exp_id, name, "traditional ML", pipe.predict_proba(X_va), sec, cfg)

PRE_SCALED = "median impute + missing indicators, 0.1/99.9% winsorising, standardise, one-hot smoking"
PRE_TREE = "median impute + missing indicators, 0.1/99.9% winsorising, one-hot smoking (no scaling)"
""")

md(r"""
### E01–E03: Logistic regression baseline, scaling and class weighting

E01 is the reference baseline. E02 removes standardisation to measure how scaling affects a gradient-based solver. E03 reweights the loss inversely to class frequency (`class_weight="balanced"`), the simplest imbalance strategy.
""")

code(r"""
run_sklearn("E01", "Logistic regression", tm.make_logistic_regression(C=1.0), True, False, "none",
            {"C": 1.0, "penalty": "l2", "solver": "lbfgs", "max_iter": 1000}, PRE_SCALED)
import warnings as _w
with _w.catch_warnings(record=True) as caught:
    _w.simplefilter("always")
    run_sklearn("E02", "Logistic regression (unscaled)", tm.make_logistic_regression(C=1.0, max_iter=200), False, False, "none",
                {"C": 1.0, "penalty": "l2", "solver": "lbfgs", "max_iter": 200, "scaling": False}, PRE_TREE)
print("Convergence warnings in E02:", sum("converge" in str(w.message).lower() for w in caught))
print("lbfgs iterations used - E01:", fitted["E01"][-1].n_iter_, "| E02:", fitted["E02"][-1].n_iter_)
run_sklearn("E03", "Logistic regression (balanced)", tm.make_logistic_regression(C=1.0, class_weight="balanced"), True, False,
            "class_weight=balanced", {"C": 1.0, "penalty": "l2", "solver": "lbfgs", "max_iter": 1000}, PRE_SCALED)
""")
interp("lr")

md(r"""
### E04–E06: Random forest and gradient boosting baselines, and engineered features

The baselines use defaults that are sensible for 300k rows: 150 trees with `min_samples_leaf=5` for the forest, and 200 boosting iterations at learning rate 0.1 with 31-leaf trees for boosting. E06 repeats the boosting baseline with the six engineered features, so any difference is attributable to the features alone.
""")

code(r"""
run_sklearn("E04", "Random forest", tm.make_random_forest(n_estimators=150, min_samples_leaf=5), False, False, "none",
            {"n_estimators": 150, "max_depth": None, "min_samples_leaf": 5, "max_features": "sqrt"}, PRE_TREE)
run_sklearn("E05", "Gradient boosting", tm.make_gradient_boosting(), False, False, "none",
            {"learning_rate": 0.1, "max_iter": 200, "max_leaf_nodes": 31, "min_samples_leaf": 20}, PRE_TREE)
run_sklearn("E06", "Gradient boosting + engineered", tm.make_gradient_boosting(), False, True, "none",
            {"learning_rate": 0.1, "max_iter": 200, "max_leaf_nodes": 31, "min_samples_leaf": 20}, PRE_TREE)
rf = fitted["E04"][-1]
print("E04 mean tree depth: %.1f, mean leaves: %.0f" % (np.mean([t.get_depth() for t in rf.estimators_]),
                                                        np.mean([t.get_n_leaves() for t in rf.estimators_])))
""")
interp("trees")

# --------------------------------------------------------------------- 8.3 (tuning, traditional)
md(r"""
### 8.3 Hyper-parameter search for the traditional models (E07–E09)

I tune the two ensembles with `RandomizedSearchCV`: 3-fold stratified CV, scored on macro average precision, run **inside the training set only**. A full-data search would take hours on 2 CPU cores, so each search runs on a class-stratified 100,000-row subsample of the training data, and the winning configuration is then refitted on the full training set and evaluated on validation.

The search spaces follow from what I observed in the baselines:

* **Gradient boosting:** learning rate (0.02–0.2) traded against the number of iterations (200–800); tree complexity via `max_leaf_nodes` (15–63) and `min_samples_leaf`; and L2 regularisation. Smaller learning rates with more trees usually generalise better, so the grid covers that trade-off.
* **Random forest:** the baseline grows very deep trees (high variance), so I search shallower depths (12–30), larger leaves (5–100) and a larger share of features per split.
""")

code(r"""
from scipy.stats import loguniform, randint, uniform
X_sub, y_sub = tm.stratified_subsample(X_tr, y_tr, 100_000)

hgb_space = {
    "model__learning_rate": loguniform(0.02, 0.2),
    "model__max_iter": randint(200, 801),
    "model__max_leaf_nodes": randint(15, 64),
    "model__min_samples_leaf": randint(20, 401),
    "model__l2_regularization": loguniform(1e-3, 10),
}
hgb_search, hgb_search_sec = tm.randomized_search(tm.build_pipeline(tm.make_gradient_boosting(), scale=False, engineered=True),
                                                  hgb_space, X_sub, y_sub, n_iter=16, cv_folds=3, n_jobs=2)
hgb_cv = tm.search_results_frame(hgb_search)
hgb_cv.to_csv(ROOT / "reports" / "search_gradient_boosting.csv", index=False)
print(f"Gradient boosting search: {hgb_search_sec/60:.1f} min")
display(hgb_cv.head(6).round(4))
""")

code(r"""
best_hgb = {k.replace("model__", ""): (v.item() if hasattr(v, "item") else v) for k, v in hgb_search.best_params_.items()}
best_hgb = {k: (round(v, 5) if isinstance(v, float) else v) for k, v in best_hgb.items()}
print("Best gradient boosting configuration:", best_hgb)
run_sklearn("E07", "Gradient boosting (tuned)", tm.make_gradient_boosting(**best_hgb), False, True, "none",
            {**best_hgb, "search": "RandomizedSearchCV n_iter=16, 3-fold, 100k subsample, macro-AP"}, PRE_TREE)
run_sklearn("E08", "Gradient boosting (tuned, balanced)", tm.make_gradient_boosting(**best_hgb, class_weight="balanced"),
            False, True, "class_weight=balanced", best_hgb, PRE_TREE)
""")

code(r"""
rf_space = {
    "model__max_depth": randint(12, 31),
    "model__min_samples_leaf": randint(5, 101),
    "model__max_features": uniform(0.15, 0.35),
}
rf_search, rf_search_sec = tm.randomized_search(
    tm.build_pipeline(tm.make_random_forest(n_estimators=100, n_jobs=1), scale=False, engineered=True),
    rf_space, X_sub, y_sub, n_iter=10, cv_folds=3, n_jobs=2)
rf_cv = tm.search_results_frame(rf_search)
rf_cv.to_csv(ROOT / "reports" / "search_random_forest.csv", index=False)
print(f"Random forest search: {rf_search_sec/60:.1f} min")
display(rf_cv.head(6).round(4))
best_rf = {k.replace("model__", ""): (round(float(v), 4) if isinstance(v, (float, np.floating)) else int(v))
           for k, v in rf_search.best_params_.items()}
run_sklearn("E09", "Random forest (tuned)", tm.make_random_forest(n_estimators=300, **best_rf), False, True, "none",
            {**best_rf, "n_estimators": 300, "search": "RandomizedSearchCV n_iter=10, 3-fold, 100k subsample, macro-AP"}, PRE_TREE)
""")
interp("tuning")

# --------------------------------------------------------------------- 9
md(r"""
## 9. TensorFlow/Keras Deep Learning Models

**Input pipeline.** I fit the preprocessing `ColumnTransformer` (imputation, indicators, winsorising, scaling, one-hot) on the training split. I then wrap the resulting arrays in `tf.data` pipelines: the training pipeline is cached, **shuffled with reshuffling every epoch** and batched; validation and test pipelines are batched in fixed order (so predictions align with labels) without shuffling; all use `prefetch(AUTOTUNE)` to overlap input preparation with training. No validation or test statistics reach the training pipeline.

**Training.** All networks use a softmax output over the three classes, sparse categorical cross-entropy loss and the Adam optimiser. Training runs for up to 60 epochs, with **early stopping** on validation loss (patience 6, restoring the best weights), **checkpointing** of the best model, a CSV log and TensorBoard logging.

**Architectures.**

* **E10 — Sequential baseline:** one hidden layer of 32 ReLU units. This is close to the simplest network that can represent non-linear interactions; it tests whether a small non-linear model adds anything over logistic regression.
* **E11 — Sequential regularised:** three hidden layers (128-64-32) with batch normalisation, dropout 0.3 and L2 weight decay 1e-4. Greater capacity allows richer interactions, and the regularisers counter the overfitting that extra capacity invites.
* **E12/E13 — learning rate** 1e-2 and 1e-4 on the E11 architecture, to see how optimisation speed and stability change.
* **E14 — batch size** 4096 instead of 512: fewer, less noisy gradient steps per epoch.
* **E15 — class weighting:** E11 trained with softened inverse-frequency weights (power 0.5), as the network analogue of E03/E08.
* **E16/E17 — Functional API wide-and-deep** (base and engineered features). A linear "wide" branch, equivalent to multinomial logistic regression, runs in parallel with a regularised "deep" branch; their logits are summed before the softmax. I chose this because E01 shows that a linear model already captures most of the signal. The wide path keeps that additive signal directly, while the deep path only has to learn the residual interactions. A two-branch graph like this needs the Functional API; it cannot be written as a Sequential stack.
""")

code(r"""
import tensorflow as tf
from tensorflow import keras
set_global_seed(SEED)

pre_base = tm.build_preprocessing_pipeline(scale=True, engineered=False).fit(X_tr)
pre_eng = tm.build_preprocessing_pipeline(scale=True, engineered=True).fit(X_tr)

A = {"base": {s: pre_base.transform(x).astype("float32") for s, x in [("tr", X_tr), ("va", X_va), ("te", X_te)]},
     "eng": {s: pre_eng.transform(x).astype("float32") for s, x in [("tr", X_tr), ("va", X_va), ("te", X_te)]}}
print("Input dimensions - base:", A["base"]["tr"].shape[1], "| engineered:", A["eng"]["tr"].shape[1])

def datasets(feat, batch):
    return (dl.make_tf_dataset(A[feat]["tr"], y_tr.values, batch_size=batch, shuffle=True),
            dl.make_tf_dataset(A[feat]["va"], y_va.values, batch_size=4096),
            dl.make_tf_dataset(A[feat]["va"], batch_size=4096))

train_ds, _, _ = datasets("base", 512)
print(train_ds.element_spec)

histories, keras_models = {}, {}

def run_keras(exp_id, name, builder, feat="base", batch=512, class_weight=None, imbalance="none", hyper=None, epochs=60):
    set_global_seed(SEED)
    tr_ds, va_ds, va_x = datasets(feat, batch)
    model = builder(A[feat]["tr"].shape[1])
    model, hist, sec, n_ep = dl.train_keras_model(model, tr_ds, va_ds, LOGS / "keras" / exp_id, epochs=epochs,
                                                  patience=6, class_weight=class_weight)
    histories[exp_id] = hist; keras_models[exp_id] = model
    best_ep = int(np.argmin(hist["val_loss"])) + 1
    cfg = {"preprocessing": PRE_SCALED + " (fitted on train) -> tf.data", "features": "base+engineered (31)" if feat == "eng" else "base (25)",
           "imbalance_strategy": imbalance,
           "hyperparameters": {**(hyper or {}), "batch_size": batch, "epochs_run": n_ep, "best_epoch": best_ep,
                               "params": int(model.count_params()), "optimizer": "Adam",
                               "early_stopping": "val_loss, patience 6"}}
    proba = model.predict(va_x, verbose=0)
    print(f"   epochs run {n_ep}, best epoch {best_ep}, train loss {hist['loss'][best_ep-1]:.4f}, val loss {hist['val_loss'][best_ep-1]:.4f}")
    return evaluate_and_log(exp_id, name, "deep learning", proba, sec, cfg)
""")

code(r"""
run_keras("E10", "Sequential MLP baseline", lambda d: dl.build_sequential_baseline(d, hidden_units=32, learning_rate=1e-3),
          hyper={"layers": "32", "activation": "relu", "learning_rate": 1e-3, "dropout": 0, "l2": 0})
keras_models["E10"].summary()
""")

code(r"""
REG = {"layers": "128-64-32", "batch_norm": True, "dropout": 0.3, "l2": 1e-4}
run_keras("E11", "Sequential MLP regularised", lambda d: dl.build_sequential_regularized(d, learning_rate=1e-3),
          hyper={**REG, "learning_rate": 1e-3})
run_keras("E12", "Sequential MLP regularised, lr 1e-2", lambda d: dl.build_sequential_regularized(d, learning_rate=1e-2),
          hyper={**REG, "learning_rate": 1e-2})
run_keras("E13", "Sequential MLP regularised, lr 1e-4", lambda d: dl.build_sequential_regularized(d, learning_rate=1e-4),
          hyper={**REG, "learning_rate": 1e-4})
run_keras("E14", "Sequential MLP regularised, batch 4096", lambda d: dl.build_sequential_regularized(d, learning_rate=1e-3),
          batch=4096, hyper={**REG, "learning_rate": 1e-3})
cw = dl.balanced_class_weights(y_tr.values, power=0.5)
print("Softened class weights:", {prep.CLASS_NAMES[k]: round(v, 3) for k, v in cw.items()})
run_keras("E15", "Sequential MLP regularised, class-weighted", lambda d: dl.build_sequential_regularized(d, learning_rate=1e-3),
          class_weight=cw, imbalance="class weights (n/(k*n_c))^0.5", hyper={**REG, "learning_rate": 1e-3})
""")

code(r"""
WD = {"architecture": "wide (linear) + deep 128-64, logits summed", "dropout": 0.3, "l2": 1e-4, "learning_rate": 1e-3}
run_keras("E16", "Functional wide-and-deep", lambda d: dl.build_wide_and_deep(d), hyper=WD)
run_keras("E17", "Functional wide-and-deep + engineered", lambda d: dl.build_wide_and_deep(d), feat="eng", hyper=WD)
keras_models["E16"].summary()
""")

code(r"""
fig = ev.plot_learning_curves({k: histories[k] for k in ["E10", "E11", "E12", "E13", "E14", "E16"]},
                              path=FIG / "fig05_keras_learning_curves.png")
""")
interp("keras")

md(r"""
**Learning curve for the boosting model.** Gradient boosting has no epochs, but its staged predictions give the equivalent diagnostic: training versus validation log-loss as trees are added.
""")

code(r"""
from sklearn.metrics import log_loss
best_trad_pipe = fitted["E07"]
pre_fitted = tm.preprocessing_part(best_trad_pipe)
hgb_model = best_trad_pipe[-1]
sub_idx = X_tr.sample(60_000, random_state=SEED).index
Xtr_sub_t = pre_fitted.transform(X_tr.loc[sub_idx]); ytr_sub = y_tr.loc[sub_idx]
Xva_t = pre_fitted.transform(X_va)
stages = np.arange(1, hgb_model.n_iter_ + 1)
tr_loss = [log_loss(ytr_sub, p, labels=[0, 1, 2]) for p in hgb_model.staged_predict_proba(Xtr_sub_t)]
va_loss = [log_loss(y_va, p, labels=[0, 1, 2]) for p in hgb_model.staged_predict_proba(Xva_t)]
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(stages, tr_loss, label="training (60k sample)"); ax.plot(stages, va_loss, label="validation")
ax.axvline(stages[int(np.argmin(va_loss))], color="grey", ls=":", label=f"min val loss at {stages[int(np.argmin(va_loss))]} trees")
ax.set(xlabel="boosting iterations", ylabel="log-loss", title="E07 gradient boosting: training vs validation loss"); ax.legend()
fig.tight_layout(); fig.savefig(FIG / "fig06_boosting_learning_curve.png", dpi=150)
print("final train loss %.4f | final val loss %.4f | min val loss %.4f" % (tr_loss[-1], va_loss[-1], min(va_loss)))
""")
interp("boost_curve")

# --------------------------------------------------------------------- 10
md(r"""
## 10. Hyperparameter Optimization and Experiment Tracking

All 17 experiments are now complete. This section summarises how the hyper-parameters were optimised and makes every run traceable:

* **Randomized search** (Section 8.3) for gradient boosting (16 candidates) and random forest (10 candidates), with 3-fold CV on training data only. The full candidate tables are saved to `reports/search_*.csv`.
* **Manual, one-factor-at-a-time variation** for the networks (Section 9): learning rate (E11 vs E12/E13), batch size (E14), regularisation and capacity (E10 vs E11), class weighting (E15), and architecture and features (E16/E17).
* **Tracking:** each run has a run ID and timestamp in `reports/experiment_log.jsonl`, a row in `reports/experiment_results.csv`, and a TensorBoard directory. Keras runs also log per-epoch curves.

Below I attach a one-line observation to every run, which completes the results table. I wrote these after reading the executed outputs. The notebook is fully seeded, and a second clean execution reproduced every metric exactly, so the observations hold on re-execution.
""")

code(r"""
OBSERVATIONS = {
    "E01": "Reference model: converges in 36 iterations; 0.86 accuracy hides 0 prediabetes and 0.18 diabetes recall under argmax.",
    "E02": "Without scaling, L-BFGS hits its 200-iteration cap (convergence warning), runs ~5x slower and is worse than E01 on every metric.",
    "E03": "Balanced weights vs E01: diabetes recall 0.18->0.59, prediabetes 0->0.31, but accuracy 0.645 and macro AP drops slightly; threshold moved, ranking not improved.",
    "E04": "Fully grown trees (mean depth ~39, ~13k leaves): macro AP above E01 but ROC-AUC below it, the signature of a high-variance learner.",
    "E05": "Boosting improves both macro AP and ROC-AUC over the forest: sequential bias reduction beats averaging deep trees here.",
    "E06": "Engineered features add only +0.0004 macro AP to E05; trees already form interactions such as age x BMI by splitting.",
    "E07": "Search picks the most regularised candidate (lr 0.036, 16 leaves, L2 4.34; train-fold 0.625 vs CV 0.475): best macro AP of all runs.",
    "E08": "Balanced weights on E07 repeat the E03 pattern: diabetes recall 0.66, prediabetes 0.28, macro AP lower.",
    "E09": "Shallower forest (depth 14, ~16% features per split) raises ROC-AUC 0.786->0.792 at unchanged macro AP: regularisation smooths mid-risk ranking.",
    "E10": "32-unit MLP early-stops at epoch 9 with a negligible train/val gap; beats LR slightly but trails boosting.",
    "E11": "Deeper MLP with BN/dropout/L2: small gain over E10; training loss above validation loss because regularisation is active only in training.",
    "E12": "lr 1e-2 is unstable (best epoch 2, oscillating validation loss) and loses macro AP versus E11.",
    "E13": "lr 1e-4 is still improving at the 60-epoch cap and takes ~2x E11's time for the same macro AP: too slow for the budget.",
    "E14": "Batch 4096 needs 58 of 60 epochs to match E11: fewer updates per epoch, no quality gain.",
    "E15": "Softened class weights raise diabetes recall 0.16->0.50 and macro-F1 to 0.454 with macro AP unchanged; prediabetes still never predicted.",
    "E16": "Wide-and-deep is the best deep model: the linear wide path keeps the additive signal, the deep path adds little on top.",
    "E17": "Engineered features give no gain on E16, consistent with E06.",
}
tracker.annotate(OBSERVATIONS)
print("Experiments executed and logged:", len(tracker.records))
print("Best gradient boosting CV candidates (macro AP):", hgb_cv.mean_test_score.head(3).round(4).tolist())
print("Best random forest CV candidates (macro AP):", rf_cv.mean_test_score.head(3).round(4).tolist())
runs = sorted(p.name for p in (LOGS / "tensorboard").iterdir())
print(f"TensorBoard run directories: {len(runs)}, e.g. {runs[0]} ... {runs[-1]}  (inspect with: tensorboard --logdir logs)")
tracker.results_frame()[["experiment_id", "run_id", "model", "imbalance_strategy", "avg_precision_macro", "observation"]]
""")

# --------------------------------------------------------------------- 11
md(r"""
## 11. Model Evaluation and Comparison

### 11.1 Validation comparison of all experiments
""")

code(r"""
results = tracker.results_frame()
show = ["experiment_id", "model", "features", "imbalance_strategy", "train_seconds", "accuracy", "balanced_accuracy",
        "f1_macro", "roc_auc_ovr_macro", "avg_precision_macro", "ap_prediabetes", "ap_diabetes", "recall_prediabetes", "recall_diabetes"]
display(results[show].round(4))
fig = ev.plot_model_comparison(results, ["avg_precision_macro", "roc_auc_ovr_macro", "f1_macro", "balanced_accuracy"],
                               path=FIG / "fig07_model_comparison.png")
""")
interp("comparison")

md(r"""
### 11.2 Model selection and decision rule

I select one traditional and one deep-learning model by **validation macro average precision**. With argmax decisions, both models almost never predict the minority classes, because the probability of prediabetes rarely exceeds that of "no diabetes" for anyone. A 0.5-style default threshold is therefore inappropriate. I rescale class probabilities with per-class weights tuned **on validation** to maximise macro-F1, which is the multiclass analogue of moving a binary threshold. The test set is still unused at this point.
""")

code(r"""
trad = results[results.paradigm == "traditional ML"].sort_values("avg_precision_macro", ascending=False).iloc[0]
deep = results[results.paradigm == "deep learning"].sort_values("avg_precision_macro", ascending=False).iloc[0]
BEST_TRAD, BEST_DEEP = trad.experiment_id, deep.experiment_id
print("Selected traditional model:", BEST_TRAD, trad.model, "| val macro-AP %.4f" % trad.avg_precision_macro)
print("Selected deep model:       ", BEST_DEEP, deep.model, "| val macro-AP %.4f" % deep.avg_precision_macro)

decision = {}
for eid in [BEST_TRAD, BEST_DEEP, "E01"]:
    w, score, table = ev.tune_decision_weights(y_va.values, val_probas[eid])
    base = ev.compute_metrics(y_va, val_probas[eid])["f1_macro"]
    decision[eid] = w
    print(f"{eid}: weights (none, prediabetes, diabetes) = {w.tolist()} | val macro-F1 argmax {base:.4f} -> tuned {score:.4f}")
""")
interp("selection")

md(r"""
### 11.3 Final evaluation on the held-out test set

The three final models are each evaluated **once** on the same independent test set: logistic regression (E01) as the interpretable reference, plus the selected traditional and deep-learning models, all with their validation-tuned decision weights.
""")

code(r"""
FEATS = {eid: ("eng" if "engineered" in results.set_index("experiment_id").loc[eid, "features"] else "base") for eid in results.experiment_id}
def test_proba(eid):
    if eid in fitted:
        return fitted[eid].predict_proba(X_te)
    return keras_models[eid].predict(dl.make_tf_dataset(A[FEATS[eid]]["te"], batch_size=4096), verbose=0)

final_ids = ["E01", BEST_TRAD, BEST_DEEP]
names = {eid: f"{eid} {results.set_index('experiment_id').loc[eid, 'model']}" for eid in final_ids}
test_probas, test_preds, rows = {}, {}, []
for eid in final_ids:
    p = test_proba(eid); test_probas[names[eid]] = p
    yhat = ev.predict_with_weights(p, decision[eid]); test_preds[eid] = yhat
    m = ev.compute_metrics(y_te, p, y_pred=yhat)
    mw = ev.compute_metrics(y_te, p, y_pred=yhat, sample_weight=m_te["_LLCPWT"].values)
    rows.append({"model": names[eid], **{k: m[k] for k in ["accuracy", "balanced_accuracy", "precision_macro", "recall_macro",
                 "f1_macro", "roc_auc_ovr_macro", "avg_precision_macro", "recall_prediabetes", "precision_prediabetes",
                 "recall_diabetes", "precision_diabetes"]},
                 "weighted_f1_macro": mw["f1_macro"], "weighted_roc_auc": mw["roc_auc_ovr_macro"], "weighted_recall_diabetes": mw["recall_diabetes"]})
test_table = pd.DataFrame(rows).set_index("model")
test_table.to_csv(ROOT / "reports" / "test_results.csv")
test_table.T.round(4)
""")

code(r"""
for eid in final_ids:
    print(names[eid]); display(ev.report_frame(y_te, test_preds[eid]))
    ev.plot_confusion_matrices(y_te, test_preds[eid], title=f"{names[eid]} - test set (tuned decision weights)",
                               path=FIG / f"fig08_confusion_{eid}.png")
    plt.show()
""")
interp("test")

code(r"""
fig = ev.plot_roc_pr_curves(y_te, test_probas, path=FIG / "fig09_roc_pr_test.png")
""")
interp("curves")

md(r"""
### 11.4 Binary screening view: diabetes versus not

In a screening application, the most consequential decision is whether to flag someone as likely to have diabetes. I convert the selected model's diabetes probability into a binary flag. The threshold is chosen on **validation** to maximise F2, which weights recall twice as heavily as precision: a missed case (false negative) leaves a person unaware of a treatable condition, while a false positive costs an unnecessary but cheap confirmatory HbA1c or fasting-glucose test. I report the consequences on the test set alongside the default 0.5 threshold.
""")

code(r"""
score_va = val_probas[BEST_TRAD][:, 2]; score_te = test_probas[names[BEST_TRAD]][:, 2]
thr, f2_va, p_va, r_va = ev.screening_threshold((y_va == 2).astype(int), score_va, beta=2)
print(f"Validation-selected threshold: {thr:.3f} (val F2 {f2_va:.3f}, precision {p_va:.3f}, recall {r_va:.3f})")
screen = pd.DataFrame({
    "default_0.5": ev.binary_screening_metrics((y_te == 2).astype(int), score_te, 0.5),
    "F2_threshold": ev.binary_screening_metrics((y_te == 2).astype(int), score_te, thr),
    "F2_threshold_survey_weighted": ev.binary_screening_metrics((y_te == 2).astype(int), score_te, thr, sample_weight=m_te["_LLCPWT"].values),
}).T
screen.to_csv(ROOT / "reports" / "screening_results.csv")
screen.round(3)
""")
interp("screening")

# --------------------------------------------------------------------- 12
md(r"""
## 12. Error Analysis and Explainability

### 12.1 Where do the errors come from?

I compare correctly identified and missed diabetes cases on key indicators, and I check where true prediabetes cases end up. A model that misses diabetes mostly among young, normal-BMI respondents is failing in a predictable way; a model that misses cases with classic risk profiles would be failing for a different reason.
""")

code(r"""
yhat = test_preds[BEST_TRAD]
dia = (y_te == 2).values
groups = pd.Series(np.where(dia & (yhat == 2), "diabetes: detected", np.where(dia, "diabetes: missed", None)))
profile_cols = ["age", "bmi", "high_bp", "high_chol", "gen_health", "cardiometabolic_count", "diff_walk", "checkup_recency"]
Xte_eng = fe.ClinicalFeatureEngineer().fit_transform(X_te)
prof = Xte_eng[profile_cols].groupby(groups).mean().T
prof["non-diabetes (reference)"] = Xte_eng.loc[(y_te == 0).values, profile_cols].mean()
display(prof.round(2))
pre = (y_te == 1).values
print("Where true prediabetes cases are predicted:",
      pd.Series(yhat[pre]).map(dict(enumerate(prep.CLASS_NAMES))).value_counts(normalize=True).round(3).to_dict())
fp = (yhat == 2) & (y_te.values == 0)
print("Profile of false-positive diabetes flags vs true negatives (mean cardiometabolic count): %.2f vs %.2f"
      % (Xte_eng.loc[fp, "cardiometabolic_count"].mean(), Xte_eng.loc[(yhat == 0) & (y_te.values == 0), "cardiometabolic_count"].mean()))
""")
interp("errors")

md(r"""
### 12.2 Explainability

I use two complementary methods on the selected boosting model:

* **Permutation importance** on a 20,000-row validation sample: how much macro average precision drops when one raw feature is shuffled. It is model-agnostic and measured on held-out data, but correlated features share credit.
* **SHAP (TreeExplainer)** on 2,000 validation rows, which decomposes each prediction into additive feature contributions (in log-odds) for the diabetes class.

For contrast, I also show the standardised logistic-regression coefficients (diabetes vs. no diabetes). None of these measures is causal: they describe what the model relies on, in a cross-sectional survey where diabetes can also cause the "predictors", such as poorer self-rated health or a recent check-up.
""")

code(r"""
from sklearn.inspection import permutation_importance
Xva_s = X_va.sample(20_000, random_state=SEED); yva_s = y_va.loc[Xva_s.index]
t0 = time.perf_counter()
perm = permutation_importance(fitted[BEST_TRAD], Xva_s, yva_s, scoring=ev.MACRO_AP_SCORER, n_repeats=5, random_state=SEED, n_jobs=1)
perm_df = pd.DataFrame({"importance_mean": perm.importances_mean, "importance_std": perm.importances_std}, index=X_va.columns)
perm_df = perm_df.sort_values("importance_mean", ascending=False)
perm_df.to_csv(ROOT / "reports" / "permutation_importance.csv")
print(f"permutation importance computed in {time.perf_counter()-t0:.0f}s")
fig, ax = plt.subplots(figsize=(7, 6))
perm_df.importance_mean.iloc[::-1].plot.barh(xerr=perm_df.importance_std.iloc[::-1], ax=ax, color="#4C72B0")
ax.set(xlabel="drop in validation macro average precision", title=f"Permutation importance - {BEST_TRAD}")
fig.tight_layout(); fig.savefig(FIG / "fig10_permutation_importance.png", dpi=150)
perm_df.head(10).round(4)
""")

code(r"""
import shap
pre_best = tm.preprocessing_part(fitted[BEST_TRAD])
Xs = pre_best.transform(X_va.sample(2_000, random_state=SEED))
feat_names = list(pre_best[-1].get_feature_names_out())
explainer = shap.TreeExplainer(fitted[BEST_TRAD][-1])
sv = explainer.shap_values(Xs)
sv_dia = sv[:, :, 2] if np.ndim(sv) == 3 else sv[2]
plt.figure()
shap.summary_plot(sv_dia, Xs, feature_names=feat_names, max_display=15, show=False)
plt.title(f"SHAP values for the diabetes class - {BEST_TRAD}")
plt.tight_layout(); plt.savefig(FIG / "fig11_shap_diabetes.png", dpi=150, bbox_inches="tight"); plt.show()
shap_rank = pd.Series(np.abs(sv_dia).mean(axis=0), index=feat_names).sort_values(ascending=False)
shap_rank.head(12).round(4)
""")

code(r"""
lr = fitted["E01"][-1]
lr_names = list(fitted["E01"][0].get_feature_names_out())
coef = pd.DataFrame({"diabetes_vs_none": lr.coef_[2] - lr.coef_[0], "prediabetes_vs_none": lr.coef_[1] - lr.coef_[0]}, index=lr_names)
coef["odds_ratio_per_SD_diabetes"] = np.exp(coef.diabetes_vs_none)
coef = coef.reindex(coef.diabetes_vs_none.abs().sort_values(ascending=False).index)
coef.to_csv(ROOT / "reports" / "logistic_coefficients.csv")
coef.head(12).round(3)
""")
interp("explain")

md(r"""
### 12.3 Subgroup performance

I check whether diabetes detection differs across sex, age, race/ethnicity and income on the test set, using the selected model and its tuned decision rule. Race/ethnicity was never a model input. Subgroups with fewer than 30 diabetes cases are flagged as unreliable.
""")

code(r"""
age_band = pd.cut(X_te["age"], [17, 44, 64, 80], labels=["18-44", "45-64", "65+"])
sub_specs = {
    "sex": X_te["sex_female"].map({0.0: "Male", 1.0: "Female"}),
    "age": age_band.astype(str),
    "race/ethnicity": m_te["_RACEGR3"].map(prep.RACE_LABELS),
    "income": pd.cut(X_te["income"], [0, 3, 6, 8], labels=["<$20k", "$20-50k", "$50k+"]).astype(str).replace("nan", np.nan),
}
sub_tables = []
for dim, g in sub_specs.items():
    t = ev.subgroup_metrics(y_te.values, test_probas[names[BEST_TRAD]], test_preds[BEST_TRAD], g)
    t.insert(0, "dimension", dim); sub_tables.append(t)
subgroups = pd.concat(sub_tables, ignore_index=True)
subgroups.to_csv(ROOT / "reports" / "subgroup_metrics.csv", index=False)
subgroups.round(3)
""")
interp("subgroups")

# --------------------------------------------------------------------- 13
md(r"""
## 13. Discussion and Conclusions
""")
interp("discussion")

# --------------------------------------------------------------------- 14
md(r"""
## 14. Reproducibility and Model Export

I export the two selected models only now, after selection and test evaluation. The traditional model is saved as the **complete scikit-learn pipeline** (feature engineering, imputation, winsorising, encoding and the estimator), so raw cleaned features can be scored directly. The Keras model is saved in the native `.keras` format together with the fitted preprocessing pipeline it expects. A metadata JSON records features, class names, decision weights, the screening threshold, test metrics, package versions and the data hash. I then reload both artifacts and confirm they reproduce the in-memory predictions.
""")

code(r"""
import joblib, sklearn
trad_path = MODELS / "best_traditional_pipeline.joblib"
deep_path = MODELS / "best_deep_model.keras"
deep_pre_path = MODELS / "deep_model_preprocessor.joblib"
try:
    joblib.dump(fitted[BEST_TRAD], trad_path, compress=3)
    keras_models[BEST_DEEP].save(deep_path)
    joblib.dump(pre_eng if FEATS[BEST_DEEP] == "eng" else pre_base, deep_pre_path, compress=3)
except (OSError, ValueError) as exc:
    raise RuntimeError(f"Model export failed: {exc}") from exc

metadata = {
    "created_utc": pd.Timestamp.now(tz="UTC").isoformat(),
    "dataset": {"source": data_loading.CDC_LANDING_PAGE, "file": "LLCP2015.XPT", "records": report.n_records,
                "sha256": report.sha256, "analysis_sample": int(len(X_all))},
    "target": {"variable": "DIABETE3", "classes": prep.CLASS_NAMES, "mapping": {str(k): v for k, v in prep.TARGET_MAP.items()}},
    "input_features": prep.BASE_FEATURES,
    "traditional_model": {"experiment": BEST_TRAD, "file": trad_path.name,
                          "config": json.loads(results.set_index("experiment_id").loc[BEST_TRAD, "hyperparameters"]),
                          "decision_weights": decision[BEST_TRAD].tolist(), "screening_threshold_diabetes": thr,
                          "test_metrics": test_table.loc[names[BEST_TRAD]].to_dict()},
    "deep_model": {"experiment": BEST_DEEP, "file": deep_path.name, "preprocessor": deep_pre_path.name,
                   "feature_set": FEATS[BEST_DEEP], "decision_weights": decision[BEST_DEEP].tolist(),
                   "test_metrics": test_table.loc[names[BEST_DEEP]].to_dict()},
    "environment": env, "seed": SEED,
}
(MODELS / "model_metadata.json").write_text(json.dumps(metadata, indent=2, default=float))

reloaded_trad = joblib.load(trad_path)
reloaded_deep = keras.models.load_model(deep_path)
reloaded_pre = joblib.load(deep_pre_path)
p1 = reloaded_trad.predict_proba(X_te.head(1000))
p2 = reloaded_deep.predict(reloaded_pre.transform(X_te.head(1000)).astype("float32"), verbose=0)
print("Traditional pipeline reload matches:", np.allclose(p1, test_probas[names[BEST_TRAD]][:1000]))
print("Keras model reload matches:", np.allclose(p2, test_probas[names[BEST_DEEP]][:1000], atol=1e-6))
for p in [trad_path, deep_path, deep_pre_path, MODELS / "model_metadata.json"]:
    print(f"{p.name}: {p.stat().st_size/1e6:.2f} MB")
""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                  "language_info": {"name": "python"},
                  "colab": {"provenance": []}}
out = ROOT / "notebooks" / "diabetes_classification.ipynb"
nbf.write(nb, out)
print("wrote", out, "with", len(cells), "cells")
