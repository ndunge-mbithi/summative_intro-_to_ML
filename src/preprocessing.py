"""Cleaning, target definition, leakage-safe preprocessing and data splitting.

All recodes follow the 2015 BRFSS codebook. Special survey codes such as
7/77 ("don't know"), 9/99 ("refused") and calculated-variable code 9
("don't know/refused/missing") are converted to NaN rather than treated as
measurements. Imputation, clipping and scaling are fitted inside
scikit-learn pipelines on the training data only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .utils import SEED

CLASS_NAMES = ["No diabetes", "Prediabetes", "Diabetes"]

# DIABETE3 -> 3-class target. Code 2 ("yes, but female told only during
# pregnancy") is mapped to "No diabetes", following CDC's standard
# diabetes-indicator convention; 7, 9 and blank are unusable labels.
TARGET_MAP = {3.0: 0, 2.0: 0, 4.0: 1, 1.0: 2}

NOMINAL_FEATURES = ["smoker_status"]
BINARY_FEATURES = [
    "high_bp", "high_chol", "chol_check", "stroke", "heart_disease", "phys_activity",
    "fruits_daily", "veggies_daily", "heavy_drinker", "has_healthplan", "cost_barrier",
    "diff_walk", "sex_female", "kidney_disease", "arthritis", "depression",
]
ORDINAL_FEATURES = ["gen_health", "education", "income", "checkup_recency"]
CONTINUOUS_FEATURES = ["bmi", "age", "ment_days", "phys_days"]
NUMERIC_FEATURES = BINARY_FEATURES + ORDINAL_FEATURES + CONTINUOUS_FEATURES
BASE_FEATURES = NUMERIC_FEATURES + NOMINAL_FEATURES

# Columns kept alongside the features for weighting and subgroup evaluation only.
META_COLUMNS = ["_LLCPWT", "_STSTR", "_PSU", "_STATE", "_RACEGR3", "_AGEG5YR", "DIABETE3"]

RACE_LABELS = {1: "White NH", 2: "Black NH", 3: "Other NH", 4: "Multiracial NH", 5: "Hispanic"}


def _yes_no(series: pd.Series, yes: float = 1.0, no: float = 2.0) -> pd.Series:
    """Map a BRFSS yes/no item to 1/0, everything else (7, 9, blank) to NaN."""
    return series.map({yes: 1.0, no: 0.0})


def _days(series: pd.Series) -> pd.Series:
    """Map 'number of days' items: 1-30 kept, 88 ('none') -> 0, 77/99/blank -> NaN."""
    out = series.where(series.between(1, 30))
    out = out.mask(series == 88, 0.0)
    return out.astype(float)


def map_target(diabete3: pd.Series) -> pd.Series:
    """Convert DIABETE3 to the 3-class target (NaN for unusable responses)."""
    return diabete3.map(TARGET_MAP)


def clean_brfss(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Recode raw BRFSS variables into analysis features.

    Returns ``(X, y, meta)`` restricted to respondents with a usable target.
    Missing predictor values are left as NaN for the pipeline to impute.
    """
    df = raw.copy()
    X = pd.DataFrame(index=df.index)
    X["high_bp"] = df["_RFHYPE5"].map({2.0: 1.0, 1.0: 0.0})
    # TOLDHI2 is only asked when cholesterol was ever checked (BLOODCHO == 1).
    # Respondents never checked (BLOODCHO == 2) have not been told they have
    # high cholesterol, so they are coded 0 rather than missing.
    high_chol = _yes_no(df["TOLDHI2"])
    high_chol = high_chol.mask(df["BLOODCHO"] == 2.0, 0.0)
    X["high_chol"] = high_chol
    X["chol_check"] = df["_CHOLCHK"].map({1.0: 1.0, 2.0: 0.0, 3.0: 0.0})
    X["stroke"] = _yes_no(df["CVDSTRK3"])
    X["heart_disease"] = _yes_no(df["_MICHD"])
    X["phys_activity"] = _yes_no(df["_TOTINDA"])
    X["fruits_daily"] = _yes_no(df["_FRTLT1"])
    X["veggies_daily"] = _yes_no(df["_VEGLT1"])
    X["heavy_drinker"] = df["_RFDRHV5"].map({2.0: 1.0, 1.0: 0.0})
    X["has_healthplan"] = _yes_no(df["HLTHPLN1"])
    X["cost_barrier"] = _yes_no(df["MEDCOST"])
    X["diff_walk"] = _yes_no(df["DIFFWALK"])
    X["sex_female"] = df["SEX"].map({2.0: 1.0, 1.0: 0.0})
    X["kidney_disease"] = _yes_no(df["CHCKIDNY"])
    X["arthritis"] = _yes_no(df["HAVARTH3"])
    X["depression"] = _yes_no(df["ADDEPEV2"])
    X["gen_health"] = df["GENHLTH"].where(df["GENHLTH"].between(1, 5))  # 1 excellent .. 5 poor
    X["education"] = df["EDUCA"].where(df["EDUCA"].between(1, 6))
    X["income"] = df["INCOME2"].where(df["INCOME2"].between(1, 8))
    # CHECKUP1: 1 within past year ... 4 five or more years; 8 never -> 5.
    X["checkup_recency"] = df["CHECKUP1"].where(df["CHECKUP1"].between(1, 4)).mask(df["CHECKUP1"] == 8, 5.0)
    X["bmi"] = df["_BMI5"] / 100.0  # codebook: two implied decimal places
    X["age"] = df["_AGE80"]  # CDC-imputed age, top-coded at 80
    X["ment_days"] = _days(df["MENTHLTH"])
    X["phys_days"] = _days(df["PHYSHLTH"])
    # _SMOKER3: 1 every day, 2 some days, 3 former, 4 never, 9 missing.
    X["smoker_status"] = df["_SMOKER3"].map({1.0: "current_daily", 2.0: "current_someday",
                                             3.0: "former", 4.0: "never"})
    X = X.astype({c: float for c in NUMERIC_FEATURES})

    y = map_target(df["DIABETE3"])
    meta = df[[c for c in META_COLUMNS if c in df.columns]].copy()
    keep = y.notna()
    return (X.loc[keep].reset_index(drop=True),
            y.loc[keep].astype(int).reset_index(drop=True),
            meta.loc[keep].reset_index(drop=True))


class QuantileClipper(BaseEstimator, TransformerMixin):
    """Winsorise numeric columns at quantiles learned from the training data.

    Extreme values are kept (not deleted) but capped, which limits their
    leverage on linear models and neural networks without discarding rows.
    """

    def __init__(self, lower: float = 0.001, upper: float = 0.999):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        arr = np.asarray(X, dtype=float)
        self.lower_ = np.nanquantile(arr, self.lower, axis=0)
        self.upper_ = np.nanquantile(arr, self.upper, axis=0)
        self.n_features_in_ = arr.shape[1]
        return self

    def transform(self, X):
        arr = np.asarray(X, dtype=float)
        return np.clip(arr, self.lower_, self.upper_)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)


def build_preprocessor(numeric_features: list[str], nominal_features: list[str] = NOMINAL_FEATURES,
                       scale: bool = True, clip_continuous: list[str] | None = None) -> ColumnTransformer:
    """Create the leakage-safe ColumnTransformer used by every model.

    * numeric: optional train-fitted winsorising of continuous columns, median
      imputation with missing-value indicator columns, optional standardisation;
    * nominal: explicit 'missing' category + one-hot encoding.
    """
    clip_continuous = [c for c in (clip_continuous if clip_continuous is not None else CONTINUOUS_FEATURES)
                       if c in numeric_features]
    other_numeric = [c for c in numeric_features if c not in clip_continuous]

    def numeric_steps(clip: bool):
        steps = []
        if clip:
            steps.append(("clip", QuantileClipper()))
        steps.append(("impute", SimpleImputer(strategy="median", add_indicator=True)))
        if scale:
            steps.append(("scale", StandardScaler()))
        return Pipeline(steps)

    transformers = []
    if clip_continuous:
        transformers.append(("continuous", numeric_steps(True), clip_continuous))
    if other_numeric:
        transformers.append(("numeric", numeric_steps(False), other_numeric))
    if nominal_features:
        transformers.append(("nominal", Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), nominal_features))
    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)


def stratified_split(X: pd.DataFrame, y: pd.Series, meta: pd.DataFrame,
                     sizes: tuple[float, float, float] = (0.70, 0.15, 0.15), seed: int = SEED):
    """Stratified train/validation/test split that keeps X, y and meta aligned."""
    if not np.isclose(sum(sizes), 1.0):
        raise ValueError("Split sizes must sum to 1")
    train_size, val_size, test_size = sizes
    X_tr, X_tmp, y_tr, y_tmp, m_tr, m_tmp = train_test_split(
        X, y, meta, test_size=val_size + test_size, stratify=y, random_state=seed)
    rel_test = test_size / (val_size + test_size)
    X_va, X_te, y_va, y_te, m_va, m_te = train_test_split(
        X_tmp, y_tmp, m_tmp, test_size=rel_test, stratify=y_tmp, random_state=seed)
    return {
        "train": (X_tr.reset_index(drop=True), y_tr.reset_index(drop=True), m_tr.reset_index(drop=True)),
        "val": (X_va.reset_index(drop=True), y_va.reset_index(drop=True), m_va.reset_index(drop=True)),
        "test": (X_te.reset_index(drop=True), y_te.reset_index(drop=True), m_te.reset_index(drop=True)),
    }


def missingness_table(X: pd.DataFrame) -> pd.DataFrame:
    """Count and percentage of missing values per feature, sorted descending."""
    miss = X.isna().sum()
    return (pd.DataFrame({"missing": miss, "percent": (100 * miss / len(X)).round(2)})
            .sort_values("missing", ascending=False))
