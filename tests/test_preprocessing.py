"""Unit tests for the BRFSS recoding, feature engineering and leakage safeguards."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.feature_engineering import ENGINEERED_FEATURES, ClinicalFeatureEngineer  # noqa: E402
from src.preprocessing import (BASE_FEATURES, NOMINAL_FEATURES, NUMERIC_FEATURES,  # noqa: E402
                               QuantileClipper, build_preprocessor, clean_brfss, map_target,
                               stratified_split)


def make_raw(n=6):
    """Small synthetic frame shaped like the raw BRFSS extract (test fixture only)."""
    base = {
        "SEQNO": np.arange(n), "_STATE": 1.0, "_PSU": np.arange(n), "_STSTR": 11011.0, "_LLCPWT": 100.0,
        "DISPCODE": 1100.0, "_RACEGR3": 1.0, "_AGEG5YR": 5.0,
        "DIABETE3": [3.0, 1.0, 4.0, 2.0, 7.0, 9.0][:n],
        "_RFHYPE5": [1.0, 2.0, 9.0, 1.0, 2.0, 1.0][:n],
        "TOLDHI2": [1.0, 2.0, np.nan, 7.0, 1.0, 2.0][:n],
        "BLOODCHO": [1.0, 1.0, 2.0, 1.0, 1.0, 1.0][:n],
        "_CHOLCHK": [1.0, 2.0, 3.0, 9.0, 1.0, 1.0][:n],
        "_BMI5": [2500.0, 3550.0, np.nan, 1800.0, 9995.0, 2200.0][:n],
        "_SMOKER3": [1.0, 2.0, 3.0, 4.0, 9.0, 4.0][:n],
        "CVDSTRK3": [2.0, 1.0, 7.0, 2.0, 2.0, 9.0][:n],
        "_MICHD": [2.0, 1.0, np.nan, 2.0, 2.0, 2.0][:n],
        "_TOTINDA": [1.0, 2.0, 9.0, 1.0, 1.0, 1.0][:n],
        "_FRTLT1": [1.0, 2.0, 9.0, 1.0, 1.0, 1.0][:n],
        "_VEGLT1": [1.0, 2.0, 9.0, 1.0, 1.0, 1.0][:n],
        "_RFDRHV5": [1.0, 2.0, 9.0, 1.0, 1.0, 1.0][:n],
        "HLTHPLN1": [1.0, 2.0, 7.0, 1.0, 1.0, 1.0][:n],
        "MEDCOST": [2.0, 1.0, 9.0, 2.0, 2.0, 2.0][:n],
        "GENHLTH": [1.0, 5.0, 7.0, 3.0, 2.0, 9.0][:n],
        "MENTHLTH": [88.0, 30.0, 77.0, 5.0, 99.0, 88.0][:n],
        "PHYSHLTH": [88.0, 15.0, 99.0, 30.0, 77.0, 2.0][:n],
        "DIFFWALK": [2.0, 1.0, np.nan, 2.0, 2.0, 7.0][:n],
        "SEX": [1.0, 2.0, 1.0, 2.0, 1.0, 2.0][:n],
        "_AGE80": [30.0, 70.0, 55.0, 41.0, 80.0, 18.0][:n],
        "EDUCA": [6.0, 4.0, 9.0, 5.0, 3.0, 6.0][:n],
        "INCOME2": [8.0, 3.0, 77.0, 99.0, 5.0, 1.0][:n],
        "CHCKIDNY": [2.0, 1.0, 7.0, 2.0, 2.0, 2.0][:n],
        "HAVARTH3": [2.0, 1.0, 9.0, 1.0, 2.0, 2.0][:n],
        "ADDEPEV2": [2.0, 1.0, 7.0, 2.0, 2.0, 2.0][:n],
        "CHECKUP1": [1.0, 2.0, 8.0, 4.0, 7.0, 9.0][:n],
    }
    return pd.DataFrame(base)


def test_target_mapping_follows_codebook():
    y = map_target(pd.Series([1.0, 2.0, 3.0, 4.0, 7.0, 9.0, np.nan]))
    assert y.tolist()[:4] == [2, 0, 0, 1]
    assert y.iloc[4:].isna().all()


def test_clean_drops_unusable_targets_and_keeps_alignment():
    X, y, meta = clean_brfss(make_raw())
    assert len(X) == len(y) == len(meta) == 4  # DIABETE3 7 and 9 removed
    assert y.tolist() == [0, 2, 1, 0]
    assert list(X.columns) == BASE_FEATURES


def test_special_codes_become_missing():
    X, _, _ = clean_brfss(make_raw())
    assert np.isnan(X.loc[2, "high_bp"])          # _RFHYPE5 == 9
    assert np.isnan(X.loc[2, "gen_health"])       # GENHLTH == 7
    assert np.isnan(X.loc[2, "income"])           # INCOME2 == 77
    assert np.isnan(X.loc[3, "income"])           # INCOME2 == 99
    assert np.isnan(X.loc[2, "ment_days"])        # 77 don't know
    assert X.loc[0, "ment_days"] == 0             # 88 -> none -> 0 days
    assert X.loc[1, "phys_days"] == 15


def test_high_cholesterol_skip_pattern():
    X, _, _ = clean_brfss(make_raw())
    assert X.loc[0, "high_chol"] == 1
    assert X.loc[1, "high_chol"] == 0
    assert X.loc[2, "high_chol"] == 0              # never checked -> not told high
    assert np.isnan(X.loc[3, "high_chol"])         # don't know


def test_bmi_scaling_and_checkup_never():
    X, _, _ = clean_brfss(make_raw())
    assert X.loc[0, "bmi"] == pytest.approx(25.0)
    assert np.isnan(X.loc[2, "bmi"])
    assert X.loc[2, "checkup_recency"] == 5       # CHECKUP1 == 8 (never)


def test_feature_engineering_values():
    X, _, _ = clean_brfss(make_raw())
    Z = ClinicalFeatureEngineer().fit_transform(X)
    assert set(ENGINEERED_FEATURES) <= set(Z.columns)
    assert Z.loc[0, "bmi_category"] == 2           # BMI 25.0 -> overweight
    assert Z.loc[1, "bmi_category"] == 4           # BMI 35.5 -> obese class II
    assert Z.loc[1, "cardiometabolic_count"] == 4  # BP, heart, stroke, kidney (chol=0)
    assert Z.loc[1, "unhealthy_days"] == 30        # 15 + 30 capped at 30
    assert Z.loc[1, "access_barrier"] == 1


def test_quantile_clipper_uses_training_bounds_only():
    train = np.array([[1.0], [2.0], [3.0], [4.0], [100.0]])
    clip = QuantileClipper(lower=0.0, upper=0.75).fit(train)
    out = clip.transform(np.array([[1000.0], [-5.0]]))
    assert out[0, 0] == pytest.approx(4.0)
    assert out[1, 0] == pytest.approx(1.0)


def test_preprocessor_outputs_no_missing_values():
    X, _, _ = clean_brfss(make_raw())
    pre = build_preprocessor(NUMERIC_FEATURES, NOMINAL_FEATURES, scale=True)
    Z = pre.fit_transform(X)
    assert not np.isnan(Z).any()
    assert Z.shape[0] == len(X)


def test_stratified_split_preserves_class_proportions():
    rng = np.random.default_rng(0)
    n = 2000
    X = pd.DataFrame({"a": rng.normal(size=n)})
    y = pd.Series(rng.choice([0, 1, 2], size=n, p=[0.85, 0.02, 0.13]))
    meta = pd.DataFrame({"w": np.ones(n)})
    s = stratified_split(X, y, meta)
    sizes = {k: len(v[0]) for k, v in s.items()}
    assert sizes == {"train": 1400, "val": 300, "test": 300}
    for part in s.values():
        assert abs(part[1].eq(2).mean() - y.eq(2).mean()) < 0.01
    idx = [set(v[0]["a"]) for v in s.values()]
    assert not (idx[0] & idx[1]) and not (idx[0] & idx[2]) and not (idx[1] & idx[2])
