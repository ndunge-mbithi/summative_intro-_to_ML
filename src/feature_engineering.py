"""Domain-motivated engineered features.

The transformer is stateless (it learns nothing from the data), so placing it
at the start of a pipeline cannot leak information across splits.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

ENGINEERED_FEATURES = [
    "bmi_category", "cardiometabolic_count", "lifestyle_risk",
    "unhealthy_days", "age_bmi", "access_barrier",
]

# WHO adult BMI cut-points: underweight, normal, overweight, obese I-III.
BMI_BINS = [-np.inf, 18.5, 25.0, 30.0, 35.0, 40.0, np.inf]


def _nan_sum(df: pd.DataFrame) -> pd.Series:
    """Row sum that stays NaN only when every component is missing."""
    out = df.sum(axis=1, min_count=1)
    return out


class ClinicalFeatureEngineer(BaseEstimator, TransformerMixin):
    """Append engineered risk features to the cleaned feature frame.

    * bmi_category - WHO obesity class (0 underweight .. 5 obese class III);
    * cardiometabolic_count - count of diagnosed high BP, high cholesterol,
      heart disease, stroke and kidney disease;
    * lifestyle_risk - current smoking, inactivity, <1 fruit/day,
      <1 vegetable/day and heavy drinking;
    * unhealthy_days - physical + mental unhealthy days, capped at 30
      (the CDC 'healthy days' construct);
    * age_bmi - age x BMI interaction (scaled /100);
    * access_barrier - no health plan or cost prevented care.
    """

    def fit(self, X, y=None):
        self.feature_names_in_ = np.asarray(list(X.columns), dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X["bmi_category"] = pd.cut(X["bmi"], BMI_BINS, labels=False, right=False).astype(float)
        X["cardiometabolic_count"] = _nan_sum(
            X[["high_bp", "high_chol", "heart_disease", "stroke", "kidney_disease"]])
        current_smoker = X["smoker_status"].isin(["current_daily", "current_someday"]).astype(float)
        current_smoker = current_smoker.where(X["smoker_status"].notna())
        X["lifestyle_risk"] = _nan_sum(pd.DataFrame({
            "smoke": current_smoker,
            "inactive": 1 - X["phys_activity"],
            "low_fruit": 1 - X["fruits_daily"],
            "low_veg": 1 - X["veggies_daily"],
            "drink": X["heavy_drinker"],
        }))
        X["unhealthy_days"] = (X["phys_days"] + X["ment_days"]).clip(upper=30)
        X["age_bmi"] = X["age"] * X["bmi"] / 100.0
        no_plan = 1 - X["has_healthplan"]
        X["access_barrier"] = np.fmax(no_plan, X["cost_barrier"])
        return X

    def get_feature_names_out(self, input_features=None):
        base = list(input_features) if input_features is not None else list(self.feature_names_in_)
        return np.asarray(base + ENGINEERED_FEATURES, dtype=object)
