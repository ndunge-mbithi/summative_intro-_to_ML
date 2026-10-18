"""Scikit-learn pipelines for the traditional machine learning experiments."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from .evaluation import MACRO_AP_SCORER
from .feature_engineering import ENGINEERED_FEATURES, ClinicalFeatureEngineer
from .preprocessing import NOMINAL_FEATURES, NUMERIC_FEATURES, build_preprocessor
from .utils import SEED


def build_pipeline(estimator, scale: bool = True, engineered: bool = False) -> Pipeline:
    """Feature engineering (optional) -> preprocessing -> estimator."""
    numeric = NUMERIC_FEATURES + (ENGINEERED_FEATURES if engineered else [])
    steps = []
    if engineered:
        steps.append(("engineer", ClinicalFeatureEngineer()))
    steps.append(("preprocess", build_preprocessor(numeric, NOMINAL_FEATURES, scale=scale)))
    steps.append(("model", estimator))
    return Pipeline(steps)


def build_preprocessing_pipeline(scale: bool = True, engineered: bool = False) -> Pipeline:
    """The same feature engineering + preprocessing steps, without an estimator.

    Used to produce model-ready arrays for the Keras models and to inspect
    transformed features (e.g. for SHAP).
    """
    full = build_pipeline("passthrough", scale=scale, engineered=engineered)
    return Pipeline(full.steps[:-1])


def preprocessing_part(pipeline: Pipeline) -> Pipeline:
    """Return the fitted preprocessing steps of a fitted model pipeline."""
    return Pipeline(pipeline.steps[:-1])


def make_logistic_regression(C: float = 1.0, class_weight=None, max_iter: int = 1000) -> LogisticRegression:
    """Multinomial logistic regression with L2 penalty (lbfgs solver)."""
    return LogisticRegression(C=C, class_weight=class_weight, max_iter=max_iter,
                              solver="lbfgs", random_state=SEED)


def make_random_forest(n_estimators: int = 150, max_depth=None, min_samples_leaf: int = 5,
                       max_features="sqrt", class_weight=None, n_jobs: int = -1) -> RandomForestClassifier:
    return RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                  min_samples_leaf=min_samples_leaf, max_features=max_features,
                                  class_weight=class_weight, n_jobs=n_jobs, random_state=SEED)


def make_gradient_boosting(learning_rate: float = 0.1, max_iter: int = 200, max_leaf_nodes: int = 31,
                           min_samples_leaf: int = 20, l2_regularization: float = 0.0,
                           class_weight=None, early_stopping: bool = False) -> HistGradientBoostingClassifier:
    """Histogram-based gradient boosting (scikit-learn's scalable GBDT implementation)."""
    return HistGradientBoostingClassifier(learning_rate=learning_rate, max_iter=max_iter,
                                          max_leaf_nodes=max_leaf_nodes, min_samples_leaf=min_samples_leaf,
                                          l2_regularization=l2_regularization, class_weight=class_weight,
                                          early_stopping=early_stopping, random_state=SEED)


def fit_timed(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series, **fit_params) -> tuple[Pipeline, float]:
    """Fit a pipeline and return it with the wall-clock training time in seconds."""
    start = time.perf_counter()
    pipeline.fit(X, y, **fit_params)
    return pipeline, time.perf_counter() - start


def stratified_subsample(X: pd.DataFrame, y: pd.Series, n: int, seed: int = SEED):
    """Class-stratified subsample used to keep cross-validated searches affordable."""
    if n >= len(X):
        return X, y
    frac = n / len(X)
    idx = (y.groupby(y, group_keys=False)
             .apply(lambda s: s.sample(frac=frac, random_state=seed)).index)
    idx = np.sort(idx)
    return X.loc[idx].reset_index(drop=True), y.loc[idx].reset_index(drop=True)


def randomized_search(pipeline: Pipeline, param_distributions: dict, X: pd.DataFrame, y: pd.Series,
                      n_iter: int = 12, cv_folds: int = 3, scoring=MACRO_AP_SCORER,
                      n_jobs: int = -1, seed: int = SEED) -> tuple[RandomizedSearchCV, float]:
    """Randomized hyper-parameter search with stratified CV on training data only."""
    search = RandomizedSearchCV(pipeline, param_distributions, n_iter=n_iter, scoring=scoring,
                                cv=StratifiedKFold(cv_folds, shuffle=True, random_state=seed),
                                n_jobs=n_jobs, random_state=seed, refit=False, return_train_score=True)
    start = time.perf_counter()
    search.fit(X, y)
    return search, time.perf_counter() - start


def search_results_frame(search: RandomizedSearchCV) -> pd.DataFrame:
    """Tidy summary of a completed search, best configuration first."""
    res = pd.DataFrame(search.cv_results_)
    keep = [c for c in res.columns if c.startswith("param_")] + [
        "mean_train_score", "mean_test_score", "std_test_score", "mean_fit_time", "rank_test_score"]
    return res[keep].sort_values("rank_test_score").reset_index(drop=True)
