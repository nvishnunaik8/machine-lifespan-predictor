"""Reusable building blocks shared by train.py and ablation.py."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import RANDOM_STATE
from .data_loader import add_rul, last_row_per_unit
from .features import build_features, feature_columns
from .metrics import regression_report


def make_models(seed: int = RANDOM_STATE) -> dict[str, BaseEstimator]:
    """Candidate regressors (trees need no scaling; Ridge gets a scaler)."""
    return {
        "Ridge (baseline)": make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
        # max_samples subsamples rows per tree: ~5x faster than a full bootstrap
        # on a 2-vCPU machine (Colab / small VMs) with similar accuracy.
        "Random Forest": RandomForestRegressor(
            n_estimators=200, min_samples_leaf=5, max_features=0.2, max_samples=0.5,
            n_jobs=-1, random_state=seed,
        ),
        "Extra Trees": ExtraTreesRegressor(
            n_estimators=300, min_samples_leaf=3, max_features=0.33, n_jobs=-1, random_state=seed
        ),
        "Hist Gradient Boosting": HistGradientBoostingRegressor(
            max_iter=400, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0, random_state=seed
        ),
    }


@dataclass
class Dataset:
    X: pd.DataFrame
    y: pd.Series
    groups: pd.Series
    columns: list[str]


def prepare_training(train: pd.DataFrame, base_cols: Sequence[str], windows: Sequence[int], cap: int | None) -> Dataset:
    """Build X, y and engine groups from raw training data."""
    train = train.sort_values(["unit", "cycle"]).reset_index(drop=True)
    y = add_rul(train, cap)["RUL"]
    feats = build_features(train, base_cols, windows)
    cols = feature_columns(feats)
    return Dataset(X=feats[cols], y=y, groups=feats["unit"], columns=cols)


def test_snapshot(test: pd.DataFrame, base_cols: Sequence[str], windows: Sequence[int], columns: list[str]) -> pd.DataFrame:
    """Features of each test engine's LAST observed cycle (ordered by unit id)."""
    feats = build_features(test, base_cols, windows)
    return last_row_per_unit(feats)[["unit", *columns]]


def fit_and_evaluate(
    model: BaseEstimator,
    train: pd.DataFrame,
    test: pd.DataFrame,
    rul_test: pd.Series,
    base_cols: Sequence[str],
    windows: Sequence[int],
    cap: int | None,
):
    """Fit on all training engines, score on the official test snapshot."""
    ds = prepare_training(train, base_cols, windows, cap)
    fitted = clone(model).fit(ds.X, ds.y)
    snap = test_snapshot(test, base_cols, windows, ds.columns)
    pred = fitted.predict(snap[ds.columns])
    if cap is not None:
        pred = np.clip(pred, 0, cap)
    metrics = regression_report(rul_test.to_numpy(), pred)
    return fitted, ds, snap, pred, metrics


def stage_breakdown(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    """MAE by true-RUL stage: does the model stay accurate near failure?"""
    bins = pd.cut(y_true, [-np.inf, 50, 100, np.inf], labels=["late (RUL<=50)", "mid (50-100)", "early (>100)"])
    df = pd.DataFrame({"stage": bins, "abs_err": np.abs(y_true - y_pred)})
    out = df.groupby("stage", observed=True)["abs_err"].agg(["count", "mean"]).rename(columns={"mean": "MAE"})
    return out.reset_index()
