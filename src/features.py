"""Leakage-safe, vectorised feature engineering.

Every feature at cycle t uses observations from cycles <= t of the SAME engine
only (trailing windows), so the pipeline never looks at the future and can be
re-used unchanged for online prediction.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from .config import MIN_STD, SENSOR_COLS, SETTING_COLS, WINDOWS


def select_informative_columns(train: pd.DataFrame, min_std: float = MIN_STD) -> list[str]:
    """Settings/sensors that actually vary in the TRAINING data (drops constants)."""
    candidates = SETTING_COLS + SENSOR_COLS
    std = train[candidates].std()
    return std[std > min_std].index.tolist()


def build_features(
    df: pd.DataFrame,
    base_cols: Sequence[str],
    windows: Sequence[int] = WINDOWS,
) -> pd.DataFrame:
    """Return [unit, cycle, base cols, rolling mean/std/slope per window].

    Rows are sorted by (unit, cycle); the output keeps one row per input row.
    """
    base_cols = list(base_cols)
    df = df.sort_values(["unit", "cycle"]).reset_index(drop=True)
    grouped = df.groupby("unit", sort=False)[base_cols]

    parts = [df[["unit", "cycle", *base_cols]]]
    for w in windows:
        roll = grouped.rolling(window=w, min_periods=1)
        mean = roll.mean().reset_index(level=0, drop=True).add_suffix(f"_mean_{w}")
        std = roll.std().reset_index(level=0, drop=True).fillna(0.0).add_suffix(f"_std_{w}")
        # Average change per cycle across the window: (x_t - x_{t-w+1}) / (w-1)
        slope = (grouped.diff(periods=w - 1) / max(w - 1, 1)).fillna(0.0).add_suffix(f"_slope_{w}")
        parts.extend([mean, std, slope])

    feats = pd.concat(parts, axis=1)
    value_cols = feats.columns.difference(["unit", "cycle"])
    feats[value_cols] = feats[value_cols].astype(np.float32)
    return feats


def feature_columns(feats: pd.DataFrame) -> list[str]:
    """Model input columns: everything except the engine id."""
    return [c for c in feats.columns if c != "unit"]
