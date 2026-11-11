"""Inference helpers used by the Streamlit app (and usable from scripts)."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .config import CRITICAL_RUL, WARNING_RUL
from .data_loader import last_row_per_unit, validate_history
from .features import build_features


def _resolve(path: str) -> Path:
    """Return a local path; download first if `path` is a gs:// URI."""
    if not path.startswith("gs://"):
        return Path(path)
    from google.cloud import storage  # imported lazily: only needed on GCP

    bucket_name, _, blob_name = path[5:].partition("/")
    local = Path(tempfile.gettempdir()) / Path(blob_name).name
    storage.Client().bucket(bucket_name).blob(blob_name).download_to_filename(local)
    return local


def load_bundle(path: str | None = None) -> dict:
    """Load model_bundle.joblib from a local path or gs://bucket/path."""
    path = path or os.environ.get("MODEL_PATH", "artifacts/model_bundle.joblib")
    return joblib.load(_resolve(path))


def health_status(rul: float) -> str:
    if rul <= CRITICAL_RUL:
        return "Critical"
    if rul <= WARNING_RUL:
        return "Warning"
    return "Healthy"


def predict_history(bundle: dict, history: pd.DataFrame, every_cycle: bool = False) -> pd.DataFrame:
    """Predict RUL from raw sensor history.

    every_cycle=False -> one row per engine (its latest cycle).
    every_cycle=True  -> a prediction for every cycle (RUL trajectory).
    """
    clean = validate_history(history, bundle["base_cols"])
    feats = build_features(clean, bundle["base_cols"], bundle["windows"])
    if not every_cycle:
        feats = last_row_per_unit(feats)
    pred = bundle["model"].predict(feats[bundle["feature_columns"]])
    cap = bundle.get("rul_cap")
    pred = np.clip(pred, 0, cap) if cap else np.clip(pred, 0, None)
    out = feats[["unit", "cycle"]].copy()
    out["predicted_rul"] = pred.round(1)
    out["status"] = out["predicted_rul"].map(health_status)
    return out.reset_index(drop=True)
