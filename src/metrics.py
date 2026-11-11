"""Evaluation metrics for RUL regression."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def nasa_score(y_true, y_pred) -> float:
    """Asymmetric PHM08 score (lower is better). Late predictions cost more.

    d = y_pred - y_true;  s = exp(-d/13) - 1 if d < 0 else exp(d/10) - 1
    """
    d = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)))


def regression_report(y_true, y_pred) -> dict[str, float]:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
        "NASA_score": nasa_score(y_true, y_pred),
    }
