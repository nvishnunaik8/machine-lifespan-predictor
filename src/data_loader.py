"""Loading, labelling and validating C-MAPSS data."""
from __future__ import annotations

from pathlib import Path
from typing import IO, Union

import pandas as pd

from .config import INDEX_COLS, RAW_COLUMNS, SENSOR_COLS, SETTING_COLS

PathOrFile = Union[str, Path, IO]


def _read_whitespace_file(path: PathOrFile, names: list[str]) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", header=None, names=names)
    return df.dropna(axis=1, how="all") if len(df.columns) != len(names) else df


def load_cmapss(data_dir: str | Path, subset: str = "FD001"):
    """Return (train, test, rul_test) for one C-MAPSS subset.

    Files expected in `data_dir`: train_<subset>.txt, test_<subset>.txt,
    RUL_<subset>.txt (space separated, no header).
    """
    data_dir = Path(data_dir)
    train = _read_whitespace_file(data_dir / f"train_{subset}.txt", RAW_COLUMNS)
    test = _read_whitespace_file(data_dir / f"test_{subset}.txt", RAW_COLUMNS)
    rul = pd.read_csv(data_dir / f"RUL_{subset}.txt", sep=r"\s+", header=None, names=["RUL"])

    value_cols = SETTING_COLS + SENSOR_COLS
    for df in (train, test):
        df[INDEX_COLS] = df[INDEX_COLS].astype("int32")
        df[value_cols] = df[value_cols].astype("float32")
    return train, test, rul


def add_rul(train: pd.DataFrame, cap: int | None = None) -> pd.DataFrame:
    """Add RUL = (last cycle of the engine) - (current cycle), optionally capped."""
    out = train.copy()
    last_cycle = out.groupby("unit")["cycle"].transform("max")
    out["RUL"] = (last_cycle - out["cycle"]).astype("float32")
    if cap is not None:
        out["RUL"] = out["RUL"].clip(upper=cap)
    return out


def parse_uploaded(file: PathOrFile) -> pd.DataFrame:
    """Parse a user-supplied sensor history (CSV with header, or raw C-MAPSS txt).

    A missing `unit` column is interpreted as a single engine (unit = 1).
    """
    try:
        df = pd.read_csv(file, sep=None, engine="python")
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Could not read the file: {exc}") from exc

    if "cycle" not in df.columns:
        # Probably headerless raw C-MAPSS data.
        if hasattr(file, "seek"):
            file.seek(0)
        raw = pd.read_csv(file, sep=r"\s+", header=None)
        if raw.shape[1] == len(RAW_COLUMNS):
            raw.columns = RAW_COLUMNS
            df = raw
        elif raw.shape[1] == len(RAW_COLUMNS) - 1:  # no unit column
            raw.columns = RAW_COLUMNS[1:]
            df = raw
        else:
            raise ValueError(
                f"Expected a 'cycle' column or {len(RAW_COLUMNS)} headerless columns, "
                f"found {raw.shape[1]} columns."
            )
    if "unit" not in df.columns:
        df.insert(0, "unit", 1)
    return df


def validate_history(df: pd.DataFrame, required: list[str]) -> pd.DataFrame:
    """Check required columns / numeric types and return a clean, sorted copy."""
    missing = [c for c in ["unit", "cycle", *required] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing[:10]}{'...' if len(missing) > 10 else ''}")
    out = df[["unit", "cycle", *required]].copy()
    out = out.apply(pd.to_numeric, errors="coerce")
    if out.isna().any().any():
        bad = out.columns[out.isna().any()].tolist()
        raise ValueError(f"Non-numeric or missing values found in: {bad[:10]}")
    out[["unit", "cycle"]] = out[["unit", "cycle"]].astype("int32")
    if out.duplicated(["unit", "cycle"]).any():
        raise ValueError("Duplicate (unit, cycle) rows found.")
    return out.sort_values(["unit", "cycle"]).reset_index(drop=True)


def last_row_per_unit(df: pd.DataFrame) -> pd.DataFrame:
    """Keep each engine's latest observation (assumes df sorted by unit, cycle)."""
    return df.groupby("unit", sort=True).tail(1).reset_index(drop=True)


__all__ = [
    "load_cmapss",
    "add_rul",
    "parse_uploaded",
    "validate_history",
    "last_row_per_unit",
]
