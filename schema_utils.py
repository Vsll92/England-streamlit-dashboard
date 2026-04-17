from __future__ import annotations

import ast
import importlib.util
import re
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


def safe_get_column(df: pd.DataFrame, candidates: Sequence[str], default: str | None = None) -> str | None:
    for col in candidates:
        if col and col in df.columns:
            return col
    return default


def get_series(
    df: pd.DataFrame,
    candidates: Sequence[str],
    default=np.nan,
    dtype: str | None = None,
) -> pd.Series:
    col = safe_get_column(df, candidates)
    if col is None:
        s = pd.Series([default] * len(df), index=df.index)
    else:
        s = df[col]
    if dtype == "numeric":
        return safe_numeric_cast(s, default=default)
    if dtype == "date":
        return safe_date_parse(s)
    return s


def add_missing_placeholder_columns(df: pd.DataFrame, placeholders: dict[str, object]) -> pd.DataFrame:
    out = df.copy()
    for col, value in placeholders.items():
        if col not in out.columns:
            out[col] = value
    return out


def validate_required_columns(df: pd.DataFrame, required: Sequence[str]) -> list[str]:
    return [col for col in required if col not in df.columns]


def safe_numeric_cast(series: pd.Series, default=np.nan) -> pd.Series:
    out = pd.to_numeric(series, errors="coerce")
    if not pd.isna(default):
        out = out.fillna(default)
    return out


def safe_date_parse(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")


def safe_merge(
    left: pd.DataFrame,
    right: pd.DataFrame,
    on: str | list[str],
    how: str = "left",
    right_columns: Sequence[str] | None = None,
    suffix: str = "_r",
) -> pd.DataFrame:
    if isinstance(on, str):
        on_cols = [on]
    else:
        on_cols = list(on)
    if right_columns is None:
        keep = [c for c in right.columns if c not in left.columns or c in on_cols]
    else:
        keep = [c for c in right_columns if c in right.columns]
        keep = list(dict.fromkeys(on_cols + keep))
    r = right[keep].copy()
    rename_map: dict[str, str] = {}
    for col in r.columns:
        if col in left.columns and col not in on_cols:
            rename_map[col] = f"{col}{suffix}"
    if rename_map:
        r = r.rename(columns=rename_map)
    return left.merge(r, on=on, how=how)


def parse_location(value) -> tuple[float, float]:
    if pd.isna(value):
        return (np.nan, np.nan)
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return (float(value[0]), float(value[1]))
    text = str(value).strip()
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, (list, tuple)) and len(parsed) >= 2:
            return (float(parsed[0]), float(parsed[1]))
    except Exception:
        pass
    nums = re.findall(r"[-+]?\d*\.?\d+", text)
    if len(nums) >= 2:
        return (float(nums[0]), float(nums[1]))
    return (np.nan, np.nan)


def safe_bool(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(False, index=df.index, dtype=bool)
    s = df[column]
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False).astype(bool)
    if pd.api.types.is_numeric_dtype(s):
        return s.fillna(0).astype(int).astype(bool)
    mapped = s.map(lambda v: False if pd.isna(v) else str(v).strip().lower() not in {"", "0", "false", "none", "nan"})
    return mapped.astype(bool)


def module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def empty_like(columns: Iterable[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=list(columns))


def first_nonempty_text(values: Iterable[object], default: str = "Unknown") -> str:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text and text.lower() != "nan":
            return text
    return default
