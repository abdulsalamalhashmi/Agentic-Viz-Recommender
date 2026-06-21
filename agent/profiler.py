"""Dataset profiler: turn a DataFrame into a structured dict the agent can reason over."""

from __future__ import annotations

import json
import math
import re
from typing import Any

import numpy as np
import pandas as pd

DATETIME_NAME_PATTERN = re.compile(r"\b(date|time|year|timestamp|datetime)\b", re.IGNORECASE)
ID_NAME_PATTERN = re.compile(r"\b(id|index|key|uuid|guid)\b", re.IGNORECASE)


def _classify_column(name: str, series: pd.Series, n_rows: int) -> str:
    """Classify a column as one of: id, datetime, boolean, numeric, categorical.

    The spec's id rule ("unique values == number of rows → id") is too aggressive
    for small datasets — a 20-row datetime or numeric column would always trip it.
    So dtype-based detection (datetime, boolean) wins over the unique-count heuristic;
    the id rule only fires on name match, or on string/integer columns that are fully
    unique (the typical "primary key" shape).
    """
    n_unique = int(series.nunique(dropna=True))
    non_null = int(series.notna().sum())

    # 1. dtype-driven detections that should never be overridden by a heuristic
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    if pd.api.types.is_bool_dtype(series):
        return "boolean"

    # 2. id by name (always wins over numeric/categorical interpretation)
    if ID_NAME_PATTERN.search(name):
        return "id"

    # 3. id by shape: object or integer columns that are fully unique look like keys.
    #    Floats are excluded — a 20-row continuous measurement shouldn't be an id.
    fully_unique = n_rows > 0 and n_unique == non_null == n_rows
    if fully_unique and (pd.api.types.is_object_dtype(series) or pd.api.types.is_integer_dtype(series)):
        return "id"

    # 4. datetime by name hint (fallback for string-typed date columns)
    if DATETIME_NAME_PATTERN.search(name):
        return "datetime"

    # 5. boolean: two distinct values. For numeric dtypes, only call it boolean
    #    when the values are a 0/1 flag — otherwise a 2-valued measurement should
    #    keep flowing through the numeric path below (stats, correlations).
    if n_unique == 2:
        if not pd.api.types.is_numeric_dtype(series):
            return "boolean"
        if set(series.dropna().unique()) <= {0, 1}:
            return "boolean"

    # 6. numeric: float/int with reasonable cardinality.
    #    The 10%-of-rows rule alone misclassifies real numeric columns (age,
    #    ratings, prices) as categorical on large datasets — on 18k rows it
    #    demands >1800 distinct values — so also accept any numeric column with a
    #    reasonable absolute number of distinct values.
    if pd.api.types.is_numeric_dtype(series):
        if n_rows == 0 or n_unique > 0.10 * n_rows or n_unique >= 20:
            return "numeric"
        return "categorical"

    # 7. fallback
    return "categorical"


def _column_stats(series: pd.Series, col_type: str) -> dict[str, Any]:
    stats: dict[str, Any] = {
        "null_count": int(series.isna().sum()),
        "unique_count": int(series.nunique(dropna=True)),
    }

    if col_type == "numeric" and pd.api.types.is_numeric_dtype(series):
        numeric = series.dropna()
        if not numeric.empty:
            stats["mean"] = float(numeric.mean())
            std = float(numeric.std()) if len(numeric) > 1 else 0.0
            stats["std"] = std if not math.isnan(std) else 0.0
            stats["min"] = float(numeric.min())
            stats["max"] = float(numeric.max())

    return stats


def _top_correlations(df: pd.DataFrame, numeric_cols: list[str], top_n: int = 5) -> list[tuple[str, str, float]]:
    if len(numeric_cols) < 2:
        return []

    corr = df[numeric_cols].corr(numeric_only=True)
    pairs: list[tuple[str, str, float]] = []
    for i, col_a in enumerate(numeric_cols):
        for col_b in numeric_cols[i + 1:]:
            value = corr.loc[col_a, col_b]
            if pd.isna(value):
                continue
            pairs.append((col_a, col_b, float(value)))

    pairs.sort(key=lambda triple: abs(triple[2]), reverse=True)
    return pairs[:top_n]


_MAX_SAMPLE_STR = 200  # truncate long cell strings before they reach the prompt


def _json_safe_sample(df: pd.DataFrame, n: int = 3) -> list[dict[str, Any]]:
    """Sample rows as JSON-safe dicts: NaN/NaT -> null, numpy scalars -> Python.

    Long string cells are truncated so a pathological value can't bloat — or
    smuggle instructions into — the LLM prompt.
    """
    sample = df.head(n).copy()
    for col in sample.columns:
        if pd.api.types.is_datetime64_any_dtype(sample[col]):
            sample[col] = sample[col].astype(str)
    # to_json maps NaN/inf -> null and numpy scalars -> JSON numbers; round-trip
    # back to plain Python types so a later json.dumps emits clean values
    # (no bare NaN tokens, no integers stringified by a default= fallback).
    records = json.loads(sample.to_json(orient="records"))
    for record in records:
        for key, value in record.items():
            if isinstance(value, str) and len(value) > _MAX_SAMPLE_STR:
                record[key] = value[:_MAX_SAMPLE_STR] + "…"
    return records


def profile_dataset(df: pd.DataFrame) -> dict[str, Any]:
    """Return a structured profile of `df` consumable by the decision/evaluator agents."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("profile_dataset expects a pandas DataFrame")
    if df.columns.duplicated().any():
        dupes = sorted({str(c) for c in df.columns[df.columns.duplicated()]})
        raise ValueError(
            "Duplicate column names are not supported; please make them unique: "
            + ", ".join(dupes)
        )

    n_rows, n_cols = df.shape

    column_types: dict[str, str] = {}
    stats: dict[str, dict[str, Any]] = {}

    for col in df.columns:
        col_type = _classify_column(str(col), df[col], n_rows)
        column_types[col] = col_type
        stats[col] = _column_stats(df[col], col_type)

    numeric_cols = [c for c, t in column_types.items() if t == "numeric"]
    correlations = _top_correlations(df, numeric_cols)

    return {
        "columns": list(df.columns),
        "shape": (int(n_rows), int(n_cols)),
        "column_types": column_types,
        "stats": stats,
        "correlations": correlations,
        "sample": _json_safe_sample(df),
    }
