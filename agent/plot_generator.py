"""Turn visualization specs into Plotly figures."""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.express as px


def _scatter(df: pd.DataFrame, spec: dict[str, Any]):
    cols = spec["columns"]
    if len(cols) < 2:
        raise ValueError("scatter requires two columns")
    x, y = cols[0], cols[1]
    color = spec.get("color_by")
    return px.scatter(df, x=x, y=y, color=color, title=spec.get("title"))


def _bar(df: pd.DataFrame, spec: dict[str, Any], top_n: int = 20):
    col = spec["columns"][0]
    counts = df[col].value_counts(dropna=False).head(top_n).reset_index()
    counts.columns = [col, "count"]
    title = spec.get("title")
    if df[col].nunique(dropna=False) > top_n:
        title = f"{title} (top {top_n})" if title else f"Top {top_n}"
    return px.bar(counts, x=col, y="count", title=title)


def _histogram(df: pd.DataFrame, spec: dict[str, Any]):
    col = spec["columns"][0]
    color = spec.get("color_by")
    return px.histogram(df, x=col, color=color, title=spec.get("title"))


def _heatmap(df: pd.DataFrame, spec: dict[str, Any]):
    requested = spec.get("columns") or []
    numeric_cols = [c for c in requested if pd.api.types.is_numeric_dtype(df[c])]
    if len(numeric_cols) < 2:
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if len(numeric_cols) < 2:
        raise ValueError("heatmap requires at least two numeric columns")
    corr = df[numeric_cols].corr(numeric_only=True)
    return px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title=spec.get("title"),
    )


def _line(df: pd.DataFrame, spec: dict[str, Any]):
    cols = spec["columns"]
    if len(cols) < 2:
        raise ValueError("line requires a datetime column and a numeric column")
    x, y = cols[0], cols[1]
    plot_df = df[[x, y]].copy()
    if not pd.api.types.is_datetime64_any_dtype(plot_df[x]):
        plot_df[x] = pd.to_datetime(plot_df[x], errors="coerce")
    plot_df = plot_df.dropna(subset=[x]).sort_values(x)
    color = spec.get("color_by")
    if color and color in df.columns:
        plot_df[color] = df.loc[plot_df.index, color]
    return px.line(plot_df, x=x, y=y, color=color, title=spec.get("title"))


def _box(df: pd.DataFrame, spec: dict[str, Any]):
    cols = spec["columns"]
    if len(cols) < 2:
        raise ValueError("box requires a categorical column and a numeric column")
    a, b = cols[0], cols[1]
    if pd.api.types.is_numeric_dtype(df[a]) and not pd.api.types.is_numeric_dtype(df[b]):
        x, y = b, a
    else:
        x, y = a, b
    return px.box(df, x=x, y=y, color=spec.get("color_by"), title=spec.get("title"))


def _pie(df: pd.DataFrame, spec: dict[str, Any]):
    col = spec["columns"][0]
    n_unique = int(df[col].nunique(dropna=True))
    if n_unique > 6:
        raise ValueError(f"pie skipped: column '{col}' has {n_unique} unique values (>6)")
    counts = df[col].value_counts(dropna=False).reset_index()
    counts.columns = [col, "count"]
    return px.pie(counts, names=col, values="count", title=spec.get("title"))


_BUILDERS = {
    "scatter": _scatter,
    "bar": _bar,
    "histogram": _histogram,
    "heatmap": _heatmap,
    "line": _line,
    "box": _box,
    "pie": _pie,
}


def generate_plots(df: pd.DataFrame, viz_specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return one result dict per spec: {spec, figure, error}."""
    results: list[dict[str, Any]] = []
    for spec in viz_specs:
        chart_type = spec.get("chart_type")
        builder = _BUILDERS.get(chart_type)
        result = {"spec": spec, "figure": None, "error": None}

        if builder is None:
            result["error"] = f"Unsupported chart_type: {chart_type!r}"
            results.append(result)
            continue

        try:
            result["figure"] = builder(df, spec)
        except Exception as exc:  # noqa: BLE001 - one bad chart shouldn't kill the loop
            result["error"] = str(exc)

        results.append(result)
    return results
