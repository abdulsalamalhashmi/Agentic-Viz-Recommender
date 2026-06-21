"""Turn visualization specs into Plotly figures."""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.express as px

# A discrete colour legend with hundreds of entries freezes the browser, so
# only colour by a categorical column when its cardinality is reasonable.
# Continuous (numeric) colour uses a colourbar and is always fine.
MAX_COLOR_CATEGORIES = 20

# Cap the number of points sent to point-per-row charts (scatter) so a
# ~19k-row dataset doesn't render tens of thousands of marks client-side.
MAX_SCATTER_POINTS = 5000


def _safe_color(df: pd.DataFrame, spec: dict[str, Any]) -> str | None:
    """Return a usable colour column, or None if it would explode the legend."""
    color = spec.get("color_by")
    if not color or color not in df.columns:
        return None
    if pd.api.types.is_numeric_dtype(df[color]):
        return color  # continuous colourbar, no legend blow-up
    if df[color].nunique(dropna=True) > MAX_COLOR_CATEGORIES:
        return None
    return color


def _maybe_sample(df: pd.DataFrame, max_points: int = MAX_SCATTER_POINTS) -> pd.DataFrame:
    """Down-sample large frames for point-per-row charts (deterministic)."""
    if len(df) > max_points:
        return df.sample(max_points, random_state=0)
    return df


def _count_col(*data_cols: str) -> str:
    """A name for the counts column that won't collide with the data columns, so a
    dataset with a column literally named 'count' still renders."""
    name = "count"
    while name in data_cols:
        name += "_"
    return name


def _scatter(df: pd.DataFrame, spec: dict[str, Any]):
    cols = spec["columns"]
    if len(cols) < 2:
        raise ValueError("scatter requires two columns")
    x, y = cols[0], cols[1]
    color = _safe_color(df, spec)
    plot_df = _maybe_sample(df)
    return px.scatter(plot_df, x=x, y=y, color=color, title=spec.get("title"))


def _bar(df: pd.DataFrame, spec: dict[str, Any], top_n: int = 20):
    col = spec["columns"][0]
    cnt = _count_col(col)
    vc = df[col].value_counts(dropna=False).head(top_n)
    counts = pd.DataFrame({col: vc.index, cnt: vc.to_numpy()})
    title = spec.get("title")
    if df[col].nunique(dropna=False) > top_n:
        title = f"{title} (top {top_n})" if title else f"Top {top_n}"
    return px.bar(counts, x=col, y=cnt, title=title)


def _histogram(df: pd.DataFrame, spec: dict[str, Any]):
    col = spec["columns"][0]
    color = _safe_color(df, spec)
    return px.histogram(df, x=col, color=color, title=spec.get("title"))


def _heatmap(df: pd.DataFrame, spec: dict[str, Any], max_cols: int = 15):
    requested = spec.get("columns") or []
    numeric_cols = [c for c in requested if pd.api.types.is_numeric_dtype(df[c])]
    if len(numeric_cols) < 2:
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if len(numeric_cols) < 2:
        raise ValueError("heatmap requires at least two numeric columns")
    numeric_cols = numeric_cols[:max_cols]  # keep the matrix readable on wide datasets
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
    if not pd.api.types.is_datetime64_any_dtype(plot_df[x]) and not pd.api.types.is_numeric_dtype(plot_df[x]):
        # Keep a numeric x (e.g. a "year" column) numeric — coercing integers to
        # datetime would read them as nanoseconds and collapse everything to 1970.
        plot_df[x] = pd.to_datetime(plot_df[x], errors="coerce")
    plot_df = plot_df.dropna(subset=[x]).sort_values(x)
    color = _safe_color(df, spec)
    if color:
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
    return px.box(df, x=x, y=y, color=_safe_color(df, spec), title=spec.get("title"))


def _pie(df: pd.DataFrame, spec: dict[str, Any]):
    col = spec["columns"][0]
    n_unique = int(df[col].nunique(dropna=True))
    if n_unique > 6:
        raise ValueError(f"pie skipped: column '{col}' has {n_unique} unique values (>6)")
    cnt = _count_col(col)
    vc = df[col].value_counts(dropna=False)
    counts = pd.DataFrame({col: vc.index, cnt: vc.to_numpy()})
    return px.pie(counts, names=col, values=cnt, title=spec.get("title"))


def _grouped_bar(df: pd.DataFrame, spec: dict[str, Any], top_n: int = 15):
    """Counts of one categorical column broken down by a second (grouped bars)."""
    cols = spec["columns"]
    if len(cols) < 2:
        raise ValueError("grouped_bar requires two categorical columns")
    x, group = cols[0], cols[1]
    if df[group].nunique(dropna=True) > MAX_COLOR_CATEGORIES:
        raise ValueError(
            f"grouped_bar skipped: '{group}' has more than {MAX_COLOR_CATEGORIES} categories"
        )
    work = df.dropna(subset=[x, group])  # only count rows where both categories are present
    if work.empty:
        raise ValueError("grouped_bar has no rows where both categories are present")
    cnt = _count_col(x, group)
    counts = (
        work.assign(**{group: work[group].astype(str)})
        .groupby([x, group], dropna=False)
        .size()
        .reset_index(name=cnt)
    )
    top = counts.groupby(x)[cnt].sum().nlargest(top_n).index
    counts = counts[counts[x].isin(top)]
    title = spec.get("title")
    if df[x].nunique(dropna=False) > top_n:
        title = f"{title} (top {top_n})" if title else f"Top {top_n}"
    return px.bar(counts, x=x, y=cnt, color=group, barmode="group", title=title)


def _area(df: pd.DataFrame, spec: dict[str, Any]):
    """Filled trend of a numeric column over a datetime column."""
    cols = spec["columns"]
    if len(cols) < 2:
        raise ValueError("area requires a datetime column and a numeric column")
    x, y = cols[0], cols[1]
    plot_df = df[[x, y]].copy()
    if not pd.api.types.is_datetime64_any_dtype(plot_df[x]) and not pd.api.types.is_numeric_dtype(plot_df[x]):
        # Keep a numeric x (e.g. a "year" column) numeric — coercing integers to
        # datetime would read them as nanoseconds and collapse everything to 1970.
        plot_df[x] = pd.to_datetime(plot_df[x], errors="coerce")
    plot_df = plot_df.dropna(subset=[x]).sort_values(x)
    color = _safe_color(df, spec)
    if color:
        plot_df[color] = df.loc[plot_df.index, color]
    return px.area(plot_df, x=x, y=y, color=color, title=spec.get("title"))


def _treemap(df: pd.DataFrame, spec: dict[str, Any], top_n: int = 30):
    """Part-to-whole composition of one (or two nested) categorical columns."""
    path = [c for c in (spec.get("columns") or [])][:2]
    if not path:
        raise ValueError("treemap requires at least one categorical column")
    work = df.dropna(subset=path)  # a treemap row needs every level present, so drop nulls
    if work.empty:
        raise ValueError("treemap has no rows where the chosen categories are all present")
    cnt = _count_col(*path)
    counts = work.groupby(path, dropna=False).size().reset_index(name=cnt)
    counts = counts.nlargest(top_n, cnt)
    for col in path:
        counts[col] = counts[col].astype(str)
    return px.treemap(counts, path=path, values=cnt, title=spec.get("title"))


_BUILDERS = {
    "scatter": _scatter,
    "bar": _bar,
    "grouped_bar": _grouped_bar,
    "histogram": _histogram,
    "heatmap": _heatmap,
    "line": _line,
    "area": _area,
    "box": _box,
    "pie": _pie,
    "treemap": _treemap,
}


def generate_plots(df: pd.DataFrame, viz_specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return one result dict per spec: {spec, figure, error, html}.

    The download HTML is rendered once here (not on every Streamlit rerun) so
    repeated UI interactions stay fast even with several large charts.
    """
    results: list[dict[str, Any]] = []
    for spec in viz_specs:
        chart_type = spec.get("chart_type")
        builder = _BUILDERS.get(chart_type)
        result: dict[str, Any] = {"spec": spec, "figure": None, "error": None, "html": None}

        if builder is None:
            result["error"] = f"Unsupported chart_type: {chart_type!r}"
            results.append(result)
            continue

        try:
            figure = builder(df, spec)
            result["figure"] = figure
            result["html"] = figure.to_html(include_plotlyjs="cdn")
        except Exception as exc:  # noqa: BLE001 - one bad chart shouldn't kill the loop
            result["error"] = str(exc)

        results.append(result)
    return results
