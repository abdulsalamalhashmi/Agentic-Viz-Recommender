"""Data-inspection tools the decision agent can call via Gemini function calling.

Each returned callable runs real pandas on the uploaded DataFrame and returns a
JSON-serializable result. This gives the agent a genuine tool-use capability: it
can interrogate the data (check a correlation, compare group means, look at value
counts) before deciding which visualizations to make — rather than relying only
on the static profile.
"""

from __future__ import annotations

import pandas as pd


def build_data_tools(df: pd.DataFrame) -> list:
    """Return tool callables bound to `df`.

    The function names, type hints and docstrings become the tool schema the model
    sees, so keep them descriptive.
    """

    def describe_column(column: str) -> dict:
        """Inspect one column: its dtype, null and unique counts, and — for a
        numeric column — mean, std, min, median and max."""
        if column not in df.columns:
            return {"error": f"unknown column '{column}'"}
        s = df[column]
        info: dict = {
            "dtype": str(s.dtype),
            "null_count": int(s.isna().sum()),
            "unique_count": int(s.nunique(dropna=True)),
        }
        if pd.api.types.is_numeric_dtype(s):
            d = s.dropna()
            if not d.empty:
                info.update(
                    mean=round(float(d.mean()), 3),
                    std=round(float(d.std()), 3) if len(d) > 1 else 0.0,
                    min=float(d.min()),
                    median=float(d.median()),
                    max=float(d.max()),
                )
        return info

    def correlation(column_a: str, column_b: str) -> dict:
        """Pearson correlation between two numeric columns (between -1 and 1).
        Useful to confirm a relationship before choosing a scatter plot."""
        for c in (column_a, column_b):
            if c not in df.columns:
                return {"error": f"unknown column '{c}'"}
            if not pd.api.types.is_numeric_dtype(df[c]):
                return {"error": f"column '{c}' is not numeric"}
        r = df[[column_a, column_b]].corr().iloc[0, 1]
        return {"correlation": round(float(r), 4) if pd.notna(r) else None}

    def top_values(column: str, n: int = 8) -> dict:
        """The most frequent values of a column and their counts (top n). Useful
        to gauge cardinality before a bar, pie or treemap chart."""
        if column not in df.columns:
            return {"error": f"unknown column '{column}'"}
        vc = df[column].value_counts(dropna=False).head(int(n))
        return {"top_values": {str(k): int(v) for k, v in vc.items()}}

    def group_means(group_column: str, value_column: str) -> dict:
        """Mean of a numeric `value_column` for the top groups of a categorical
        `group_column`. Useful to spot differences worth a bar or box plot."""
        if group_column not in df.columns or value_column not in df.columns:
            return {"error": "unknown column(s)"}
        if not pd.api.types.is_numeric_dtype(df[value_column]):
            return {"error": f"column '{value_column}' is not numeric"}
        g = (
            df.groupby(group_column)[value_column]
            .mean()
            .sort_values(ascending=False)
            .head(8)
        )
        return {"group_means": {str(k): round(float(v), 3) for k, v in g.items()}}

    return [describe_column, correlation, top_values, group_means]
