"""Gemini-powered visualization decision agent."""

from __future__ import annotations

import json
from typing import Any

from utils.helpers import generate_text, strip_json_fences, summarize_profile

VALID_CHART_TYPES = {"scatter", "bar", "histogram", "heatmap", "line", "box", "pie"}

RULES_BLOCK = """\
Visualization Rules:
- Use scatter plot for two numeric columns with correlation between 0.3 and 0.9
- Use bar chart for one categorical column (show value counts)
- Use histogram for a single numeric column (show distribution)
- Use heatmap for correlation matrix when there are 4+ numeric columns
- Use line chart only if a datetime column exists (x=datetime, y=numeric)
- Use box plot to compare a numeric column across categories
- Avoid pie charts when unique categories > 6
- Skip columns classified as "id"
- Generate between 3 and 6 visualizations total"""

EXAMPLE_BLOCK = """\
Respond ONLY with a valid JSON array. No explanation, no markdown. Example format:
[
  {
    "columns": ["col1", "col2"],
    "chart_type": "scatter",
    "color_by": "optional_col_or_null",
    "title": "Chart Title",
    "reasoning": "Why this chart was chosen"
  }
]"""


class DecisionAgentError(RuntimeError):
    """Raised when the decision agent can't produce a valid spec list."""


def _build_prompt(profile: dict, feedback: str | None) -> str:
    profile_summary = summarize_profile(profile)
    feedback_section = ""
    if feedback:
        feedback_section = (
            "\nThe previous attempt received the following critic feedback. "
            "Address each point when choosing new visualizations:\n"
            f"{feedback}\n"
        )

    return (
        "You are a data visualization expert. Given the following dataset profile, "
        "decide which visualizations would best represent the data.\n\n"
        f"Dataset Profile:\n{profile_summary}\n\n"
        f"{RULES_BLOCK}\n"
        f"{feedback_section}\n"
        f"{EXAMPLE_BLOCK}"
    )


def _build_retry_prompt(profile: dict, feedback: str | None) -> str:
    profile_summary = summarize_profile(profile)
    feedback_line = f"\nCritic feedback to address:\n{feedback}\n" if feedback else ""
    return (
        "Return ONLY a JSON array (no prose, no markdown fences) of 3-6 visualization "
        "specs for the dataset described below. Each item must have keys: columns "
        "(list of strings), chart_type (one of scatter, bar, histogram, heatmap, line, box, pie), "
        "color_by (string or null), title (string), reasoning (string). "
        "Skip any column whose type is 'id'.\n\n"
        f"Dataset Profile:\n{profile_summary}\n"
        f"{feedback_line}"
    )


def _parse_specs(raw_text: str) -> list[dict[str, Any]]:
    cleaned = strip_json_fences(raw_text)
    parsed = json.loads(cleaned)
    if not isinstance(parsed, list):
        raise ValueError("Decision agent response is not a JSON array")
    return parsed


def _validate_specs(specs: list[dict[str, Any]], profile: dict) -> list[dict[str, Any]]:
    columns = set(profile.get("columns", []))
    column_types = profile.get("column_types", {})
    id_columns = {c for c, t in column_types.items() if t == "id"}

    valid: list[dict[str, Any]] = []
    for spec in specs:
        if not isinstance(spec, dict):
            continue

        chart_type = spec.get("chart_type")
        if chart_type not in VALID_CHART_TYPES:
            continue

        spec_cols = spec.get("columns") or []
        if not isinstance(spec_cols, list) or not spec_cols:
            continue
        if any(col not in columns for col in spec_cols):
            continue
        if any(col in id_columns for col in spec_cols):
            continue

        color_by = spec.get("color_by")
        if color_by in ("", "null"):
            color_by = None
        if color_by is not None and color_by not in columns:
            color_by = None

        valid.append({
            "columns": spec_cols,
            "chart_type": chart_type,
            "color_by": color_by,
            "title": spec.get("title") or f"{chart_type.title()} of {', '.join(spec_cols)}",
            "reasoning": spec.get("reasoning") or "",
        })

    return valid


def decide_visualizations(profile: dict, feedback: str | None = None) -> list[dict[str, Any]]:
    """Ask Gemini for a list of visualization specs. Retries once on parse failure."""
    prompt = _build_prompt(profile, feedback)
    last_error: Exception | None = None

    for attempt_prompt in (prompt, _build_retry_prompt(profile, feedback)):
        try:
            raw_text = generate_text(attempt_prompt)
            if not raw_text.strip():
                raise ValueError("Gemini returned an empty response")

            specs = _parse_specs(raw_text)
            valid = _validate_specs(specs, profile)
            if not valid:
                raise ValueError("No valid visualization specs after validation")
            return valid
        except Exception as exc:  # noqa: BLE001 - we want a single retry path
            last_error = exc
            continue

    raise DecisionAgentError(f"Decision agent failed after retry: {last_error}") from last_error
