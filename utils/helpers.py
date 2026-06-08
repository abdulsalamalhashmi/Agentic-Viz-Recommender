"""Shared helpers: Gemini configuration, JSON cleanup, and prompt summaries."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from google import genai

DEFAULT_MODEL = "gemini-2.5-flash"

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

_client: genai.Client | None = None


def get_gemini_client() -> genai.Client:
    """Return a process-wide Gemini client, loading the API key from env / .env on first use."""
    global _client
    if _client is not None:
        return _client

    load_dotenv(_PROJECT_ROOT / ".env", override=True)
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Add it to your environment or a .env file. "
            "On Hugging Face Spaces, add it as a Space secret."
        )

    _client = genai.Client(api_key=api_key)
    return _client


def generate_text(prompt: str, model_name: str = DEFAULT_MODEL) -> str:
    """Send a single-turn prompt to Gemini and return the response text."""
    global _client
    client = get_gemini_client()
    try:
        response = client.models.generate_content(model=model_name, contents=prompt)
    except Exception:
        # Drop the cached client so a fixed key / transient failure can recover without restart.
        _client = None
        raise
    return getattr(response, "text", "") or ""


def strip_json_fences(text: str) -> str:
    """Remove ```json / ``` markdown fences that Gemini sometimes returns."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.removeprefix("```json").removeprefix("```").lstrip()
    if cleaned.endswith("```"):
        cleaned = cleaned.removesuffix("```").rstrip()
    return cleaned.strip()


def _round(value: Any, ndigits: int = 4) -> Any:
    if isinstance(value, float):
        return round(value, ndigits)
    return value


def summarize_profile(profile: dict) -> str:
    """Render the profile as a compact JSON-ish block suitable for an LLM prompt."""
    rows, cols = profile.get("shape", (0, 0))

    stats_compact = {
        col: {k: _round(v) for k, v in stats.items()}
        for col, stats in profile.get("stats", {}).items()
    }

    correlations = [
        {"columns": [a, b], "correlation": _round(c)}
        for a, b, c in profile.get("correlations", [])
    ]

    summary = {
        "shape": {"rows": rows, "columns": cols},
        "columns": profile.get("columns", []),
        "column_types": profile.get("column_types", {}),
        "stats": stats_compact,
        "top_correlations": correlations,
        "sample_rows": profile.get("sample", []),
    }
    return json.dumps(summary, indent=2, default=str)


def summarize_viz_specs(viz_specs: list[dict]) -> str:
    """Compact JSON of viz specs for the evaluator prompt."""
    trimmed = [
        {
            "title": spec.get("title"),
            "chart_type": spec.get("chart_type"),
            "columns": spec.get("columns"),
            "color_by": spec.get("color_by"),
            "reasoning": spec.get("reasoning"),
        }
        for spec in viz_specs
    ]
    return json.dumps(trimmed, indent=2, default=str)
