"""Shared helpers: Gemini configuration, JSON cleanup, and prompt summaries."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from google import genai

DEFAULT_MODEL = "gemini-2.5-flash"

# If the default model is overloaded, fall back to these in order.
_FALLBACK_MODELS = ("gemini-2.0-flash", "gemini-flash-latest")

# Transient server-side errors worth retrying with backoff.
_RETRYABLE_MARKERS = ("503", "unavailable", "429", "resource_exhausted", "overloaded", "high demand")
_MAX_ATTEMPTS = 4

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


def _is_retryable(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(marker in text for marker in _RETRYABLE_MARKERS)


def _generate_once(model_name: str, prompt: str) -> str:
    """Call one model with exponential-backoff retries on transient errors."""
    global _client
    client = get_gemini_client()

    for attempt in range(_MAX_ATTEMPTS):
        try:
            response = client.models.generate_content(model=model_name, contents=prompt)
            return getattr(response, "text", "") or ""
        except Exception as exc:  # noqa: BLE001
            # Drop the cached client so a fixed key / transient failure can recover without restart.
            _client = None
            client = get_gemini_client()
            if _is_retryable(exc) and attempt < _MAX_ATTEMPTS - 1:
                time.sleep(2 ** attempt)  # 1s, 2s, 4s
                continue
            raise
    return ""


def generate_text(prompt: str, model_name: str = DEFAULT_MODEL) -> str:
    """Send a single-turn prompt to Gemini and return the response text.

    Retries transient server errors (503 overloaded, 429 rate limit) with
    exponential backoff. If the primary model stays overloaded, falls back to
    alternative models so a busy server doesn't break the run.
    """
    models_to_try = [model_name, *(m for m in _FALLBACK_MODELS if m != model_name)]

    last_exc: Exception | None = None
    for model in models_to_try:
        try:
            return _generate_once(model, prompt)
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            # Only move on to a fallback model for transient/overload errors.
            if _is_retryable(exc):
                continue
            raise

    if last_exc:
        raise last_exc
    return ""


def strip_json_fences(text: str) -> str:
    """Remove ```json / ``` markdown fences that Gemini sometimes returns.

    Handles any opening fence tag case-insensitively (```json, ```JSON,
    ```python, or a bare ```), not just lowercase ```json, by dropping the
    whole opening fence line.
    """
    cleaned = text.strip()
    if cleaned.startswith("```"):
        newline = cleaned.find("\n")
        cleaned = cleaned[newline + 1:] if newline != -1 else cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
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
