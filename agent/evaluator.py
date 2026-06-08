"""Gemini-powered LLM-as-judge evaluator for visualization choices."""

from __future__ import annotations

import json
from typing import Any

from utils.helpers import (
    generate_text,
    strip_json_fences,
    summarize_profile,
    summarize_viz_specs,
)


class EvaluatorError(RuntimeError):
    """Raised when the evaluator can't produce a valid score list."""


def _build_prompt(profile: dict, viz_specs: list[dict]) -> str:
    return (
        "You are an expert data visualization critic. Evaluate the following "
        "visualization choices for the given dataset profile.\n\n"
        f"Dataset Profile:\n{summarize_profile(profile)}\n\n"
        f"Visualization Choices:\n{summarize_viz_specs(viz_specs)}\n\n"
        "For each visualization, provide a score from 1 to 5:\n"
        "5 = Perfect choice for this data\n"
        "4 = Good choice, minor improvements possible\n"
        "3 = Acceptable but not optimal\n"
        "2 = Poor choice, better alternatives exist\n"
        "1 = Wrong chart type for this data\n\n"
        "Respond ONLY with a valid JSON array. No explanation, no markdown. Example:\n"
        "[\n"
        "  {\n"
        '    "visualization": "Chart Title",\n'
        '    "score": 4,\n'
        '    "feedback": "Explanation here"\n'
        "  }\n"
        "]"
    )


def _build_retry_prompt(profile: dict, viz_specs: list[dict]) -> str:
    return (
        "Return ONLY a JSON array (no prose, no markdown). For each visualization "
        "below, output an object with keys 'visualization' (the chart title), "
        "'score' (integer 1-5), and 'feedback' (one short sentence).\n\n"
        f"Dataset Profile:\n{summarize_profile(profile)}\n\n"
        f"Visualization Choices:\n{summarize_viz_specs(viz_specs)}"
    )


def _coerce_score(value: Any) -> int | None:
    try:
        score = int(round(float(value)))
    except (TypeError, ValueError):
        return None
    if score < 1 or score > 5:
        return None
    return score


def _parse_evaluations(raw_text: str, viz_specs: list[dict]) -> list[dict[str, Any]]:
    cleaned = strip_json_fences(raw_text)
    parsed = json.loads(cleaned)
    if not isinstance(parsed, list):
        raise ValueError("Evaluator response is not a JSON array")

    titles = [spec.get("title") for spec in viz_specs]
    results: list[dict[str, Any]] = []

    for idx, item in enumerate(parsed):
        if not isinstance(item, dict):
            continue
        score = _coerce_score(item.get("score"))
        if score is None:
            continue
        title = item.get("visualization") or (titles[idx] if idx < len(titles) else f"Chart {idx + 1}")
        feedback = item.get("feedback") or ""
        results.append({
            "visualization": title,
            "score": score,
            "feedback": feedback,
        })

    return results


def evaluate_visualizations(profile: dict, viz_specs: list[dict]) -> list[dict[str, Any]]:
    """Score each visualization choice 1-5 with an LLM critic."""
    if not viz_specs:
        return []

    last_error: Exception | None = None

    for attempt_prompt in (_build_prompt(profile, viz_specs), _build_retry_prompt(profile, viz_specs)):
        try:
            raw_text = generate_text(attempt_prompt)
            if not raw_text.strip():
                raise ValueError("Evaluator returned an empty response")
            evaluations = _parse_evaluations(raw_text, viz_specs)
            if not evaluations:
                raise ValueError("Evaluator returned no usable scores")
            return evaluations
        except Exception as exc:  # noqa: BLE001 - retry once
            last_error = exc
            continue

    raise EvaluatorError(f"Evaluator failed after retry: {last_error}") from last_error
