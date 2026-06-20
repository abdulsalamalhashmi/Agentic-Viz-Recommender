"""Build a single self-contained HTML report of an entire agent run:
every chart + why it was chosen + critic score/feedback + an optional data story.
"""

from __future__ import annotations

import html as _html
from typing import Any

import plotly.io as pio


def _esc(value: Any) -> str:
    return _html.escape(str(value))


def _score_color(score: int) -> str:
    if score >= 4:
        return "#1a7f37"  # green
    if score == 3:
        return "#9a6700"  # amber
    return "#b00020"  # red


_STYLE = """
body{font-family:system-ui,-apple-system,Segoe UI,Arial,sans-serif;max-width:1000px;
margin:24px auto;padding:0 16px;color:#1a1a1a;line-height:1.5}
h1{margin-bottom:2px} h2{margin:6px 0}
.muted{color:#666}
.card{border:1px solid #e3e3e3;border-radius:10px;padding:16px 18px;margin:18px 0}
.badge{display:inline-block;padding:2px 10px;border-radius:12px;color:#fff;font-weight:600;font-size:13px}
.reason{margin:10px 0 4px} .feedback{color:#444;font-style:italic;margin:4px 0}
table{border-collapse:collapse;margin-top:8px} td,th{border:1px solid #ddd;padding:4px 10px;font-size:14px;text-align:left}
.err{color:#b00020}
"""


def build_report_html(
    dataset_name: str,
    profile: dict,
    cards: list[dict],
    story: str | None = None,
) -> str:
    """Render the run as one HTML string.

    `cards` is a list of {spec, figure, error, evaluation} dicts. Plotly.js is
    loaded once from the CDN (on the first figure) and reused by the rest.
    """
    rows, cols = profile.get("shape", (0, 0))
    out: list[str] = [
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        f"<title>Report — {_esc(dataset_name)}</title>",
        f"<style>{_STYLE}</style></head><body>",
        "<h1>Agentic Data Visualization Report</h1>",
        f"<p class='muted'>Dataset: <b>{_esc(dataset_name)}</b> — "
        f"{rows} rows × {cols} columns</p>",
    ]

    if story:
        out.append(f"<div class='card'><h2>✨ Data story</h2><p>{_esc(story)}</p></div>")

    out.append("<div class='card'><h2>Data profile</h2>")
    out.append("<table><tr><th>Column</th><th>Type</th></tr>")
    for col, ctype in profile.get("column_types", {}).items():
        out.append(f"<tr><td>{_esc(col)}</td><td>{_esc(ctype)}</td></tr>")
    out.append("</table></div>")

    plotly_loaded = False
    for i, card in enumerate(cards, start=1):
        spec = card.get("spec", {})
        ev = card.get("evaluation")
        out.append("<div class='card'>")
        out.append(f"<h2>{i}. {_esc(spec.get('title', 'Untitled chart'))}</h2>")
        if ev:
            out.append(
                f"<p><span class='badge' style='background:{_score_color(ev['score'])}'>"
                f"Critic score: {ev['score']}/5</span></p>"
            )
        if card.get("error"):
            out.append(f"<p class='err'>Could not render this chart: {_esc(card['error'])}</p>")
        elif card.get("figure") is not None:
            include = "cdn" if not plotly_loaded else False
            out.append(pio.to_html(card["figure"], include_plotlyjs=include, full_html=False))
            plotly_loaded = True
        if spec.get("reasoning"):
            out.append(f"<p class='reason'><b>Why this chart:</b> {_esc(spec['reasoning'])}</p>")
        if ev and ev.get("feedback"):
            out.append(f"<p class='feedback'><b>Critic:</b> {_esc(ev['feedback'])}</p>")
        out.append("</div>")

    out.append("</body></html>")
    return "".join(out)
