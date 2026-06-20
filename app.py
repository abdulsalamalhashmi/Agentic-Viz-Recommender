"""Streamlit UI for the agentic data visualization recommender."""

from __future__ import annotations

import hashlib
import io
from collections import Counter
from typing import Any

import pandas as pd
import streamlit as st

from agent.decision_agent import DecisionAgentError, decide_visualizations
from agent.evaluator import EvaluatorError, evaluate_visualizations
from agent.plot_generator import generate_plots
from agent.profiler import profile_dataset
from agent.storyteller import generate_data_story
from utils.helpers import friendly_error
from utils.report import build_report_html

st.set_page_config(
    page_title="Agentic Data Visualization Recommender",
    page_icon="📊",
    layout="wide",
)


# --------------------------------------------------------------------------- #
# Small pure helpers
# --------------------------------------------------------------------------- #
def _read_uploaded(name: str, data: bytes) -> pd.DataFrame:
    buffer = io.BytesIO(data)
    if name.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(buffer)
    # CSV: try UTF-8 first, fall back to latin-1 for non-UTF-8 files.
    try:
        return pd.read_csv(buffer)
    except UnicodeDecodeError:
        buffer.seek(0)
        return pd.read_csv(buffer, encoding="latin-1")


def _score_label(score: int) -> str:
    """A Material icon plus the colored score, e.g. ':material/check_circle: :green[**5/5**]'."""
    if score >= 4:
        color, icon = "green", "check_circle"
    elif score == 3:
        color, icon = "orange", "warning"
    else:
        color, icon = "red", "cancel"
    return f":material/{icon}: :{color}[**{score}/5**]"


def _safe_filename(title: str) -> str:
    cleaned = "".join(c if c.isalnum() or c in " -_" else "_" for c in title).strip()
    return (cleaned.replace(" ", "_") or "chart").lower()


def _type_summary(profile: dict[str, Any]) -> str:
    counts = Counter(profile.get("column_types", {}).values())
    order = ["numeric", "categorical", "datetime", "boolean", "id"]
    parts = [f"{counts[t]} {t}" for t in order if counts.get(t)]
    parts += [f"{c} {t}" for t, c in counts.items() if t not in order]
    return ", ".join(parts) if parts else "no columns"


def _align_evaluations(
    specs: list[dict[str, Any]], evaluations: list[dict[str, Any]]
) -> list[dict[str, Any] | None]:
    """Return one evaluation (or None) per spec, matched by title, each used once."""
    queues: dict[Any, list[int]] = {}
    for i, ev in enumerate(evaluations):
        queues.setdefault(ev.get("visualization"), []).append(i)

    used = [False] * len(evaluations)
    aligned: list[dict[str, Any] | None] = []
    for spec in specs:
        chosen = None
        queue = queues.get(spec.get("title"))
        while queue:
            i = queue.pop(0)
            if not used[i]:
                used[i] = True
                chosen = evaluations[i]
                break
        aligned.append(chosen)
    return aligned


def _final_cards(results: dict[str, Any]) -> list[dict[str, Any]]:
    """Plot dicts for the final set, each annotated with its evaluation (for the report)."""
    plots = results["final_plots"]
    aligned = _align_evaluations([p["spec"] for p in plots], results["final_evals"])
    return [{**plot, "evaluation": ev} for plot, ev in zip(plots, aligned)]


def _format_feedback(low_scoring: list[dict[str, Any]]) -> str:
    return "\n".join(f"- {ev['visualization']}: {ev['feedback']}" for ev in low_scoring)


# --------------------------------------------------------------------------- #
# Renderers
# --------------------------------------------------------------------------- #
def _render_profile(profile: dict[str, Any]) -> None:
    rows, cols = profile["shape"]
    st.write(f"**Shape:** {rows} rows × {cols} columns")

    type_df = pd.DataFrame(
        [
            {
                "column": col,
                "type": ctype,
                "nulls": profile["stats"][col].get("null_count", 0),
                "unique": profile["stats"][col].get("unique_count", 0),
            }
            for col, ctype in profile["column_types"].items()
        ]
    )
    st.write("**Column types**")
    st.dataframe(type_df, use_container_width=True, hide_index=True)

    correlations = profile.get("correlations", [])
    if correlations:
        st.write("**Top correlations**")
        corr_df = pd.DataFrame(
            [{"column_a": a, "column_b": b, "correlation": round(c, 3)} for a, b, c in correlations]
        )
        st.dataframe(corr_df, use_container_width=True, hide_index=True)


def _render_reasoning_log(results: dict[str, Any]) -> None:
    log = results.get("log") or []
    if not log:
        return
    with st.container(border=True):
        st.markdown("#### :material/psychology: Agent reasoning log")
        for step in log:
            st.markdown(f":material/chevron_right: {step}")


def _render_cards(
    plots: list[dict[str, Any]],
    evaluations: list[dict[str, Any]],
    key_prefix: str,
) -> None:
    """One card per chart: title + score badge, chart, why-chosen, critic feedback, download."""
    aligned = _align_evaluations([p["spec"] for p in plots], evaluations)
    for i, (plot, ev) in enumerate(zip(plots, aligned), start=1):
        spec = plot["spec"]
        title = spec.get("title", "Untitled chart")
        with st.container(border=True):
            header = f"**{i}. {title}**"
            if ev:
                header += f" — {_score_label(ev['score'])}"
            st.markdown(header)

            cols = ", ".join(spec.get("columns", []))
            color_by = spec.get("color_by")
            color_hint = f" — colored by `{color_by}`" if color_by else ""
            st.caption(f"`{spec.get('chart_type')}` on `{cols}`{color_hint}")

            if plot["error"]:
                st.error(f"Could not render this chart: {plot['error']}")
            elif plot["figure"] is not None:
                st.plotly_chart(plot["figure"], use_container_width=True)
                if plot.get("html"):
                    st.download_button(
                        "Download chart",
                        data=plot["html"],
                        file_name=f"{_safe_filename(title)}.html",
                        mime="text/html",
                        key=f"{key_prefix}_dl_{i}",
                    )

            if spec.get("reasoning"):
                st.markdown(f"**Why this chart:** {spec['reasoning']}")
            if ev and ev.get("feedback"):
                st.caption(f":material/fact_check: Critic: {ev['feedback']}")


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
def _run_agent_pipeline(
    df: pd.DataFrame, profile: dict[str, Any], dataset_name: str
) -> dict[str, Any] | None:
    log: list[str] = []
    with st.status("Running agent pipeline...", expanded=True) as status:
        rows, cols = profile["shape"]
        log.append(
            f"Profiled the dataset — {rows} rows × {cols} columns "
            f"({_type_summary(profile)}). No API call used here."
        )

        status.write("Agent is choosing visualizations...")
        try:
            viz_specs = decide_visualizations(profile)
        except DecisionAgentError as exc:
            status.update(label="Decision agent failed", state="error")
            st.error(friendly_error(exc))
            return None
        except Exception as exc:  # noqa: BLE001 - typically API/network errors
            status.update(label="Decision agent error", state="error")
            st.error(friendly_error(exc))
            return None
        log.append(f"Agent chose {len(viz_specs)} visualization(s).")

        status.write("Rendering plots...")
        plots = generate_plots(df, viz_specs)

        status.write("Critic is grading the choices...")
        try:
            evaluations = evaluate_visualizations(profile, viz_specs)
        except Exception as exc:  # noqa: BLE001 - EvaluatorError or API/network
            st.warning(friendly_error(exc))
            evaluations = []

        low_scoring = [ev for ev in evaluations if ev["score"] < 3]
        avg = sum(e["score"] for e in evaluations) / len(evaluations) if evaluations else 0.0
        if evaluations:
            log.append(
                f"Critic scored them — average {avg:.1f}/5"
                + (
                    f"; flagged {len(low_scoring)} chart(s) below 3."
                    if low_scoring
                    else "; every chart scored 3 or higher."
                )
            )
        else:
            log.append("Critic returned no scores.")

        rerun_info: dict[str, Any] | None = None
        if low_scoring:
            feedback = _format_feedback(low_scoring)
            status.write("Some charts scored low — re-running with critic feedback...")
            try:
                viz_specs_v2 = decide_visualizations(profile, feedback=feedback)
                plots_v2 = generate_plots(df, viz_specs_v2)
                evaluations_v2 = evaluate_visualizations(profile, viz_specs_v2)
                rerun_info = {
                    "viz_specs": viz_specs_v2,
                    "plots": plots_v2,
                    "evaluations": evaluations_v2,
                    "feedback": feedback,
                }
                if evaluations_v2:
                    avg2 = sum(e["score"] for e in evaluations_v2) / len(evaluations_v2)
                    trend = "improved" if avg2 > avg else ("lower" if avg2 < avg else "about the same")
                    log.append(
                        f"Re-ran once with the feedback — new average {avg2:.1f}/5 ({trend})."
                    )
            except (DecisionAgentError, EvaluatorError) as exc:
                st.warning(friendly_error(exc))
                log.append("Re-run failed; kept the original results.")
            except Exception as exc:  # noqa: BLE001
                st.warning(friendly_error(exc))
                log.append("Re-run errored; kept the original results.")
        elif evaluations:
            log.append("No re-run needed — every chart scored 3 or higher.")

        status.update(label="Pipeline complete", state="complete", expanded=False)

    if rerun_info:
        final_specs = rerun_info["viz_specs"]
        final_plots = rerun_info["plots"]
        final_evals = rerun_info["evaluations"]
    else:
        final_specs, final_plots, final_evals = viz_specs, plots, evaluations

    results: dict[str, Any] = {
        "profile": profile,
        "viz_specs": viz_specs,
        "plots": plots,
        "evaluations": evaluations,
        "rerun": rerun_info,
        "log": log,
        "final_specs": final_specs,
        "final_plots": final_plots,
        "final_evals": final_evals,
        "story": None,
    }
    try:
        results["report_html"] = build_report_html(dataset_name, profile, _final_cards(results))
    except Exception:  # noqa: BLE001 - a missing report shouldn't break the run
        results["report_html"] = None
    return results


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    st.title("Agentic Data Visualization Recommender")
    st.caption("Upload a dataset and let the AI decide how to visualize it.")

    uploaded = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])
    col_left, col_right = st.columns([1, 1])
    with col_left:
        show_profile = st.toggle("Show data profile", value=True)
    with col_right:
        run_clicked = st.button("Run Agent", type="primary", disabled=uploaded is None)

    if uploaded is None:
        st.info("Upload a CSV or Excel file above to begin.")
        return

    data = uploaded.getvalue()
    file_hash = hashlib.md5(data).hexdigest()
    file_key = (uploaded.name, file_hash)

    # Read + cache the DataFrame per file content.
    if st.session_state.get("df_key") != file_key:
        try:
            st.session_state.df = _read_uploaded(uploaded.name, data)
            st.session_state.df_key = file_key
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not read file: {exc}")
            return
    df: pd.DataFrame = st.session_state.df

    st.write(f"**File:** `{uploaded.name}` — {df.shape[0]} rows × {df.shape[1]} columns")
    with st.expander("Preview first 10 rows", expanded=False):
        st.dataframe(df.head(10), use_container_width=True)

    # --- #2 Profile is computed on upload, before any API call ---
    profile_cache: dict[str, Any] = st.session_state.setdefault("profile_cache", {})
    if file_hash not in profile_cache:
        try:
            profile_cache[file_hash] = profile_dataset(df)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not profile this dataset: {exc}")
            return
    profile = profile_cache[file_hash]

    if show_profile:
        with st.expander("Data profile", expanded=True):
            _render_profile(profile)

    # --- #7 Session result cache, keyed by file content ---
    results_cache: dict[str, Any] = st.session_state.setdefault("results_cache", {})
    story_cache: dict[str, str] = st.session_state.setdefault("story_cache", {})
    if run_clicked:
        story_cache.pop(file_hash, None)  # a fresh run invalidates the old story
        results_cache[file_hash] = _run_agent_pipeline(df, profile, uploaded.name)
    results = results_cache.get(file_hash)

    if not results:
        # A failed run already surfaced its own error; only show the start hint
        # before the first run so the two messages don't contradict each other.
        if not run_clicked:
            st.info(
                "Click **Run Agent** above to start the pipeline. The data profile "
                "is already available — no API call has been made yet."
            )
        return

    if not run_clicked:
        st.caption(":material/check_circle: Showing saved results for this file — no new API calls. Click **Run Agent** to re-run.")

    final_specs = results["final_specs"]
    final_plots = results["final_plots"]
    final_evals = results["final_evals"]

    # --- #5 Reasoning log ---
    _render_reasoning_log(results)

    # --- #3 One unified card per chart (final set) ---
    st.subheader("Recommended visualizations")
    _render_cards(final_plots, final_evals, key_prefix="final")

    # Attempt 1 kept for transparency when a re-run happened.
    if results.get("rerun"):
        with st.expander(":material/search: Attempt 1 (before the critic-driven re-run)", expanded=False):
            st.caption("Feedback the critic gave on attempt 1:")
            st.code(results["rerun"]["feedback"])
            _render_cards(results["plots"], results["evaluations"], key_prefix="v1")

    # --- #6 Data story (on-demand, one extra LLM call) ---
    story = story_cache.get(file_hash)
    with st.container(border=True):
        st.markdown("#### :material/auto_awesome: Data story")
        if story is None:
            st.caption("Generate a short, plain-language summary of what these charts reveal (uses one extra AI call).")
            if st.button("Generate data story"):
                try:
                    with st.spinner("Writing the data story..."):
                        story = generate_data_story(profile, final_specs)
                    story_cache[file_hash] = story
                    results["story"] = story
                    results["report_html"] = build_report_html(
                        uploaded.name, profile, _final_cards(results), story=story
                    )
                except Exception as exc:  # noqa: BLE001
                    st.error(friendly_error(exc))
                    story = None
        if story:
            st.write(story)

    # --- Summary + #4 downloadable report ---
    st.divider()
    st.subheader("Summary")
    if final_evals:
        avg_score = sum(ev["score"] for ev in final_evals) / len(final_evals)
        rendered = sum(1 for p in final_plots if p["error"] is None)
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Average critic score", f"{avg_score:.2f} / 5")
        col_b.metric("Visualizations", len(final_specs))
        col_c.metric("Rendered OK", f"{rendered}/{len(final_plots)}")
        if len(final_evals) < len(final_specs):
            st.caption(
                f":material/warning: The critic scored {len(final_evals)} of {len(final_specs)} "
                "charts; the rest were left unscored."
            )
    else:
        st.info("No critic evaluations available for the summary.")

    if results.get("report_html"):
        st.download_button(
            "Download full report (HTML)",
            data=results["report_html"],
            file_name=f"{_safe_filename(uploaded.name)}_report.html",
            mime="text/html",
            key="report_dl",
        )


if __name__ == "__main__":
    main()
