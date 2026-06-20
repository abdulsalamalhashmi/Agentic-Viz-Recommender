"""Streamlit UI for the agentic data visualization recommender."""

from __future__ import annotations

import hashlib
import io
from typing import Any

import pandas as pd
import streamlit as st

from agent.decision_agent import DecisionAgentError, decide_visualizations
from agent.evaluator import EvaluatorError, evaluate_visualizations
from agent.plot_generator import generate_plots
from agent.profiler import profile_dataset

st.set_page_config(
    page_title="Agentic Data Visualization Recommender",
    page_icon="📊",
    layout="wide",
)


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


def _score_badge(score: int) -> str:
    if score >= 4:
        return "🟢"
    if score == 3:
        return "🟡"
    return "🔴"


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


def _render_specs(viz_specs: list[dict[str, Any]], heading: str) -> None:
    st.subheader(heading)
    for i, spec in enumerate(viz_specs, start=1):
        with st.container(border=True):
            st.markdown(f"**{i}. {spec.get('title', 'Untitled chart')}**")
            cols = ", ".join(spec.get("columns", []))
            color_by = spec.get("color_by")
            color_hint = f" — colored by `{color_by}`" if color_by else ""
            st.caption(f"`{spec.get('chart_type')}` on `{cols}`{color_hint}")
            reasoning = spec.get("reasoning")
            if reasoning:
                st.write(reasoning)


def _safe_filename(title: str) -> str:
    cleaned = "".join(c if c.isalnum() or c in " -_" else "_" for c in title).strip()
    return (cleaned.replace(" ", "_") or "chart").lower()


def _render_plots(plots: list[dict[str, Any]], key_prefix: str = "plots") -> None:
    st.subheader("3. Generated plots")
    for i, plot in enumerate(plots, start=1):
        spec = plot["spec"]
        title = spec.get("title", "Untitled chart")
        with st.container(border=True):
            st.markdown(f"**{i}. {title}**")
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
            reasoning = spec.get("reasoning")
            if reasoning:
                st.caption(reasoning)


def _render_evaluations(evaluations: list[dict[str, Any]], heading: str = "4. Critic evaluation") -> None:
    st.subheader(heading)
    for ev in evaluations:
        badge = _score_badge(ev["score"])
        with st.container(border=True):
            st.markdown(f"**{ev['visualization']}** — {badge} **Score: {ev['score']}/5**")
            if ev.get("feedback"):
                st.write(ev["feedback"])


def _format_feedback(low_scoring: list[dict[str, Any]]) -> str:
    return "\n".join(f"- {ev['visualization']}: {ev['feedback']}" for ev in low_scoring)


def _run_agent_pipeline(df: pd.DataFrame) -> dict[str, Any] | None:
    with st.status("Running agent pipeline...", expanded=True) as status:
        status.write("Profiling the dataset...")
        try:
            profile = profile_dataset(df)
        except Exception as exc:  # noqa: BLE001
            status.update(label="Profiling failed", state="error")
            st.error(f"Profiling failed: {exc}")
            return None

        status.write("Agent is choosing visualizations...")
        try:
            viz_specs = decide_visualizations(profile)
        except DecisionAgentError as exc:
            status.update(label="Decision agent failed", state="error")
            st.error(f"Decision agent failed: {exc}")
            return None
        except Exception as exc:  # noqa: BLE001 - typically API/network errors
            status.update(label="Decision agent error", state="error")
            st.error(f"Decision agent error: {exc}")
            return None

        status.write("Rendering plots...")
        plots = generate_plots(df, viz_specs)

        status.write("Critic is grading the choices...")
        try:
            evaluations = evaluate_visualizations(profile, viz_specs)
        except EvaluatorError as exc:
            st.warning(f"Evaluator failed: {exc}")
            evaluations = []
        except Exception as exc:  # noqa: BLE001
            st.warning(f"Evaluator error: {exc}")
            evaluations = []

        rerun_info: dict[str, Any] | None = None
        low_scoring = [ev for ev in evaluations if ev["score"] < 3]
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
            except (DecisionAgentError, EvaluatorError) as exc:
                st.warning(f"Re-run failed, keeping original results: {exc}")
            except Exception as exc:  # noqa: BLE001
                st.warning(f"Re-run error, keeping original results: {exc}")

        status.update(label="Pipeline complete", state="complete", expanded=False)

    return {
        "profile": profile,
        "viz_specs": viz_specs,
        "plots": plots,
        "evaluations": evaluations,
        "rerun": rerun_info,
    }


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
    # Key on a content hash (not just name + size) so two different files that
    # happen to share a name and byte size don't reuse a stale DataFrame.
    file_key = (uploaded.name, hashlib.md5(data).hexdigest())
    if "df_key" not in st.session_state or st.session_state.df_key != file_key:
        try:
            st.session_state.df = _read_uploaded(uploaded.name, data)
            st.session_state.df_key = file_key
            st.session_state.pop("results", None)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not read file: {exc}")
            return

    df: pd.DataFrame = st.session_state.df

    st.write(f"**File:** `{uploaded.name}` — {df.shape[0]} rows × {df.shape[1]} columns")
    with st.expander("Preview first 10 rows", expanded=False):
        st.dataframe(df.head(10), use_container_width=True)

    if run_clicked:
        st.session_state.results = _run_agent_pipeline(df)

    results = st.session_state.get("results")
    if not results:
        # A failed run already surfaced its own error; only show the start hint
        # before the first run so the two messages don't contradict each other.
        if not run_clicked:
            st.info("Click **Run Agent** above to start the pipeline.")
        return

    if show_profile:
        with st.expander("1. Data profile", expanded=True):
            _render_profile(results["profile"])

    _render_specs(results["viz_specs"], heading="2. Agent decisions")
    _render_plots(results["plots"])
    _render_evaluations(results["evaluations"])

    final_evals = results["evaluations"]
    final_plots = results["plots"]
    final_specs = results["viz_specs"]

    if results.get("rerun"):
        st.divider()
        st.subheader("🔁 Re-run with critic feedback")
        st.caption("At least one chart scored below 3, so the agent re-ran once.")
        with st.expander("Feedback fed back to the agent", expanded=False):
            st.code(results["rerun"]["feedback"])
        _render_specs(results["rerun"]["viz_specs"], heading="2b. Revised decisions")
        _render_plots(results["rerun"]["plots"], key_prefix="rerun")
        _render_evaluations(results["rerun"]["evaluations"], heading="4b. Critic re-evaluation")
        final_evals = results["rerun"]["evaluations"]
        final_plots = results["rerun"]["plots"]
        final_specs = results["rerun"]["viz_specs"]

    st.divider()
    st.subheader("5. Final summary")
    if final_evals:
        avg_score = sum(ev["score"] for ev in final_evals) / len(final_evals)
        rendered = sum(1 for p in final_plots if p["error"] is None)
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.markdown("**Average critic score**")
            st.markdown(f"### {avg_score:.2f} / 5")
        with col_b:
            st.markdown("**Visualizations generated**")
            st.markdown(f"### {len(final_specs)}")
        with col_c:
            st.markdown("**Successfully rendered**")
            st.markdown(f"### {rendered}")
        if len(final_evals) < len(final_specs):
            st.caption(
                f"⚠️ The critic scored {len(final_evals)} of {len(final_specs)} "
                "charts; the rest were left unscored."
            )
    else:
        st.info("No evaluations available for summary.")


if __name__ == "__main__":
    main()
