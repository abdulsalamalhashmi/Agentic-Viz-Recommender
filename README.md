---
title: Agentic Data Visualization Recommender
emoji: 📊
colorFrom: indigo
colorTo: purple
sdk: streamlit
sdk_version: "1.36.0"
app_file: app.py
pinned: false
license: mit
---

# Agentic Data Visualization Recommender

**🔗 Live demo:** https://huggingface.co/spaces/AbsiElhashmy/agentic-data-viz-recommender

Upload a tabular dataset (CSV or Excel) and an LLM-powered **agent** decides which
visualizations best represent it. It **uses tools to interrogate the data**, generates
the charts with Plotly, and then **critiques its own choices** with a second LLM call —
re-running once if any chart scores poorly.

Built as a semester project for **SEN4018** and deployed on Hugging Face Spaces with Streamlit.

## How it works

1. **Profile** — `agent/profiler.py` inspects the dataframe locally: column types, basic
   stats, top correlations, sample rows. No API call — nothing leaves your machine here.
2. **Inspect — tool use** — the agent calls real data-inspection **tools** via Gemini
   *function calling* (`agent/tools.py`, running pandas): `describe_column`, `correlation`,
   `top_values`, `group_means`. It interrogates the data before committing — e.g. confirming
   a correlation before choosing a scatter plot. The tool calls are shown live in the app's
   reasoning log.
3. **Decide** — `agent/decision_agent.py` combines the profile and tool results and returns
   a JSON list of visualization specs.
4. **Generate** — `agent/plot_generator.py` materializes each spec into a Plotly figure.
5. **Critique** — `agent/evaluator.py` calls Gemini a second time as an LLM judge and scores
   each chart 1–5 with feedback.
6. **Re-run** — if any chart scores below 3, the decision agent re-runs once with the
   critic's feedback baked in.

## Features

- **Genuine tool use** — the agent function-calls pandas data-inspection tools before deciding.
- **10 chart types** — scatter, bar, grouped bar, histogram, heatmap, line, area, box, pie, treemap.
- **Self-critique loop** — an LLM critic scores every chart 1–5 and the agent re-runs once on weak results.
- **On-demand data story** — one extra LLM call writes a plain-language summary of the insights.
- **Downloadable HTML report** — every chart + reasoning + critic scores in a single self-contained file.
- **Robust by design** — model fallback + retry/backoff, friendly error messages, large-data sampling, null-safe charts.
- **Polished UI** — blue Streamlit theme, a sidebar for controls, per-chart critic-score badges, and session caching (re-opening the same file costs no API calls).

## How it meets the SEN4018 requirements

| Requirement | Where it lives |
|---|---|
| **Autonomous decision loop** (the agent decides what to do next) | decide → critique → conditional re-run in `app.py::_run_agent_pipeline`; the agent also autonomously decides *which* tools to call |
| **Tool usage capability** | Gemini **function calling** over pandas tools in `agent/tools.py` (`describe_column`, `correlation`, `top_values`, `group_means`) |
| **Evaluation framework — LLM in the loop** | LLM-as-critic scores each chart 1–5 with feedback in `agent/evaluator.py`, feeding the re-run |

## Quickstart

```bash
git clone <this-repo>
cd <this-repo>

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

cp .env.example .env
# edit .env and paste your Gemini API key

streamlit run app.py
```

Get a Gemini API key at <https://aistudio.google.com/app/apikey>.

## Deploying to Hugging Face Spaces

1. Create a new Space with **SDK = Streamlit**.
2. Push this repository to the Space's git remote.
3. In the Space's **Settings → Variables and secrets**, add a secret named
   `GEMINI_API_KEY` with your key as the value.

The app reads `GEMINI_API_KEY` from the environment, so HF Space secrets work without
code changes. The theme is configured in `.streamlit/config.toml`.

## Project layout

```
app.py                    # Streamlit UI + agentic pipeline
agent/
  profiler.py             # Dataframe → structured profile (local, no API)
  tools.py                # pandas data-inspection tools the agent function-calls
  decision_agent.py       # Gemini: profile (+ tool use) → viz specs
  plot_generator.py       # viz specs → Plotly figures (10 chart types)
  evaluator.py            # Gemini: LLM-as-critic, scores charts 1–5
  storyteller.py          # Gemini: optional plain-language "data story"
utils/
  helpers.py              # Gemini config, function calling, retries/fallback, JSON cleanup
  report.py               # single-file HTML report builder
.streamlit/config.toml    # app theme
requirements.txt
.env.example
```

## Tech stack

Streamlit · Google Gemini (`google-genai`, function calling) · pandas · Plotly · Hugging Face Spaces.
