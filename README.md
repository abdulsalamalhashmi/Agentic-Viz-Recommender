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

Upload a tabular dataset (CSV or Excel) and let an LLM-powered agent decide
which visualizations best represent the data, generate them with Plotly, and
critique its own choices using a second LLM call. If the critic gives any
chart a low score, the agent re-runs once with the feedback baked in.

## How it works

1. **Profile** — `agent/profiler.py` inspects the dataframe and produces a
   structured profile (column types, basic stats, top correlations, sample rows).
2. **Decide** — `agent/decision_agent.py` sends that profile to Gemini and
   asks for a JSON list of visualization specs.
3. **Generate** — `agent/plot_generator.py` materializes each spec into a
   Plotly figure.
4. **Evaluate** — `agent/evaluator.py` calls Gemini a second time as a judge
   and scores each chart 1–5.
5. **Loop** — If any score is below 3, the decision agent re-runs once with
   the critic's feedback as additional context.

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

The app reads `GEMINI_API_KEY` from the environment, so HF Space secrets work
without code changes.

## Project layout

```
app.py                  # Streamlit UI + agentic loop
agent/
  profiler.py           # Dataframe → structured profile
  decision_agent.py     # Gemini: profile → viz specs
  plot_generator.py     # viz specs → Plotly figures
  evaluator.py          # Gemini: judge the viz specs
utils/
  helpers.py            # Shared helpers (Gemini config, JSON cleanup, prompt summary)
requirements.txt
.env.example
```
