# Every Complaint Heard (PS-07, DevFest Gandhinagar Metro Hacks)

Offline-first complaint triage for metro station controllers. Must keep working with no network. Now a portfolio
project: keep numbers honest and reproducible.

## Run
- App: `python -m streamlit run app.py` (streamlit.exe may not be on PATH)
- Tests: `python test_app.py` (PASS/FAIL/SKIP per check; the Gemma check skips if Ollama isn't running)
- Accuracy report: `python tools/evaluate.py` → `results/scorecard.md` (add `--no-llm` to skip Gemma)
- Training data: `python tools/generate_training_data.py` (deterministic; test fails if the JSON is stale)

## Pipeline (triage.py `classify`)
1. Gemma via local Ollama, auto-detected by `llm_model()` (prefers gemma3:4b, then 1b; `METRO_LLM` env overrides).
2. Pure-stdlib Naive Bayes (char 2-4grams + words). Confidence is temperature-calibrated (`nb_probs`, divide by
   sqrt(#features)). Below `ML_MIN_CONF` (0.5): keyword rules if they match, else keep ML and flag "Check this one".
   Urgency-5 prediction moves a complaint to safety only if P(safety) >= `PROMOTE_MIN_SAFETY` (0.2).
3. Keyword rules (5 language styles). Rules give safety urgency 4; only danger words give 5.
Then safety-first checks: NB second opinion after an LLM answer, and `SAFETY` danger words force urgency 5.
Models in `models/` retrain automatically when `training_data.json` changes (hash stored in the model JSON).

## Data
- `sample_complaints.json`: 30 demo complaints for the UI (the keyword rules were written against these).
- `eval_complaints.json`: 120 hand-written held-out complaints, 20 per department, 5 language styles. Never train on it.
- `training_data.json`: 873 generated rows from `tools/generate_training_data.py`, no overlap with either set above.

## Current results (offline, eval set)
NB pipeline: dept 92%, urgency±1 96%, safety recall 92% (12/13), 3 false alarms. Rules only: dept 68%.
Thresholds and danger words were set after error analysis on the eval set, so recall is slightly optimistic.
Gemma not benchmarked yet: run `tools/evaluate.py` with Ollama running and update the README table.

## UI (app.py)
Station-name-board header (yellow #F2C230, Hind Vadodara font), status pills (Gemma / network / counts),
"Urgent now" lane across departments, department tabs, Incidents (+ heatmap), Maintenance (night-shift CSV),
Accuracy tab on the eval set. Theme in `.streamlit/config.toml`. Screenshots for the README in `docs/screenshots/`.

## Conventions
- Git commits are authored by Samay Rathod with no Co-Authored-By line.
- Departments: safety | cleanliness | fare | lifts | staff_behaviour | other.
