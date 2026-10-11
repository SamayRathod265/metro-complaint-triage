# Every Complaint Heard — Gujarat Metro (PS-07)

2-hour hackathon. Offline-first complaint triage. Must work with NO network.

## Stack
Python, Streamlit, sqlite3, requests, pandas. LLM = local Ollama (`http://localhost:11434/api/generate`,
`format:"json"`, `stream:false`, 20s timeout). `MODEL` constant in triage.py = the gemma model from `ollama list`.

## Files
- `sample_complaints.json` — 30 complaints (app/twitter/email/paper), 6 stations, Gujarati/Hindi/English/Gujlish,
  gold_department + gold_urgency. 4 urgent safety cases; clusters: Thaltej lift (4), Kalupur dirty toilets, Vastral Gam fare/QR.
- `triage.py` — `classify(text)` → {department, urgency 1-5, language, summary_en, draft_reply (same language), reason, method}.
  Ollama first, multilingual keyword rules fallback. Safety override (fire/aag/આગ/track/fell/harassment/stuck) → urgency 5.
- `db.py` — SQLite `complaints.db`, `synced` 0/1 column (outbox pattern).
- `app.py` — Streamlit UI: online/offline indicator, load samples, add complaint, Triage all (progress bar), metrics,
  department tabs sorted by urgency (5 pinned red), editable draft replies, Sync → `sync_outbox.json` (online) or "N queued" (offline).

- Tabs: Queue (dept sub-tabs) · Scorecard (AI vs rules: dept acc, urgency ±1, safety recall, s/complaint, misclassified)
  · Incidents (same station+dept within 2h, score = count × max urgency, altair heatmap).
- Maintenance Alerts tab (PS-02 bridge): `db.maintenance()` = incidents in lifts/fare/cleanliness with 2+ complaints,
  asset via 4-language keywords (`db.ASSETS`), action by urgency/count, CSV "night-shift roster" download.
- Sidebar "Photo of paper form": camera/upload → base64 → Ollama `images` → OCR text → classify → save channel=paper.
  Non-vision model / offline → friendly warning.

## Run
`python -m streamlit run app.py` (streamlit.exe is not on PATH) · tests: `python test_app.py` (7 PASS/FAIL checks)
· quick test: `python triage.py`. Incident grouping lives in `db.incidents()`, photo OCR wrapper is `triage.read_photo()`.

## Offline ML (pure stdlib, no sklearn — network too slow, don't install)
- `triage.train_ml()` → Multinomial NB (char 2-4grams + words, binarised, station names stripped) → `models/*.json`.
- classify(): LLM → NB (`ML_MIN_CONF` 0.4, else rules) → rules; safety override last. method llm/ml/rules + confidence.
- training_data.json: 200 synthetic rows (generator was in scratchpad), avoids keyword vocabulary.
- HONEST RESULT on held-out 30: NB dept 43% / urgency±1 77% vs rules 100%/100%. NB overconfident (≈1.0 when wrong),
  so 0.4 threshold rarely fires; 8 red cards instead of 4. Cause: training vocab ≠ real complaint vocab. Pending user decision.

## Status (2026-10-10) — FEATURE FREEZE
- test_app.py: 22/23 PASS; only fail = Ollama model broken (not fixable in code).
- Sync now probes internet via `db.SYNC_URL` (not Ollama); `db.sync()` returns (synced, n); outbox dedupes by id.
- `db.add` dedupes same station+text (sample batch load is idempotent).
- All features built + headless-tested (Streamlit AppTest) on the rules path.
- `online()` checks `/api/show` for MODEL (not just /api/tags); offline → classify skips AI entirely.
- BLOCKER for AI: `qwen2.5-coder:7b` blob is missing on disk (listed but "not found"). No gemma installed.
  Fix: `ollama pull gemma3:4b` (~3.3GB, has vision for photo OCR) then set MODEL in triage.py. gemma3:1b = text only.
- Departments: safety|cleanliness|fare|lifts|staff_behaviour|other.
