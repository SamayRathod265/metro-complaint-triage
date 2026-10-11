# Every Complaint Heard

Offline-first complaint triage for Gujarat Metro — built at **DevFest Gandhinagar Metro Hacks, problem statement PS-07**.

## Problem

Passenger complaints arrive through the app, Twitter, email and paper forms, in Gujarati, Hindi, English and Gujlish
(Gujarati typed in Latin letters). Urgent safety reports get buried among routine ones, and staff on trains and
stations often have no network. Every complaint should be read, routed and answered — even offline.

## Features

- **Multilingual triage**: department (safety, cleanliness, fare, lifts, staff behaviour, other), urgency 1–5,
  language detection, and a draft reply in the complaint's own language and script.
- **Urgency + safety override**: fire / aag / આગ / track / fell / stuck / fasaya / harassment … always force urgency 5,
  whatever the model says. Urgent cards are pinned in red.
- **Incidents**: complaints for the same station and department within 2 hours are merged, ranked by
  count × max urgency, with a station × department heatmap.
- **Maintenance alerts** (bridges to PS-02): repeat lift / ticket gate / vending machine / cleanliness complaints become
  maintenance jobs (Inspect immediately / Tonight's night shift / Monitor), exportable as a night-shift roster CSV.
- **Offline sync queue**: complaints live in SQLite with a `synced` flag (outbox pattern); Sync writes unsynced rows to `sync_outbox.json` (a stand-in for the real upload) when the
  network is up, otherwise shows "N items queued".
- **Scorecard**: accuracy of each classifier against 30 held-out, hand-labelled sample complaints.
- **Paper forms**: photo/upload → vision LLM text extraction → triage (needs a vision model; degrades gracefully).

## 3-layer offline design

```
complaint ──► 1. LLM (local Ollama, e.g. Gemma) ── if unavailable / bad JSON / timeout ──┐
                                                                                          ▼
              2. Naive Bayes (pure Python, char 2-4grams + words, trained on 200 synthetic
                 multilingual complaints) ── if confidence < 0.4 ──┐
                                                                    ▼
              3. Keyword rules (Gujarati, Hindi, English, Gujlish)
                                                                    │
              safety keyword override applied last to every layer ◄─┘
```

Each result records `method` (llm / ml / rules) and the ML confidence. No layer needs internet; the Naive Bayes
model uses only the Python standard library and trains in under a second.

## Run

```bash
pip install streamlit requests pandas
python -m streamlit run app.py
```

Optional LLM layer: install [Ollama](https://ollama.com), `ollama pull gemma3:4b`, and set `MODEL` in `triage.py`.
Without it the app runs on Naive Bayes + rules.

Tests (plain Python, prints PASS/FAIL per check): `python test_app.py`

## Scorecard (honest numbers)

Evaluated on the 30 held-out samples in `sample_complaints.json` (never used for training):

| Layer | Department accuracy | Urgency ±1 | Safety recall | Time / complaint |
|---|---|---|---|---|
| LLM | not measured (no working local model during the hackathon) | | | |
| Naive Bayes | 43% | 77% | 100% | ~0.005 s |
| Keyword rules | 100% | 100% | 100% | ~0.002 s |

Caveats: the rules and the 30 test samples were written by the same team, so 100% is optimistic. The Naive Bayes
training data deliberately avoided the keyword vocabulary, and real complaints use exactly those everyday words
("lift", "toilet", "ticket") — the model has not seen them, which explains its low accuracy. Safety recall stays at
100% because the keyword override runs after every layer.
