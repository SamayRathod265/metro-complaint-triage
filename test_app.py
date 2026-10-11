"""Plain-Python checks: python test_app.py → PASS/FAIL per check, never stops early, exit 1 on any FAIL."""
import csv
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import traceback
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
STATIONS = {"Thaltej", "Vastral Gam", "Kalupur", "Motera Stadium", "Old High Court", "Gandhinagar Sector 1"}
FIELDS = ("department", "urgency", "language", "summary_en", "draft_reply", "reason", "method", "confidence")
results = []


def check(section, name):
    def wrap(fn):
        try:
            detail = fn() or ""
            results.append((section, name, "PASS", detail))
        except Exception as e:
            tb = traceback.extract_tb(e.__traceback__)[-1]
            results.append((section, name, "FAIL", f"{type(e).__name__}: {str(e)[:160]} (line {tb.lineno})"))
        return fn
    return wrap


def temp_db():
    import db
    db.DB = os.path.join(tempfile.mkdtemp(), "t.db")
    return db.DB


def script(s):
    if re.search("[઀-૿]", s):
        return "Gujarati"
    if re.search("[ऀ-ॿ]", s):
        return "Devanagari"
    return "Latin"


def valid(r):
    missing = [k for k in FIELDS if k not in r]
    assert not missing, f"missing {missing}"
    import triage
    assert r["department"] in triage.DEPTS, r["department"]
    assert isinstance(r["urgency"], int) and 1 <= r["urgency"] <= 5, r["urgency"]
    assert r["method"] in ("llm", "ml", "rules"), r["method"]
    c = r["confidence"]
    assert (c is None) == (r["method"] != "ml") and (c is None or 0 < c <= 1), c
    for k in ("language", "summary_en", "draft_reply", "reason"):
        assert isinstance(r[k], str) and r[k].strip(), f"empty {k}"
    json.loads(json.dumps(r, ensure_ascii=False))
    return r


# ---------------- 1. SETUP ----------------
@check("1 Setup", "Ollama reachable, model responds <20s")
def _():
    import triage
    assert triage.online(), f"Ollama down or model '{triage.MODEL}' missing/broken"
    t = time.time()
    r = valid(triage.ai("Lift at Thaltej is broken"))
    s = time.time() - t
    assert s < 20, f"{s:.1f}s"
    return f"{triage.MODEL}: {r['department']}/{r['urgency']} in {s:.1f}s"


@check("1 Setup", "Imports work; DB created fresh if missing")
def _():
    import pandas, requests, streamlit, altair  # noqa: F401
    import db, triage  # noqa: F401
    path = temp_db()
    assert not os.path.exists(path)
    assert db.all_rows() == [] and os.path.exists(path)
    return "fresh DB created"


@check("1 Setup", "sample_complaints.json: 30 items, gold labels, station")
def _():
    import triage
    s = json.load(open("sample_complaints.json", encoding="utf-8"))
    assert len(s) == 30, len(s)
    for x in s:
        assert x["gold_department"] in triage.DEPTS and 1 <= x["gold_urgency"] <= 5, x["id"]
        assert x["station"] in STATIONS and x["text"].strip(), x["id"]
    assert len({x["id"] for x in s}) == 30, "duplicate ids"
    return "30 valid"


# ---------------- 2. CLASSIFICATION ----------------
CASES = [
    ("Lift ma ek kaka fasaya che, jaldi aavo", {"lifts", "safety"}, lambda u: u == 5, "gujlish"),
    ("સ્ટેશન પર ટોઇલેટ ખૂબ ગંદું છે", {"cleanliness"}, lambda u: u <= 3, "gujarati"),
    ("टिकट मशीन ने मेरे 50 रुपये खा लिए", {"fare"}, lambda u: True, "hindi"),
    ("Security guard was very rude to my mother", {"staff_behaviour"}, lambda u: True, "english"),
    ("Platform par dhuan aave che, aag ni smell", {"safety"}, lambda u: u == 5, None),
]
for text, depts, urg_ok, lang in CASES:
    @check("2 Classification", text[:38])
    def _(text=text, depts=depts, urg_ok=urg_ok, lang=lang):
        import triage
        r = valid(triage.classify(text))
        assert r["department"] in depts, f"dept {r['department']} not in {depts}"
        assert urg_ok(r["urgency"]), f"urgency {r['urgency']}"
        if lang:
            assert r["language"].lower() == lang, f"language {r['language']}"
        return f"{r['department']}/{r['urgency']}/{r['language']}/{r['method']}"


@check("2 Classification", "draft_reply script matches complaint script")
def _():
    import triage
    bad = []
    for text, *_ in CASES:
        r = triage.classify(text)
        if script(r["draft_reply"]) != script(text):
            bad.append(f"{script(text)}→{script(r['draft_reply'])}")
    assert not bad, bad
    return "5/5 match"


# ---------------- 3. SAFETY OVERRIDE ----------------
@check("3 Safety", "Keywords force urgency 5 even when LLM says 1")
def _():
    import triage
    llm_says_low = {"department": "other", "urgency": 1, "language": "English", "summary_en": "x",
                    "draft_reply": "ok", "reason": "LLM thinks minor", "method": "llm", "confidence": None}
    texts = ["small fire near gate", "coach ma aag lagi", "સ્ટેશન પર આગ લાગી", "phone dropped on the track",
             "old man fell on stairs", "we are stuck in the lift", "kaka lift ma fasaya", "harassment in ladies coach"]
    bad = []
    with mock.patch.object(triage, "ai", side_effect=lambda t: dict(llm_says_low)):
        for t in texts:
            r = triage.classify(t)
            if r["urgency"] != 5:
                bad.append(t)
    assert not bad, f"not urgent: {bad}"
    return f"{len(texts)}/{len(texts)} → 5"


# ---------------- 4. RESILIENCE ----------------
@check("4 Resilience", "Wrong Ollama port → rules, no crash")
def _():
    import triage
    with mock.patch.object(triage, "OLLAMA", "http://localhost:1/api/generate"):
        r = valid(triage.classify("Lift at Thaltej is broken"))
        r2 = valid(triage.classify("Lift at Thaltej is broken", use_ml=False))
    assert r["method"] in ("ml", "rules") and r2["method"] == "rules", (r["method"], r2["method"])
    return f"→ {r['method']}; with ML off → rules"


@check("4 Resilience", "Ollama timeout → rules")
def _():
    import requests, triage
    with mock.patch.object(triage.requests, "post", side_effect=requests.Timeout("simulated")):
        r = valid(triage.classify("Toilet is dirty at Kalupur", use_ml=False))
    assert r["method"] == "rules"
    return "method=rules"


@check("4 Resilience", "LLM invalid JSON / bad fields → rules")
def _():
    import triage
    for body in ["{not json", '{"department": "spaceships", "urgency": 3}', '{"department": "fare", "urgency": "high"}', "[]"]:
        fake = mock.Mock()
        fake.json.return_value = {"response": body}
        with mock.patch.object(triage.requests, "post", return_value=fake):
            r = valid(triage.classify("ticket gate not opening", use_ml=False))
        assert r["method"] == "rules", (body, r["method"])
    return "4 bad responses handled"


@check("4 Resilience", "Sync offline → stays synced=0, returns queued count")
def _():
    import db
    temp_db()
    out = os.path.join(tempfile.mkdtemp(), "outbox.json")
    db.add("app", "Thaltej", "a"); db.add("app", "Thaltej", "b")
    with mock.patch.object(db, "SYNC_URL", "http://localhost:1/"):
        ok, n = db.sync(out)
    assert (ok, n) == (False, 2), (ok, n)
    assert len(db.unsynced()) == 2 and not os.path.exists(out)
    return "2 queued, nothing written"


@check("4 Resilience", "Sync online (mock) → synced=1, outbox written, queue 0, no dupes")
def _():
    import db
    temp_db()
    out = os.path.join(tempfile.mkdtemp(), "outbox.json")
    a = db.add("app", "Thaltej", "a"); db.add("app", "Thaltej", "b")
    with mock.patch.object(db, "net_online", return_value=True):
        assert db.sync(out) == (True, 2)
        assert len(db.unsynced()) == 0 and len(json.load(open(out, encoding="utf-8"))) == 2
        assert db.sync(out) == (True, 0)
        db.update_reply(a, "edited")  # edit re-queues the row
        assert db.sync(out) == (True, 1)
    box = json.load(open(out, encoding="utf-8"))
    assert len(box) == 2, f"outbox has {len(box)} rows (dupes)"
    assert [r for r in box if r["id"] == a][0]["draft_reply"] == "edited"
    return "3 syncs, 2 rows, edit replaced"


# ---------------- 5. EDGE CASES ----------------
@check("5 Edge", "Empty / whitespace / emoji / 2000 chars / mixed script")
def _():
    import triage
    for t in ["", "   \n\t", "🚇😡🔥🙏", "lift broken " * 170, "Thaltej ni લિફ્ટ बंद है, please fix"]:
        valid(triage.classify(t, use_ai=triage.online()))
    return "5 inputs ok"


@check("5 Edge", "Duplicate complaint added twice → one row")
def _():
    import db
    temp_db()
    a = db.add("app", "Kalupur", "Toilet dirty")
    b = db.add("twitter", "Kalupur", "Toilet dirty")
    c = db.add("app", "Thaltej", "Toilet dirty")
    assert a == b and a != c and len(db.all_rows()) == 2
    return "same station+text deduped; other station kept"


# ---------------- 6. FEATURES ----------------
def triaged_samples():
    import pandas as pd
    import db, triage
    temp_db()
    for s in json.load(open("sample_complaints.json", encoding="utf-8")):
        db.save_triage(db.add(s["channel"], s["station"], s["text"]), triage.rules(s["text"]))
    return pd.DataFrame(db.all_rows())


@check("6 Features", "Scorecard: numbers in [0,1], safety recall, empty input ok")
def _():
    import triage
    s = json.load(open("sample_complaints.json", encoding="utf-8"))
    sc = triage.score(triage.evaluate(s, use_ai=False, use_ml=False))
    assert all(0 <= sc[k] <= 1 for k in ("dept_acc", "urg_acc", "safety_recall")), sc
    assert triage.score([])["dept_acc"] == 0.0
    no_urgent = triage.score([{"department": "x", "gold_department": "x", "urgency": 2, "gold_urgency": 2, "seconds": 0}])
    assert no_urgent["safety_recall"] == 1.0, no_urgent  # nothing urgent to miss, and no ZeroDivisionError
    return ", ".join(f"{k}={v:.2f}" for k, v in sc.items())


@check("6 Features", "Incidents: Thaltej lifts = ONE incident; rank = count × max urgency; heatmap")
def _():
    import db
    done = triaged_samples()
    inc = db.incidents(done)
    th = inc[(inc.station == "Thaltej") & (inc.department == "lifts")]
    assert len(th) == 1 and th["count"].iloc[0] == 5, th
    assert (inc["score"] == inc["count"] * inc["max_urgency"]).all()
    assert inc["score"].is_monotonic_decreasing
    heat = db.heatmap(done)
    assert heat["count"].sum() == 30 and set(heat.columns) == {"station", "department", "count"}
    return f"Thaltej lifts x5, score {th['score'].iloc[0]}, heatmap {len(heat)} cells"


@check("6 Features", "Maintenance: Thaltej lift count, action logic, CSV export")
def _():
    import db
    m = db.maintenance(triaged_samples())
    th = m[(m.station == "Thaltej") & (m.asset == "lift")]
    assert len(th) == 1 and th["complaints"].iloc[0] == 5, th
    assert db.action(5, 1) == "Inspect immediately" and db.action(4, 2) == "Inspect immediately"
    assert db.action(3, 2) == "Tonight's night shift" and db.action(2, 3) == "Tonight's night shift"
    assert db.action(2, 2) == "Monitor"
    rows = list(csv.DictReader(io.StringIO(db.roster_csv(m).decode("utf-8-sig"))))
    assert rows and {"station", "asset", "complaints", "max_urgency", "first_reported", "action"} <= set(rows[0])
    return f"{len(m)} jobs, CSV {len(rows)} rows"


@check("6 Features", "Image path: generated PNG with text → no crash")
def _():
    import triage
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (360, 60), "white")
    ImageDraw.Draw(img).text((10, 20), "Lift broken at Thaltej station", fill="black")
    buf = io.BytesIO(); img.save(buf, "PNG")
    text, err = triage.read_photo(buf.getvalue())
    assert (text is None) != (err is None), (text, err)
    return f"text={text!r}" if text else f"graceful: {err[:70]}"


# ---------------- 6b. OFFLINE ML (Naive Bayes) ----------------
@check("6b ML", "training_data.json: ~200, balanced, 4 languages, no overlap with 30 test samples")
def _():
    from collections import Counter
    import triage
    tr = json.load(open(triage.TRAIN_DATA, encoding="utf-8"))
    test = json.load(open("sample_complaints.json", encoding="utf-8"))
    depts = Counter(x["gold_department"] for x in tr)
    assert 180 <= len(tr) <= 220, len(tr)
    assert set(depts) == set(triage.DEPTS) and max(depts.values()) - min(depts.values()) <= 3, depts
    assert set(Counter(x["gold_urgency"] for x in tr)) == {1, 2, 3, 4, 5}
    assert {"Gujarati", "Hindi", "English", "Gujlish"} <= {x["language"] for x in tr}
    assert not {x["text"] for x in tr} & {x["text"] for x in test}, "test leak"
    assert not {x["id"] for x in tr} & {x["id"] for x in test}, "id clash"
    return f"{len(tr)} rows, {dict(depts)}"


@check("6b ML", "Train is pure stdlib, <5s, saves JSON models that reload")
def _():
    import triage
    assert "sklearn" not in sys.modules
    t = time.time()
    triage.train_ml()
    s = time.time() - t
    assert s < 5, f"{s:.1f}s"
    for name in ("department", "urgency"):
        m = json.load(open(os.path.join(triage.MODELS_DIR, f"{name}.json"), encoding="utf-8"))
        assert {"docs", "counts", "totals", "vocab"} <= set(m)
    triage._ml.clear()
    assert set(triage.ml_models()) == {"department", "urgency"}
    return f"trained in {s:.2f}s"


@check("6b ML", "predict_nb: label + confidence in (0,1]; fits its own training data")
def _():
    import triage
    tr = json.load(open(triage.TRAIN_DATA, encoding="utf-8"))
    m = triage.ml_models()["department"]
    hits = 0
    for x in tr:
        label, conf = triage.predict_nb(m, x["text"])
        assert label in triage.DEPTS and 0 < conf <= 1, (label, conf)
        hits += label == x["gold_department"]
    assert hits / len(tr) > 0.9, f"train acc {hits / len(tr):.0%}"  # sanity: NB must at least learn its training set
    return f"train-set acc {hits / len(tr):.0%}"


@check("6b ML", "Pipeline: ML used offline; confidence < 0.4 → rules; ML broken → rules")
def _():
    import triage
    r = valid(triage.classify("{st} par elevator be divas thi bandh che", use_ai=False))
    assert r["method"] in ("ml", "rules")
    with mock.patch.object(triage, "ML_MIN_CONF", 1.01):  # nothing clears the bar
        assert valid(triage.classify("elevator broken", use_ai=False))["method"] == "rules"
    with mock.patch.object(triage, "ml_models", side_effect=OSError("models missing")):
        assert valid(triage.classify("elevator broken", use_ai=False))["method"] == "rules"
    r = valid(triage.classify("small fire near the elevator", use_ai=False))
    assert r["urgency"] == 5 and r["department"] == "safety", "safety override must still win over ML"
    return f"elevator → {r['method']}, low-conf → rules, broken → rules, override ok"


@check("6b ML", "Template reply names department + quotes complaint, in its language")
def _():
    import triage
    r = triage.reply("Gujarati", "cleanliness", 2, "સ્ટેશન પર ટોઇલેટ ખૂબ ગંદું છે")
    assert "સફાઈ" in r and "ટોઇલેટ" in r, r
    r = triage.reply("Hindi", "fare", 3, "टिकट मशीन ने पैसे खा लिए")
    assert "किराया" in r and "मशीन" in r, r
    r = triage.reply("English", "safety", 5, "x" * 200)
    assert "urgent" in r and "…" in r and len(r) < 250, r
    return "gu/hi/en templates ok, long text truncated"


@check("6b ML", "Scorecard on held-out 30: ML and rules rows")
def _():
    import triage
    s = json.load(open("sample_complaints.json", encoding="utf-8"))
    out = []
    for name, use_ml in (("ML", True), ("rules", False)):
        sc = triage.score(triage.evaluate(s, use_ai=False, use_ml=use_ml))
        assert all(0 <= sc[k] <= 1 for k in ("dept_acc", "urg_acc", "safety_recall"))
        out.append(f"{name} dept={sc['dept_acc']:.0%}")
    return ", ".join(out)


# ---------------- 7. APP LAUNCH ----------------
@check("7 Launch", "App script runs end-to-end (AppTest: load → triage → all tabs)")
def _():
    import db
    from streamlit.testing.v1 import AppTest
    temp_db()
    at = AppTest.from_file(os.path.join(HERE, "app.py"), default_timeout=120).run()
    at.sidebar.button[0].click().run()   # load samples
    at.sidebar.button[0].click().run()   # load again → still 30 (dedupe)
    at.button[0].click().run()           # triage all
    assert not at.exception, at.exception
    assert len(db.all_rows()) == 30, len(db.all_rows())
    return f"{len(at.error)} red urgent cards, {len(at.tabs)} tabs, no exceptions"


@check("7 Launch", "streamlit headless 15s, no tracebacks")
def _():
    import requests
    p = subprocess.Popen([sys.executable, "-m", "streamlit", "run", "app.py", "--server.headless", "true",
                          "--server.port", "8599"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    try:
        time.sleep(10)
        health = requests.get("http://localhost:8599/_stcore/health", timeout=5).text
        time.sleep(5)
    finally:
        p.terminate()
        out = p.communicate(timeout=10)[0].decode("utf-8", "replace")
    assert health == "ok", health
    assert "Traceback" not in out and "Error" not in out, out[-400:]
    return "health ok, clean log"


if __name__ == "__main__":
    w = max(len(n) for _, n, _, _ in results)
    for sec, name, status, detail in results:
        print(f"{status}  [{sec}] {name:<{w}}  {detail}")
    fails = sum(r[2] == "FAIL" for r in results)
    print(f"\n{len(results) - fails}/{len(results)} passed")

    import triage
    samples = json.load(open("sample_complaints.json", encoding="utf-8"))
    runs = [("ML (Naive Bayes)", False, True), ("Rules only", False, False)]
    if triage.online():
        runs.insert(0, ("LLM", True, False))
    print("\nSCORECARD (30 held-out samples)" + ("" if triage.online() else "  — LLM unavailable, no LLM row"))
    for name, use_ai, use_ml in runs:
        rows = triage.evaluate(samples, use_ai, use_ml)
        sc = triage.score(rows)
        used = sum(r["method"] == {"LLM": "llm", "ML (Naive Bayes)": "ml"}.get(name, "rules") for r in rows)
        print(f"  {name:<18} dept {sc['dept_acc']:.0%}  urgency±1 {sc['urg_acc']:.0%}  "
              f"safety recall {sc['safety_recall']:.0%}  {sc['avg_s']:.3f}s/complaint  ({used}/30 by this method)")
    sys.exit(1 if fails else 0)
