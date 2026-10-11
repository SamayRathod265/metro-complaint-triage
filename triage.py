import base64
import json
import os
import re
import time

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN_DATA = os.path.join(HERE, "training_data.json")
MODELS_DIR = os.path.join(HERE, "models")
OLLAMA_HOST = os.environ.get("OLLAMA_HOST_URL", "http://localhost:11434")
OLLAMA = OLLAMA_HOST + "/api/generate"
# Set METRO_LLM to force a model; otherwise the best installed Gemma is picked (see llm_model()).
MODEL = os.environ.get("METRO_LLM") or None
MODEL_PREFERENCE = ["gemma3:4b", "gemma3:12b", "gemma4", "gemma3:1b", "gemma"]  # vision-capable 4b first
DEPTS = ["safety", "cleanliness", "fare", "lifts", "staff_behaviour", "other"]
DEPT_TITLE = {"safety": "Safety", "cleanliness": "Cleanliness", "fare": "Fare & tickets", "lifts": "Lifts & escalators",
              "staff_behaviour": "Staff behaviour", "other": "Other"}

# Latin keywords match on word boundaries ("aag" must not hit "aage"); native script matches as substrings.
SAFETY = [
    # fire / smoke / electrical
    "fire", "aag", "smoke", "burning smell", "sparks", "short circuit", "shock", "live wire", "current no wire",
    "dhumado", "dhuan", "આગ", "ધુમાડો", "આગની", "आग", "धुआं", "धुआँ", "करंट",
    # falls, track, doors, being trapped
    "track", "fell", "gir gaya", "gir gayi", "stuck", "doors open", "door open", "fasai", "fasayu", "fasaya",
    "fasayo", "fasai gaya", "phas", "phasa", "phans", "phansa", "phas gaya", "atki", "atka",
    "ટ્રેક", "પડી", "ફસાય", "ખુલ્લો રહી", "ट्रैक", "गिर", "फंस",
    # medical emergency
    "fainted", "unconscious", "collapsed", "bleeding", "ambulance", "heart attack", "behosh",
    "બેભાન", "ચક્કર", "बेहोश", "खून",
    # harassment / violence / weapons
    "harassment", "harass", "groping", "molest", "touching my", "knife", "weapon", "threatening", "chaku", "chaaku",
    "chhari", "dhamki", "chhed", "chhedti", "છેડતી", "છરી", "ધમકી", "छेड़", "चाकू", "धमकी",
    # suspicious objects, lost children, crush
    "unattended bag", "suspicious bag", "abandoned bag", "lawaris", "binvarsi", "બિનવારસી", "लावारिस",
    "child missing", "lost child", "bachchu khovai", "bachcha kho", "baccha kho", "stampede",
]
KEYWORDS = {
    "safety": SAFETY + ["danger", "dangerous", "security", "emergency", "ઇમરજન્સી", "खतरा"],
    "lifts": ["lift", "elevator", "escalator", "લિફ્ટ", "એસ્કેલેટર", "लिफ्ट", "एस्केलेटर"],
    "cleanliness": ["dirty", "toilet", "smell", "garbage", "clean", "gandagi", "ganda", "kachra", "dustbin",
                    "ગંદ", "શૌચાલય", "કચરો", "સફાઈ", "गंद", "शौचालय", "कचरा", "सफाई"],
    "fare": ["fare", "refund", "charged", "overcharged", "ticket", "qr", "recharge", "paisa", "paise", "money",
             "ભાડું", "ટિકિટ", "પૈસા", "किराया", "टिकट", "पैसा", "पैसे"],
    "staff_behaviour": ["rude", "staff", "shouted", "behaviour", "behavior", "badtameezi", "guard",
                        "ઉદ્ધત", "સ્ટાફ", "વર્તન", "बदतमीज़ी", "बदतमीजी", "स्टाफ", "व्यवहार"],
}
GUJLISH = {"che", "chhe", "nathi", "bahu", "ma", "ni", "na", "pachi", "thi", "par", "karo", "hato", "thatu", "aaptu"}
HINGLISH = {"hai", "nahi", "gaya", "tha", "bahut", "aur", "kya", "pe", "toh", "kuch", "roz"}

DEPT_NAMES = {  # department as written inside each language's reply
    "Gujarati": {"safety": "સુરક્ષા", "cleanliness": "સફાઈ", "fare": "ભાડા/ટિકિટ", "lifts": "લિફ્ટ/એસ્કેલેટર",
                 "staff_behaviour": "સ્ટાફના વર્તન", "other": "સેવા"},
    "Hindi": {"safety": "सुरक्षा", "cleanliness": "सफाई", "fare": "किराया/टिकट", "lifts": "लिफ्ट/एस्केलेटर",
              "staff_behaviour": "स्टाफ के व्यवहार", "other": "सेवा"},
    "Gujlish": {"safety": "safety", "cleanliness": "safai", "fare": "bhada/ticket", "lifts": "lift/escalator",
                "staff_behaviour": "staff na vartan", "other": "seva"},
    "Hinglish": {"safety": "safety", "cleanliness": "safai", "fare": "kiraya/ticket", "lifts": "lift/escalator",
                 "staff_behaviour": "staff ke vyavhar", "other": "seva"},
    "English": {"safety": "safety", "cleanliness": "cleanliness", "fare": "fare/ticketing", "lifts": "lift/escalator",
                "staff_behaviour": "staff behaviour", "other": "service"},
}
REPLIES = {
    "Gujarati": "આપની {dept} અંગેની ફરિયાદ \"{snip}\" અમને મળી છે. અમે તે સંબંધિત વિભાગને મોકલી છે અને જલ્દી પગલાં લઈશું. - ગુજરાત મેટ્રો",
    "Hindi": "आपकी {dept} से जुड़ी शिकायत \"{snip}\" हमें मिल गई है। हमने इसे संबंधित विभाग को भेज दिया है और जल्द कार्रवाई करेंगे। - गुजरात मेट्रो",
    "Gujlish": "Tamari {dept} vishe ni complaint \"{snip}\" amne mali gai che. Ame sambandhit vibhag ne mokli che ane jaldi pagla laishu. - Gujarat Metro",
    "Hinglish": "Aapki {dept} se judi shikayat \"{snip}\" humein mil gayi hai. Sambandhit vibhag ko bhej di hai, jaldi action lenge. - Gujarat Metro",
    "English": "Thank you, we have received your {dept} complaint \"{snip}\". It has been forwarded to the concerned team and we will act on it soon. - Gujarat Metro",
}
URGENT_REPLY = {
    "Gujarati": "આપની {dept} અંગેની ફરિયાદ \"{snip}\" ગંભીર છે. અમારી ઇમરજન્સી ટીમને તરત જાણ કરવામાં આવી છે. - ગુજરાત મેટ્રો",
    "Hindi": "आपकी {dept} से जुड़ी शिकायत \"{snip}\" गंभीर है। हमारी इमरजेंसी टीम को तुरंत सूचित कर दिया गया है। - गुजरात मेट्रो",
    "Gujlish": "Tamari {dept} vishe ni complaint \"{snip}\" gambhir che. Amari emergency team ne turant jaan kari che. - Gujarat Metro",
    "Hinglish": "Aapki {dept} se judi shikayat \"{snip}\" gambhir hai. Emergency team ko turant suchit kar diya gaya hai. - Gujarat Metro",
    "English": "Your {dept} report \"{snip}\" is being treated as urgent. Our emergency team has been alerted immediately. - Gujarat Metro",
}


def reply(lang, dept, urgency, text):
    """Template reply in the complaint's language, naming the department and quoting the complaint."""
    t = " ".join(text.split())
    snip = t[:60] + ("…" if len(t) > 60 else "")
    return (URGENT_REPLY if urgency == 5 else REPLIES)[lang].format(dept=DEPT_NAMES[lang][dept], snip=snip)

PROMPT = """You sort passenger complaints for Gujarat Metro (Ahmedabad-Gandhinagar).
Complaints may be in Gujarati, Hindi, English, Gujlish (Gujarati in Latin letters) or Hinglish.

Departments:
- safety: danger to people: fire, smoke, electrical hazard, medical emergency, harassment, violence, weapons,
  unattended bags, lost children, crowd crush, people trapped or falling
- lifts: lifts, elevators, escalators not working or faulty
- cleanliness: dirty toilets, garbage, smell, pests, spit or paan stains, no water in washrooms
- fare: tickets, tokens, QR, cards, recharge, refunds, double charges, ticket machines and AFC gates
- staff_behaviour: rude, careless or abusive staff, guards or counter clerks
- other: suggestions, timings, frequency, Wi-Fi, parking, announcements, lost property

Urgency: 5 = someone is in danger right now; 4 = accessibility blocked or risk of injury; 3 = service failure
costing money or time; 2 = discomfort; 1 = suggestion.

Examples:
"Lift bandh che, wheelchair vala upar nathi jai sakta" -> {{"department": "lifts", "urgency": 4, "language": "Gujlish"}}
"कोच में एक आदमी बेहोश हो गया है" -> {{"department": "safety", "urgency": 5, "language": "Hindi"}}
"Please run more trains after 9 pm" -> {{"department": "other", "urgency": 1, "language": "English"}}

Complaint: {text}

Reply with ONLY one JSON object with these keys:
"department": one of {depts}
"urgency": integer 1-5
"language": one of Gujarati, Hindi, English, Gujlish, Hinglish
"summary_en": one-line English summary
"draft_reply": a short, polite reply from Gujarat Metro in the SAME language and script as the complaint
"reason": one sentence explaining the department and urgency"""


def has(text, kw):
    if kw.isascii():
        return re.search(r"\b" + re.escape(kw) + r"\b", text) is not None
    return kw in text


def safety_hits(text):
    t = text.lower()
    return [k for k in SAFETY if has(t, k)]


def language(text):
    if re.search("[઀-૿]", text):
        return "Gujarati"
    if re.search("[ऀ-ॿ]", text):
        return "Hindi"
    words = set(re.findall(r"[a-z]+", text.lower()))
    g, h = len(words & GUJLISH), len(words & HINGLISH)
    if max(g, h) < 2:
        return "English"
    return "Gujlish" if g >= h else "Hinglish"


def rules(text):
    t = text.lower()
    scores = {d: sum(has(t, k) for k in kws) for d, kws in KEYWORDS.items()}
    dept = max(scores, key=scores.get) if max(scores.values()) else "other"
    # ponytail: fixed urgency per department, the LLM does the nuance when it's up
    # safety keywords like "security" mean 4; the danger override in classify() is what raises 5
    urgency = {"safety": 4, "lifts": 4, "fare": 3}.get(dept, 2)
    lang = language(text)
    return {"department": dept, "urgency": urgency, "language": lang,
            "summary_en": f"{dept.replace('_', ' ').title()} complaint: {text[:80]}",
            "draft_reply": reply(lang, dept, urgency, text),
            "reason": f"Keywords point to {DEPT_TITLE[dept]}" if dept != "other" else "No department keywords found",
            "method": "rules", "confidence": None}


LLM_TIMEOUT = 30  # seconds; small Gemma on a laptop CPU takes ~2-10 s
ML_MIN_CONF = 0.5  # calibrated confidence below which the NB guess is shaky (rules decide if they can)
PROMOTE_MIN_SAFETY = 0.2  # min P(safety) for an urgency-5 prediction to move a complaint to the safety queue


# Station names are metadata, not complaint content; left in, they leak spurious station→department links.
STATION_NAMES = ["thaltej", "vastral gam", "vastral", "kalupur", "motera stadium", "motera", "old high court",
                 "gandhinagar sector 1", "gandhinagar", "sector 1", "થલતેજ", "વસ્ત્રાલ ગામ", "કાલુપુર", "મોટેરા",
                 "ઓલ્ડ હાઈકોર્ટ", "ગાંધીનગર સેક્ટર 1", "ગાંધીનગર", "થલતેજ", "वस्त्राल गाम", "कालुपुर", "मोटेरा", "थलतेज", "गांधीनगर"]


def features(text):
    """Lowercased char 2-4grams + word unigrams, each counted once (binarised); works across scripts and Gujlish."""
    t = text.lower()
    for s in STATION_NAMES:
        t = t.replace(s, " ")
    t = " ".join(t.split())
    # split on whitespace/punctuation, not \w: Indic vowel signs aren't \w and would shatter words
    words = [w for w in re.split(r"[\s.,!?;:\"'()\[\]/@#…|-]+", t) if w]
    padded = f" {t} "
    grams = [padded[i:i + n] for n in (2, 3, 4) for i in range(len(padded) - n + 1)]
    return set(grams + ["w:" + w for w in words])


def train_nb(texts, labels):
    """Multinomial Naive Bayes counts; Laplace smoothing is applied at predict time."""
    from collections import Counter
    counts = {}
    for text, label in zip(texts, labels):
        counts.setdefault(str(label), Counter()).update(features(text))
    vocab = set().union(*counts.values())
    return {"docs": dict(Counter(str(l) for l in labels)), "counts": {c: dict(v) for c, v in counts.items()},
            "totals": {c: sum(v.values()) for c, v in counts.items()}, "vocab": len(vocab)}


def nb_probs(m, text):
    """Calibrated class probabilities.

    Naive Bayes treats overlapping char n-grams as independent evidence, so raw posteriors are ~1.0 even
    when wrong. Dividing the log-scores by sqrt(#features) (a temperature) makes confidence track accuracy:
    on the held-out set, predictions >= 0.9 are right ~100% of the time, < 0.7 only about a third.
    """
    import math
    n_docs, V = sum(m["docs"].values()), m["vocab"]
    feats = [f for f in features(text) if any(f in c for c in m["counts"].values())]  # skip never-seen features
    scores = {c: math.log(m["docs"][c] / n_docs) +
                 sum(math.log((m["counts"][c].get(f, 0) + 1) / (m["totals"][c] + V)) for f in feats)
              for c in m["counts"]}
    k = math.sqrt(max(1, len(feats)))
    top = max(scores.values())
    e = {c: math.exp((s - top) / k) for c, s in scores.items()}
    z = sum(e.values())
    return {c: v / z for c, v in e.items()}


def predict_nb(m, text):
    """(label, calibrated confidence of the top class)."""
    p = nb_probs(m, text)
    best = max(p, key=p.get)
    return best, p[best]


def _data_hash():
    import hashlib
    return hashlib.sha256(open(TRAIN_DATA, "rb").read()).hexdigest()[:16]


def train_ml():
    """Train department + urgency NB on training_data.json, save as JSON in models/. Pure stdlib, <1s."""
    data = json.load(open(TRAIN_DATA, encoding="utf-8"))
    os.makedirs(MODELS_DIR, exist_ok=True)
    texts, h = [d["text"] for d in data], _data_hash()
    for target in ("department", "urgency"):
        m = train_nb(texts, [d["gold_" + target] for d in data])
        m["data_hash"] = h  # lets ml_models() notice when training_data.json changes
        json.dump(m, open(os.path.join(MODELS_DIR, f"{target}.json"), "w", encoding="utf-8"), ensure_ascii=False)


_ml = {}


def ml_models():
    """Load the trained models, retraining first if they're missing or were trained on older data."""
    if not _ml:
        h = _data_hash()
        try:
            loaded = {t: json.load(open(os.path.join(MODELS_DIR, f"{t}.json"), encoding="utf-8"))
                      for t in ("department", "urgency")}
            stale = any(m.get("data_hash") != h for m in loaded.values())
        except (OSError, ValueError):
            stale = True
        if stale:
            train_ml()
            loaded = {t: json.load(open(os.path.join(MODELS_DIR, f"{t}.json"), encoding="utf-8"))
                      for t in ("department", "urgency")}
        _ml.update(loaded)
    return _ml


def ml(text):
    """Naive Bayes department + urgency.

    - Urgency 5 means danger to people, which *is* a safety matter: if the urgency model says 5, the
      complaint goes to the safety queue even if the department model guessed otherwise.
    - Below ML_MIN_CONF the guess is shaky: if keyword rules recognise the complaint we raise so classify()
      uses them, otherwise we keep the ML guess and flag it for human review.
    """
    m = ml_models()
    dept_p, urg_p = nb_probs(m["department"], text), nb_probs(m["urgency"], text)
    dept, urg = max(dept_p, key=dept_p.get), max(urg_p, key=urg_p.get)
    conf, uconf = dept_p[dept], urg_p[urg]
    urgency = int(urg)
    notes = []
    # promote only when the danger signal is backed by *some* safety evidence (rats: p5 high, p_safety ~0)
    if urgency == 5 and dept != "safety" and dept_p.get("safety", 0) >= PROMOTE_MIN_SAFETY:
        notes.append(f"moved from {DEPT_TITLE[dept]} to Safety: wording suggests danger to people")
        dept = "safety"
    if dept != "safety":
        urgency = min(urgency, 4)  # only safety (or the keyword override) raises the red flag
    if conf < ML_MIN_CONF:
        if rules(text)["department"] != "other":
            raise ValueError(f"low ML confidence {conf:.2f}, keyword rules recognise it")
        notes.append("not sure, please check")
    lang = language(text)
    reason = f"Model: {DEPT_TITLE[dept]}, {conf:.0%} sure"
    return {"department": dept, "urgency": urgency, "language": lang,
            "summary_en": f"{dept.replace('_', ' ').title()} complaint: {text[:80]}",
            "draft_reply": reply(lang, dept, urgency, text),
            "reason": reason + ("; " + "; ".join(notes) if notes else ""),
            "method": "ml", "confidence": round(conf, 3), "needs_review": conf < ML_MIN_CONF}


_llm = {}


def llm_model():
    """Name of the LLM to use: METRO_LLM if set, else the best installed Gemma, else None. Cached."""
    if "name" not in _llm:
        name = MODEL
        if not name:
            try:
                tags = [m["name"] for m in requests.get(OLLAMA_HOST + "/api/tags", timeout=3).json()["models"]]
                name = next((t for pref in MODEL_PREFERENCE for t in tags if t.startswith(pref)), None)
            except (requests.RequestException, KeyError, ValueError):
                return None  # Ollama down: don't cache, it may start later
        _llm["name"] = name
    return _llm["name"]


def ai(text):
    model = llm_model()
    if not model:
        raise RuntimeError("no LLM installed")
    r = requests.post(OLLAMA, json={"model": model, "prompt": PROMPT.format(text=text, depts=DEPTS),
                                    "format": "json", "stream": False, "options": {"temperature": 0}},
                      timeout=LLM_TIMEOUT)
    r.raise_for_status()
    out = json.loads(r.json()["response"])
    if out.get("department") not in DEPTS:
        raise ValueError(f"bad department {out.get('department')}")
    out["urgency"] = min(5, max(1, int(out.get("urgency", 3))))
    for k in ("language", "summary_en", "draft_reply", "reason"):
        out[k] = str(out.get(k) or "")
    if out.get("language") not in DEPT_NAMES:
        out["language"] = language(text)
    if not out["draft_reply"].strip():
        out["draft_reply"] = reply(out["language"], out["department"], out["urgency"], text)
    out["method"], out["confidence"] = "llm", None
    out["model"] = model
    return out


def classify(text, use_ai=True, use_ml=True):
    """LLM (if use_ai) → Naive Bayes (if use_ml) → keyword rules, then two safety-first checks:

    1. If the LLM answered, Naive Bayes still runs as a second opinion (~1 ms); if it sees danger the
       complaint is flagged red. For safety we want *any* layer's alarm, not a majority vote.
    2. Danger keywords (fire, बेहोश, chhari, …) always force safety / urgency 5.
    """
    out, second = None, None
    for on, fn in ((use_ai, ai), (use_ml, ml)):
        if on and out is None:
            try:
                out = fn(text)
            except Exception:
                pass
    out = out or rules(text)
    if out["method"] == "llm" and use_ml and out["urgency"] < 5:
        try:
            second = ml(text)
        except Exception:
            second = None
        if second and second["department"] == "safety" and second["urgency"] == 5:
            out.update(department="safety", urgency=5,
                       reason="Second opinion (Naive Bayes) sees danger to people. " + out["reason"])
    hits = safety_hits(text)
    if hits:  # safety override wins over LLM, ML and rules
        out.update(department="safety", urgency=5,
                   reason=f"Danger words: {', '.join(hits)}. " + out["reason"])
        if out["method"] != "llm":
            out["draft_reply"] = reply(out["language"], "safety", 5, text)
            out["summary_en"] = f"Safety complaint: {text[:80]}"
    return out


def evaluate(samples, use_ai, use_ml=True, progress=None):
    """Classify every sample, returning gold vs predicted rows with timing."""
    out = []
    for i, s in enumerate(samples, 1):
        t = time.perf_counter()
        r = classify(s["text"], use_ai, use_ml)
        out.append({"id": s["id"], "text": s["text"], "gold_department": s["gold_department"],
                    "gold_urgency": s["gold_urgency"], "department": r["department"], "urgency": r["urgency"],
                    "method": r["method"], "confidence": r["confidence"], "seconds": time.perf_counter() - t})
        if progress:
            progress(i / len(samples))
    return out


def score(rows):
    n = len(rows)
    if not n:
        return {"dept_acc": 0.0, "urg_acc": 0.0, "safety_recall": 0.0, "avg_s": 0.0}
    urgent = [r for r in rows if r["gold_urgency"] == 5]
    return {"dept_acc": sum(r["department"] == r["gold_department"] for r in rows) / n,
            "urg_acc": sum(abs(r["urgency"] - r["gold_urgency"]) <= 1 for r in rows) / n,
            "safety_recall": sum(r["urgency"] == 5 for r in urgent) / len(urgent) if urgent else 1.0,
            "avg_s": sum(r["seconds"] for r in rows) / n}


def model_info():
    """Ollama's /api/show for the chosen model; fails if Ollama is down or no model is installed/working."""
    model = llm_model()
    if not model:
        raise requests.ConnectionError("no Gemma model installed in Ollama")
    r = requests.post(OLLAMA_HOST + "/api/show", json={"model": model}, timeout=5)
    r.raise_for_status()
    return r.json()


def has_vision():
    return "vision" in model_info().get("capabilities", [])


def extract_text(image_bytes):
    """OCR a photographed paper complaint form via the Ollama model. Raises ValueError if no vision."""
    if not has_vision():
        raise ValueError(f"Model '{llm_model()}' can't read images. Run `ollama pull gemma3:4b` for photo reading.")
    r = requests.post(OLLAMA, json={
        "model": llm_model(), "format": "json", "stream": False,
        "prompt": 'This is a photo of a passenger complaint form. Transcribe the complaint text exactly as written, '
                  'in its original language and script. Reply as JSON: {"text": "..."}',
        "images": [base64.b64encode(image_bytes).decode()]}, timeout=60)
    r.raise_for_status()
    return str(json.loads(r.json()["response"]).get("text", "")).strip()


def read_photo(image_bytes):
    """Never raises: returns (text, None) on success or (None, user-facing message)."""
    try:
        text = extract_text(image_bytes)
    except ValueError as e:
        return None, f"Can't read photos with this model: {e}"
    except Exception as e:
        return None, f"Photo reading unavailable (Ollama offline, model broken or timed out): {str(e)[:150]}"
    return (text, None) if text else (None, "No text found in the photo, try a clearer shot.")


def llm_available():
    """True if Ollama is running with a usable Gemma model (this is about the LLM, not internet)."""
    try:
        model_info()
        return True
    except requests.RequestException:
        return False


online = llm_available  # backwards-compatible name


if __name__ == "__main__":
    samples = json.load(open("sample_complaints.json", encoding="utf-8"))
    # rules-only accuracy check (works with no Ollama)
    sc = score(evaluate(samples, use_ai=False))
    print("rules scorecard:", {k: round(v, 3) for k, v in sc.items()})
    assert sc["safety_recall"] == 1.0, "safety override must catch every urgent case"
    assert language("થલતેજ") == "Gujarati" and language("Lift is broken") == "English"
    assert not safety_hits("aage jao") and safety_hits("aag lagi")
    for i in (3, 2, 1):  # Gujarati, Gujlish, English
        s = samples[i - 1]
        print(json.dumps({"id": i, **classify(s["text"])}, ensure_ascii=False, indent=1))
