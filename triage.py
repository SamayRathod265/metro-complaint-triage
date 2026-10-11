import base64
import json
import os
import re
import time

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN_DATA = os.path.join(HERE, "training_data.json")
MODELS_DIR = os.path.join(HERE, "models")
OLLAMA = "http://localhost:11434/api/generate"
MODEL = "qwen2.5-coder:7b"  # no gemma installed; swap to "gemma3:1b" after `ollama pull gemma3:1b`
DEPTS = ["safety", "cleanliness", "fare", "lifts", "staff_behaviour", "other"]

# Latin keywords match on word boundaries ("aag" must not hit "aage"); native script matches as substrings.
SAFETY = ["fire", "aag", "smoke", "track", "fell", "gir gaya", "gir gayi", "harassment", "harass", "stuck",
          "fasai", "fasayu", "fasaya", "fasayo", "fasai gaya", "phas", "phasa", "phans", "phansa", "phas gaya", "atki", "atka",
          "આગ", "ધુમાડો", "ટ્રેક", "પડી", "ફસાય", "છેડતી", "आग", "धुआं", "ट्रैक", "गिर", "फंस", "छेड़"]
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

PROMPT = """You triage passenger complaints for Gujarat Metro (Ahmedabad-Gandhinagar).
Complaint: {text}

Reply with ONLY a JSON object with these keys:
"department": one of {depts}
"urgency": integer 1-5 (5 = immediate danger to people, 4 = accessibility/service down, 1 = suggestion)
"language": one of Gujarati, Hindi, English, Gujlish (Gujarati in Latin letters), Hinglish
"summary_en": one-line English summary
"draft_reply": short polite reply from Gujarat Metro written in the SAME language and script as the complaint
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
    urgency = {"safety": 5, "lifts": 4, "fare": 3}.get(dept, 2)
    lang = language(text)
    return {"department": dept, "urgency": urgency, "language": lang,
            "summary_en": f"{dept.replace('_', ' ').title()} complaint: {text[:80]}",
            "draft_reply": reply(lang, dept, urgency, text),
            "reason": f"Keyword match for {dept}" if dept != "other" else "No keywords matched",
            "method": "rules", "confidence": None}


ML_MIN_CONF = 0.4  # below this the NB guess is discarded and keyword rules decide


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


def predict_nb(m, text):
    """(label, confidence) where confidence is the softmax-normalised posterior of the top class."""
    import math
    n_docs, V = sum(m["docs"].values()), m["vocab"]
    feats = [f for f in features(text) if any(f in c for c in m["counts"].values())]  # skip never-seen features
    scores = {c: math.log(m["docs"][c] / n_docs) +
                 sum(math.log((m["counts"][c].get(f, 0) + 1) / (m["totals"][c] + V)) for f in feats)
              for c in m["counts"]}
    top = max(scores.values())
    z = sum(math.exp(s - top) for s in scores.values())
    best = max(scores, key=scores.get)
    return best, 1 / z  # exp(top - top) / z


def train_ml():
    """Train department + urgency NB on training_data.json, save as JSON in models/. Pure stdlib, <1s."""
    data = json.load(open(TRAIN_DATA, encoding="utf-8"))
    os.makedirs(MODELS_DIR, exist_ok=True)
    texts = [d["text"] for d in data]
    for target in ("department", "urgency"):
        m = train_nb(texts, [d["gold_" + target] for d in data])
        json.dump(m, open(os.path.join(MODELS_DIR, f"{target}.json"), "w", encoding="utf-8"), ensure_ascii=False)


_ml = {}


def ml_models():
    if not _ml:
        if not os.path.exists(os.path.join(MODELS_DIR, "urgency.json")):
            train_ml()
        _ml.update({t: json.load(open(os.path.join(MODELS_DIR, f"{t}.json"), encoding="utf-8"))
                    for t in ("department", "urgency")})
    return _ml


def ml(text):
    """Naive Bayes department + urgency; raises if department confidence < ML_MIN_CONF so classify() uses rules."""
    m = ml_models()
    dept, conf = predict_nb(m["department"], text)
    urg, uconf = predict_nb(m["urgency"], text)
    if conf < ML_MIN_CONF:
        raise ValueError(f"low ML confidence {conf:.2f}")
    # 5 = immediate danger: only a safety prediction (or the keyword override) may raise the red flag
    urgency, lang = int(urg) if dept == "safety" else min(int(urg), 4), language(text)
    return {"department": dept, "urgency": urgency, "language": lang,
            "summary_en": f"{dept.replace('_', ' ').title()} complaint: {text[:80]}",
            "draft_reply": reply(lang, dept, urgency, text),
            "reason": f"Naive Bayes: {dept} ({conf:.0%} confident), urgency {urgency} ({uconf:.0%})",
            "method": "ml", "confidence": round(conf, 3)}


def ai(text):
    r = requests.post(OLLAMA, json={"model": MODEL, "prompt": PROMPT.format(text=text, depts=DEPTS),
                                    "format": "json", "stream": False}, timeout=20)
    r.raise_for_status()
    out = json.loads(r.json()["response"])
    if out.get("department") not in DEPTS:
        raise ValueError(f"bad department {out.get('department')}")
    out["urgency"] = min(5, max(1, int(out["urgency"])))
    for k in ("language", "summary_en", "draft_reply", "reason"):
        out[k] = str(out.get(k) or "")
    out["method"], out["confidence"] = "llm", None
    return out


def classify(text, use_ai=True, use_ml=True):
    """LLM (if use_ai) → ML (if use_ml and sklearn/models load) → rules; safety keyword override applied last."""
    out = None
    for on, fn in ((use_ai, ai), (use_ml, ml)):
        if on and out is None:
            try:
                out = fn(text)
            except Exception:
                pass
    out = out or rules(text)
    hits = safety_hits(text)
    if hits:  # safety override wins over LLM, ML and rules
        out.update(department="safety", urgency=5,
                   reason=f"Safety keyword(s) {hits} → urgency 5. " + out["reason"])
        if out["method"] != "llm":
            out["draft_reply"] = reply(out["language"], "safety", 5, text)
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
    """Ollama's /api/show for MODEL; fails if the server is down or the model is missing/broken."""
    r = requests.post("http://localhost:11434/api/show", json={"model": MODEL}, timeout=5)
    r.raise_for_status()
    return r.json()


def has_vision():
    return "vision" in model_info().get("capabilities", [])


def extract_text(image_bytes):
    """OCR a photographed paper complaint form via the Ollama model. Raises ValueError if no vision."""
    if not has_vision():
        raise ValueError(f"Model '{MODEL}' has no vision support. Pull a vision model (e.g. gemma3:4b) and set MODEL.")
    r = requests.post(OLLAMA, json={
        "model": MODEL, "format": "json", "stream": False,
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


def online():
    try:
        model_info()
        return True
    except requests.RequestException:
        return False


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
