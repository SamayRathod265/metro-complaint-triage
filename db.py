import json
import sqlite3

import requests

DB = "complaints.db"
# Connectivity probe; sync_outbox.json stands in for the real upload. ponytail: swap for the real API endpoint.
SYNC_URL = "https://clients3.google.com/generate_204"
COLS = ["department", "urgency", "language", "summary_en", "draft_reply", "reason", "method", "confidence"]


def conn():
    c = sqlite3.connect(DB)
    c.row_factory = sqlite3.Row
    c.execute("""CREATE TABLE IF NOT EXISTS complaints (
        id INTEGER PRIMARY KEY AUTOINCREMENT, channel TEXT, station TEXT, text TEXT,
        department TEXT, urgency INTEGER, language TEXT, summary_en TEXT, draft_reply TEXT,
        reason TEXT, method TEXT, synced INTEGER DEFAULT 0,
        created_at TEXT DEFAULT (datetime('now', 'localtime')), confidence REAL)""")
    try:  # older complaints.db files predate the confidence column
        c.execute("ALTER TABLE complaints ADD COLUMN confidence REAL")
    except sqlite3.OperationalError:
        pass
    return c


def add(channel, station, text):
    """Insert a complaint; an identical one (same station + text) returns the existing id instead."""
    with conn() as c:
        dup = c.execute("SELECT id FROM complaints WHERE station=? AND text=?", (station, text)).fetchone()
        if dup:
            return dup["id"]
        return c.execute("INSERT INTO complaints (channel, station, text) VALUES (?,?,?)",
                         (channel, station, text)).lastrowid


def save_triage(cid, r):
    with conn() as c:
        c.execute(f"UPDATE complaints SET {', '.join(k + '=?' for k in COLS)}, synced=0 WHERE id=?",
                  [r.get(k) for k in COLS] + [cid])


def update_reply(cid, reply):
    with conn() as c:
        c.execute("UPDATE complaints SET draft_reply=?, synced=0 WHERE id=?", (reply, cid))


def all_rows():
    with conn() as c:
        return [dict(r) for r in c.execute("SELECT * FROM complaints ORDER BY urgency DESC, id")]


def unsynced():
    return [r for r in all_rows() if not r["synced"]]


def incidents(done):
    """Group triaged rows (DataFrame) by station + department within 2h; rank by count × max urgency."""
    import pandas as pd
    inc = done.assign(t=pd.to_datetime(done["created_at"])).sort_values(["station", "department", "t"])
    # ponytail: chained 2h gaps, so a steady trickle forms one long incident; use fixed windows if that matters
    new = inc.groupby(["station", "department"])["t"].diff().gt(pd.Timedelta(hours=2))
    key = inc[["station", "department"]]
    inc["incident"] = (new | (key != key.shift()).any(axis=1)).cumsum()
    g = inc.groupby("incident").agg(station=("station", "first"), department=("department", "first"),
                                    count=("id", "size"), max_urgency=("urgency", "max"),
                                    first=("t", "min"), last=("t", "max"), ids=("id", list))
    g["score"] = g["count"] * g["max_urgency"]
    return g.sort_values("score", ascending=False)


# Order matters: escalator before lift, AFC before generic gate.
ASSETS = [
    ("escalator", ["escalator", "એસ્કેલેટર", "एस्केलेटर"]),
    ("lift", ["lift", "elevator", "લિફ્ટ", "लिफ्ट"]),
    ("AFC gate", ["afc"]),
    ("vending machine", ["vending", "machine", "tvm", "મશીન", "मशीन"]),
    ("ticket gate", ["gate", "qr", "scan", "ગેટ", "गेट"]),
    ("toilet", ["toilet", "washroom", "શૌચાલય", "शौचालय"]),
    ("dustbins / platform", ["dustbin", "garbage", "kachra", "કચરો", "ડસ્ટબિન", "कचरा"]),
]
DEFAULT_ASSET = {"lifts": "lift", "fare": "ticket gate", "cleanliness": "station area"}


def asset_of(texts, department):
    """Most-mentioned asset across an incident's complaint texts (+ summaries); department default otherwise."""
    from collections import Counter
    hits = Counter()
    for t in texts:
        t = (t or "").lower()
        for name, kws in ASSETS:
            if any(k in t for k in kws):
                hits[name] += 1
                break  # one asset per complaint, first match wins
    return hits.most_common(1)[0][0] if hits else DEFAULT_ASSET[department]


def action(max_urgency, count):
    if max_urgency >= 4:
        return "Inspect immediately"
    if max_urgency == 3 or count >= 3:
        return "Tonight's night shift"
    return "Monitor"


def maintenance(done):
    """Maintenance list from incidents in lifts/fare/cleanliness with 2+ complaints, most urgent first."""
    import pandas as pd
    inc = incidents(done)
    inc = inc[inc["department"].isin(DEFAULT_ASSET) & (inc["count"] >= 2)]
    by_id = done.set_index("id")
    rows = [{"station": i["station"],
             "asset": asset_of(by_id.loc[i["ids"], "text"].tolist() + by_id.loc[i["ids"], "summary_en"].tolist(), i["department"]),
             "department": i["department"], "complaints": i["count"], "max_urgency": i["max_urgency"],
             "first_reported": i["first"], "action": action(i["max_urgency"], i["count"]), "ids": i["ids"]}
            for i in inc.to_dict("records")]
    out = pd.DataFrame(rows, columns=["station", "asset", "department", "complaints", "max_urgency",
                                      "first_reported", "action", "ids"])
    rank = {"Inspect immediately": 0, "Tonight's night shift": 1, "Monitor": 2}
    return out.sort_values(["action", "max_urgency", "complaints"], key=lambda c: c.map(rank) if c.name == "action" else -c)


def net_online():
    try:
        return requests.get(SYNC_URL, timeout=3).ok
    except requests.RequestException:
        return False


def sync(path="sync_outbox.json"):
    """Returns (synced, n): (True, rows written) when online, else (False, rows still queued)."""
    rows = unsynced()
    if not net_online():
        return False, len(rows)
    try:
        old = {r["id"]: r for r in json.load(open(path, encoding="utf-8"))}
    except (OSError, ValueError):
        old = {}
    old.update({r["id"]: r for r in rows})  # a re-synced edit replaces its old copy, never duplicates
    with open(path, "w", encoding="utf-8") as f:
        json.dump(list(old.values()), f, ensure_ascii=False, indent=1)
    # mark synced only after the file write succeeded
    with conn() as c:
        c.executemany("UPDATE complaints SET synced=1 WHERE id=?", [(r["id"],) for r in rows])
    return True, len(rows)


def heatmap(done):
    return done.groupby(["station", "department"]).size().reset_index(name="count")


def roster_csv(m):
    """Night-shift roster CSV bytes: every maintenance job that isn't just 'Monitor'."""
    return m[m["action"] != "Monitor"].drop(columns="ids").to_csv(index=False).encode("utf-8-sig")
