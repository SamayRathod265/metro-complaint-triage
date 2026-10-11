import html
import json

import altair as alt
import pandas as pd
import streamlit as st

import db
from triage import DEPTS, classify, evaluate, llm_available, llm_model, read_photo, score

STATIONS = ["Thaltej", "Vastral Gam", "Kalupur", "Motera Stadium", "Old High Court", "Gandhinagar Sector 1"]
DEPT_LABEL = {"safety": "Safety", "lifts": "Lifts & escalators", "cleanliness": "Cleanliness", "fare": "Fare & tickets",
              "staff_behaviour": "Staff behaviour", "other": "Other"}
METHOD_LABEL = {"llm": "Gemma", "ml": "Naive Bayes", "rules": "Keyword rules"}

st.set_page_config("Every Complaint Heard", "🚇", layout="wide")

# Station-name-board look: yellow board, black lettering, Gujarati / Hindi / English stacked.
# Hind Vadodara covers Gujarati + Latin; Hind covers Devanagari. Offline, the system fallbacks take over.
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Hind+Vadodara:wght@400;500;600;700&family=Hind:wght@400;600&display=swap');
:root {
  --board: #F2C230; --ink: #1D2430; --concrete: #EEF0F2; --line: #CFD4DA; --muted: #5B6573;
  --alarm: #C62828; --alarm-bg: #FBEAEA; --signal: #C77C02; --signal-bg: #FDF3E1; --track: #2E7D5B; --paper: #FFFFFF;
}
html, body, [class*="css"], .stMarkdown, .stTextArea textarea, button, input, select {
  font-family: 'Hind Vadodara', 'Hind', 'Noto Sans Gujarati', 'Noto Sans Devanagari', system-ui, sans-serif;
}
.stApp { background: var(--concrete); color: var(--ink); }
header[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 1.6rem; max-width: 1240px; }

.board { background: var(--board); color: #111; border-radius: 6px; padding: 14px 22px 12px;
         box-shadow: inset 0 0 0 3px #111; display: flex; align-items: flex-end; justify-content: space-between;
         gap: 16px; flex-wrap: wrap; margin-bottom: 14px; }
.board .gu { font-size: 2.1rem; font-weight: 700; line-height: 1.05; }
.board .hi { font-family: 'Hind', sans-serif; font-size: 1.25rem; font-weight: 600; line-height: 1.2; }
.board .en { font-size: 1.25rem; font-weight: 600; letter-spacing: .01em; line-height: 1.2; }
.board .code { font-size: .95rem; font-weight: 600; border: 2px solid #111; border-radius: 4px; padding: 2px 10px; }

.strip { display: flex; gap: 10px; flex-wrap: wrap; margin: 0 0 14px; }
.pill { background: var(--paper); border: 1px solid var(--line); border-radius: 999px; padding: 4px 12px;
        font-size: .92rem; display: inline-flex; align-items: center; gap: 7px; }
.dot { width: 9px; height: 9px; border-radius: 50%; display: inline-block; }

.card { background: var(--paper); border: 1px solid var(--line); border-left: 6px solid var(--line);
        border-radius: 6px 6px 0 0; padding: 12px 16px 12px; margin-bottom: -1rem; border-bottom: none; }
.card + div [data-testid="stExpander"] details { border-radius: 0 0 6px 6px; border-top: none; background: var(--paper); }
.card.u5 { border-left-color: var(--alarm); background: var(--alarm-bg); }
.card.u4 { border-left-color: var(--signal); }
.card .top { display: flex; gap: 12px; align-items: flex-start; }
.card .num { flex: none; width: 34px; height: 34px; border-radius: 6px; display: grid; place-items: center;
             font-weight: 700; font-size: 1.1rem; background: var(--concrete); color: var(--ink); }
.card.u5 .num { background: var(--alarm); color: #fff; }
.card.u4 .num { background: var(--signal); color: #fff; }
.card .text { font-size: 1.12rem; line-height: 1.5; font-weight: 500; }
.chips { display: flex; flex-wrap: wrap; gap: 6px; margin: 8px 0 4px 46px; }
.chip { font-size: .82rem; color: var(--muted); background: var(--concrete); border-radius: 4px; padding: 1px 8px; }
.chip.review { color: var(--signal); background: var(--signal-bg); font-weight: 600; }
.why { margin-left: 46px; color: var(--muted); font-size: .9rem; }
.empty { background: var(--paper); border: 1px dashed var(--line); border-radius: 6px; padding: 18px; color: var(--muted); }
h3.lane { font-size: 1.15rem; margin: 6px 0 8px; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="board">
  <div>
    <div class="gu">દરેક ફરિયાદ સંભળાય</div>
    <div class="hi">हर शिकायत सुनी जाए</div>
    <div class="en">Every Complaint Heard</div>
  </div>
  <div class="code">Complaint triage console</div>
</div>
""", unsafe_allow_html=True)

has_llm = llm_available()
has_net = db.net_online()


def esc(s):
    return html.escape(str(s or ""))


def card(r, show_dept=False):
    u = int(r["urgency"])
    conf = f" {r['confidence']:.0%}" if pd.notna(r.get("confidence")) else ""
    chips = [esc(r["station"]), esc(r["channel"]), esc(r["language"]),
             f"{METHOD_LABEL.get(r['method'], r['method'])}{conf}"]
    if show_dept:
        chips.insert(0, DEPT_LABEL.get(r["department"], r["department"]))
    review = pd.notna(r.get("confidence")) and r["method"] == "ml" and r["confidence"] < 0.5
    chip_html = "".join(f'<span class="chip">{c}</span>' for c in chips)
    if review:
        chip_html += '<span class="chip review">Check this one</span>'
    st.markdown(f"""
<div class="card u{u}">
  <div class="top"><div class="num" title="Urgency {u} of 5">{u}</div><div class="text">{esc(r['text'])}</div></div>
  <div class="chips">{chip_html}</div>
  <div class="why">{esc(r['reason'])}</div>
</div>""", unsafe_allow_html=True)
    with st.expander("Draft reply"):
        reply = st.text_area("Reply to passenger", r["draft_reply"], key=f"r{r['id']}", label_visibility="collapsed")
        if st.button("Save reply", key=f"s{r['id']}", disabled=reply == r["draft_reply"]):
            db.update_reply(r["id"], reply)
            st.rerun()


# ---------------- sidebar: input + sync ----------------
with st.sidebar:
    st.subheader("Add complaints")
    if st.button("Load the 30 demo complaints", width="stretch"):
        for s in json.load(open("sample_complaints.json", encoding="utf-8")):
            db.add(s["channel"], s["station"], s["text"])
        st.rerun()

    with st.form("add", clear_on_submit=True):
        txt = st.text_area("Complaint text", placeholder="Any language: ગુજરાતી, हिंदी, English or Gujlish")
        c1, c2 = st.columns(2)
        stn = c1.selectbox("Station", STATIONS)
        ch = c2.selectbox("Channel", ["app", "twitter", "email", "paper"])
        if st.form_submit_button("Add and triage", width="stretch") and txt.strip():
            cid = db.add(ch, stn, txt.strip())
            db.save_triage(cid, classify(txt.strip(), has_llm))
            st.rerun()

    with st.expander("Read a paper complaint form"):
        p_stn = st.selectbox("Station", STATIONS, key="p_stn")
        img = st.file_uploader("Photo of the form", type=["jpg", "jpeg", "png"]) or st.camera_input("Or take a photo")
        if img and st.button("Read and triage", width="stretch"):
            text, err = read_photo(img.getvalue())
            if err:
                st.warning(f"{err} You can type the complaint above instead.")
            else:
                cid = db.add("paper", p_stn, text)
                db.save_triage(cid, classify(text, has_llm))
                st.success(f"Added: {text[:80]}")

    st.subheader("Sync")
    queued = len(db.unsynced())
    st.caption(f"{queued} complaint{'s' if queued != 1 else ''} waiting to upload.")
    if st.button("Sync now", width="stretch", disabled=not queued):
        ok, n = db.sync()
        if ok:
            st.success(f"Uploaded {n} complaints.")
        else:
            st.warning(f"No network. {n} complaints stay queued and will upload on the next sync.")

# ---------------- status strip ----------------
rows = db.all_rows()
df = pd.DataFrame(rows)
pending = [r for r in rows if r["department"] is None]
done = df[df["department"].notna()] if len(df) else df
urgent = done[done["urgency"] == 5] if len(done) else done


def pill(color, text):
    return f'<span class="pill"><span class="dot" style="background:{color}"></span>{text}</span>'


llm_txt = f"Gemma ready ({llm_model()})" if has_llm else "Gemma not running: offline Naive Bayes in use"
st.markdown('<div class="strip">' +
            pill("var(--track)" if has_llm else "var(--signal)", esc(llm_txt)) +
            pill("var(--track)" if has_net else "var(--alarm)", "Network connected" if has_net else "No network: working offline") +
            pill("var(--ink)", f"{len(done)} triaged, {len(urgent)} urgent") +
            "</div>", unsafe_allow_html=True)

if pending:
    if st.button(f"Triage {len(pending)} new complaint{'s' if len(pending) != 1 else ''}", type="primary"):
        bar = st.progress(0.0)
        for i, r in enumerate(pending, 1):
            db.save_triage(r["id"], classify(r["text"], has_llm))
            bar.progress(i / len(pending), f"{i} of {len(pending)}")
        st.rerun()

queue_tab, inc_tab, mnt_tab, score_tab = st.tabs(["Queue", "Incidents", "Maintenance", "Accuracy"])

# ---------------- queue ----------------
with queue_tab:
    if not len(done):
        st.markdown('<div class="empty">No complaints yet. Load the demo complaints from the sidebar, '
                    'or type one in any language.</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<h3 class="lane">Urgent now ({len(urgent)})</h3>', unsafe_allow_html=True)
        if not len(urgent):
            st.markdown('<div class="empty">Nothing urgent. Danger to people shows up here first, '
                        'whatever department it belongs to.</div>', unsafe_allow_html=True)
        for r in urgent.sort_values("id").to_dict("records"):
            card(r, show_dept=True)

        st.markdown('<h3 class="lane">By department</h3>', unsafe_allow_html=True)
        rest = done[done["urgency"] < 5]
        tabs = st.tabs([f"{DEPT_LABEL[d]} ({(rest['department'] == d).sum()})" for d in DEPTS])
        for tab, dept in zip(tabs, DEPTS):
            with tab:
                sub = rest[rest["department"] == dept].sort_values(["urgency", "id"], ascending=[False, True])
                if not len(sub):
                    st.caption("Nothing waiting here.")
                for r in sub.to_dict("records"):
                    card(r)

# ---------------- incidents ----------------
with inc_tab:
    if not len(done):
        st.caption("Triage some complaints first.")
    else:
        st.caption("Complaints about the same thing at the same station within 2 hours are one incident. "
                   "Ranked by number of complaints × highest urgency.")
        inc = db.incidents(done)
        st.dataframe(inc.assign(department=inc["department"].map(DEPT_LABEL),
                                first=inc["first"].dt.strftime("%d %b, %H:%M"), last=inc["last"].dt.strftime("%d %b, %H:%M"))
                     [["station", "department", "count", "max_urgency", "first", "last"]]
                     .rename(columns={"station": "Station", "department": "Department", "count": "Complaints",
                                      "max_urgency": "Highest urgency", "first": "First reported", "last": "Last reported"}),
                     hide_index=True, width="stretch")
        heat = db.heatmap(done).assign(department=lambda d: d["department"].map(DEPT_LABEL))
        st.markdown('<h3 class="lane">Where complaints cluster</h3>', unsafe_allow_html=True)
        base = alt.Chart(heat).encode(
            x=alt.X("department:N", title=None, axis=alt.Axis(labelAngle=0, labelLimit=160, orient="top")),
            y=alt.Y("station:N", title=None, axis=alt.Axis(labelLimit=200)))
        st.altair_chart(base.mark_rect().encode(color=alt.Color("count:Q", scale=alt.Scale(scheme="yelloworangered"),
                                                                legend=None), tooltip=["station", "department", "count"])
                        + base.mark_text(fontSize=13).encode(text="count:Q"), width="stretch")

# ---------------- maintenance ----------------
with mnt_tab:
    st.caption("Repeated complaints about lifts, ticket machines, gates and toilets become maintenance jobs, "
               "so the night shift hears about a failing asset from passengers before it breaks completely.")
    m = db.maintenance(done) if len(done) else pd.DataFrame()
    if not len(m):
        st.markdown('<div class="empty">No asset has 2 or more complaints yet.</div>', unsafe_allow_html=True)
    else:
        view = (m.drop(columns="ids")
                .assign(department=m["department"].map(DEPT_LABEL),
                        first_reported=pd.to_datetime(m["first_reported"]).dt.strftime("%d %b, %H:%M"))
                .rename(columns={"station": "Station", "asset": "Asset", "department": "Department",
                                 "complaints": "Complaints", "max_urgency": "Highest urgency",
                                 "first_reported": "First reported", "action": "Action"}))
        st.dataframe(view.style.apply(
            lambda r: ["background-color: #FBEAEA" if r["Action"] == "Inspect immediately" else ""] * len(r), axis=1),
            hide_index=True, width="stretch")
        n_jobs = int((m["action"] != "Monitor").sum())
        st.download_button(f"Download night-shift roster ({n_jobs} jobs)", db.roster_csv(m),
                           "night_shift_roster.csv", "text/csv", disabled=not n_jobs)

# ---------------- accuracy ----------------
with score_tab:
    st.caption("Measured on 120 hand-written complaints the model never trained on "
               "(20 per department, in Gujarati, Hindi, English, Gujlish and Hinglish).")
    if st.button("Run accuracy test"):
        samples = json.load(open("eval_complaints.json", encoding="utf-8"))
        runs = {"Naive Bayes + rules (offline)": (False, True), "Keyword rules only": (False, False)}
        if has_llm:
            runs = {f"Gemma ({llm_model()}) + safety checks": (True, True)} | runs
        st.session_state.sc = {}
        for name, (use_ai, use_ml) in runs.items():
            bar = st.progress(0.0, name)
            st.session_state.sc[name] = evaluate(samples, use_ai, use_ml, bar.progress)
            bar.empty()
    if "sc" in st.session_state:
        res = st.session_state.sc
        table = pd.DataFrame({k: score(v) for k, v in res.items()}).T
        table.columns = ["Department correct", "Urgency within 1", "Urgent cases caught", "Seconds per complaint"]
        st.dataframe(table.style.format({c: "{:.0%}" for c in table.columns[:3]} | {table.columns[3]: "{:.3f}"}),
                     width="stretch")
        for name, rs in res.items():
            bad = pd.DataFrame(rs)
            bad = bad[(bad["department"] != bad["gold_department"]) | ((bad["urgency"] - bad["gold_urgency"]).abs() > 1)]
            with st.expander(f"{name}: {len(bad)} mistakes"):
                st.dataframe(bad[["text", "gold_department", "department", "gold_urgency", "urgency", "method"]],
                             hide_index=True, width="stretch")
