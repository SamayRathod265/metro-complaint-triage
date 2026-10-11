import json

import altair as alt
import pandas as pd
import streamlit as st

import db
from triage import DEPTS, classify, evaluate, online, read_photo, score

STATIONS = ["Thaltej", "Vastral Gam", "Kalupur", "Motera Stadium", "Old High Court", "Gandhinagar Sector 1"]

st.set_page_config("Every Complaint Heard", "🚇", layout="wide")
st.title("🚇 Every Complaint Heard — Gujarat Metro")

is_online = online()
with st.sidebar:
    st.markdown("### 🟢 Online (AI)" if is_online else "### 🔴 Offline (rules)")
    st.caption("Ollama reachable" if is_online else "Ollama unreachable — keyword rules in use")

    if st.button("Load sample batch"):
        for s in json.load(open("sample_complaints.json", encoding="utf-8")):
            db.add(s["channel"], s["station"], s["text"])
        st.rerun()

    with st.form("add", clear_on_submit=True):
        st.markdown("**Add complaint**")
        ch = st.selectbox("Channel", ["app", "twitter", "email", "paper"])
        stn = st.selectbox("Station", STATIONS)
        txt = st.text_area("Text")
        if st.form_submit_button("Add") and txt.strip():
            db.add(ch, stn, txt.strip())
            st.rerun()

    with st.expander("📷 Photo of paper form"):
        p_stn = st.selectbox("Station", STATIONS, key="p_stn")
        img = st.camera_input("Take photo") or st.file_uploader("…or upload", type=["jpg", "jpeg", "png"])
        if img and st.button("Read & triage"):
            text, err = read_photo(img.getvalue())
            if err:
                st.warning(f"📷 {err} Type the complaint in 'Add complaint' instead.")
            else:
                cid = db.add("paper", p_stn, text)
                db.save_triage(cid, classify(text, is_online))
                st.success(f"Saved #{cid}: {text[:80]}")

    if st.button("Sync"):
        ok, n = db.sync()
        if ok:
            st.success(f"Synced {n} items to sync_outbox.json")
        else:
            st.warning(f"No network — {n} items queued")

rows = db.all_rows()
pending = [r for r in rows if r["department"] is None]
if st.button(f"⚡ Triage all ({len(pending)} pending)", disabled=not pending):
    bar = st.progress(0.0)
    for i, r in enumerate(pending, 1):
        db.save_triage(r["id"], classify(r["text"], is_online))
        bar.progress(i / len(pending), f"{i}/{len(pending)}")
    st.rerun()

df = pd.DataFrame(rows)
done = df[df["department"].notna()] if len(df) else df
c = st.columns(5)
c[0].metric("Total", len(df))
c[1].metric("Triaged", len(done))
c[2].metric("🚨 Urgent (5)", int((done["urgency"] == 5).sum()) if len(done) else 0)
c[3].metric("LLM / ML / rules", " / ".join(str((done["method"] == m).sum()) for m in ("llm", "ml", "rules")) if len(done) else "0 / 0 / 0")
c[4].metric("Unsynced", int((df["synced"] == 0).sum()) if len(df) else 0)

queue_tab, score_tab, inc_tab, mnt_tab = st.tabs(["📥 Queue", "📊 Scorecard", "🔥 Incidents", "🔧 Maintenance Alerts"])

with mnt_tab:
    st.caption("Repeat complaints about lifts, ticket gates/machines and cleanliness, turned into maintenance jobs (PS-02).")
    m = db.maintenance(done) if len(done) else pd.DataFrame()
    if not len(m):
        st.info("No asset with 2+ complaints yet.")
    else:
        st.dataframe(m.style.apply(lambda r: ["background-color: #f8d7da" if r["action"] == "Inspect immediately" else ""] * len(r), axis=1),
                     hide_index=True)
        n_jobs = int((m["action"] != "Monitor").sum())
        st.download_button(f"⬇️ Export night-shift roster ({n_jobs} jobs)", db.roster_csv(m),
                           "night_shift_roster.csv", "text/csv", disabled=not n_jobs)

with score_tab:
    st.caption("Runs both classifiers on sample_complaints.json against the gold labels.")
    if st.button("Run scorecard"):
        samples = json.load(open("sample_complaints.json", encoding="utf-8"))
        runs = {"ML (Naive Bayes)": (False, True), "Rules only": (False, False)}
        if is_online:
            runs = {"LLM": (True, False)} | runs
        st.session_state.sc = {}
        for name, (use_ai, use_ml) in runs.items():
            bar = st.progress(0.0, name)
            st.session_state.sc[name] = evaluate(samples, use_ai, use_ml, bar.progress)
    if "sc" in st.session_state:
        res = st.session_state.sc
        st.caption("Held-out: the 30 samples are never used for training (ML trains on training_data.json).")
        table = pd.DataFrame({k: score(v) for k, v in res.items()}).T
        table.columns = ["Dept accuracy", "Urgency ±1 accuracy", "Safety recall", "Avg s / complaint"]
        st.dataframe(table.style.format({c: "{:.0%}" for c in table.columns[:3]} | {table.columns[3]: "{:.3f}"}))
        for name, rs in res.items():
            want = {"LLM": "llm", "ML (Naive Bayes)": "ml"}.get(name)
            fell_back = sum(r["method"] != want for r in rs) if want else 0
            if fell_back:
                st.warning(f"{name} fell back to rules on {fell_back}/{len(rs)} items (unavailable or low confidence).")
        for name, rs in res.items():
            bad = pd.DataFrame(rs)
            bad = bad[(bad["department"] != bad["gold_department"]) | ((bad["urgency"] - bad["gold_urgency"]).abs() > 1)]
            st.markdown(f"**{name} — misclassified ({len(bad)})**")
            st.dataframe(bad.drop(columns="seconds"), hide_index=True)

with inc_tab:
    if not len(done):
        st.info("Triage some complaints first.")
    else:
        st.dataframe(db.incidents(done), hide_index=True)
        heat = db.heatmap(done)
        st.altair_chart(alt.Chart(heat).mark_rect().encode(
            x="department:N", y="station:N", color=alt.Color("count:Q", scale=alt.Scale(scheme="reds")),
            tooltip=["station", "department", "count"]) +
            alt.Chart(heat).mark_text().encode(x="department:N", y="station:N", text="count:Q"),
            width="stretch")

with queue_tab:
    if not len(done):
        st.info("Nothing triaged yet — load the sample batch, then ⚡ Triage all.")
    for tab, dept in zip(st.tabs([f"{d} ({(done['department'] == d).sum()})" for d in DEPTS]) if len(done) else [], DEPTS):
        with tab:
            for r in done[done["department"] == dept].sort_values(["urgency", "id"], ascending=[False, True]).to_dict("records"):
                with st.container(border=True):
                    conf = f" {r['confidence']:.0%}" if pd.notna(r.get("confidence")) else ""
                    head = f"**#{r['id']} · {r['station']} · {r['channel']} · {r['language']} · urgency {r['urgency']}** · _{r['method']}{conf}_"
                    if r["urgency"] == 5:
                        st.error("🚨 " + head)
                    else:
                        st.markdown(head)
                    st.write(r["text"])
                    st.caption(f"Summary: {r['summary_en']}")
                    st.markdown(f"**Why {'urgent' if r['urgency'] >= 4 else 'this rating'}:** {r['reason']}")
                    reply = st.text_area("Draft reply", r["draft_reply"], key=f"r{r['id']}")
                    if reply != r["draft_reply"] and st.button("Save reply", key=f"s{r['id']}"):
                        db.update_reply(r["id"], reply)
                        st.rerun()
