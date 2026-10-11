"""Score every triage layer on the held-out test set and write results/scorecard.json + results/scorecard.md.

    python tools/evaluate.py            # all layers (LLM only if Ollama + a Gemma model are running)
    python tools/evaluate.py --no-llm   # skip the LLM

Test set: eval_complaints.json — 120 hand-written complaints (20 per department, 5 language styles) that are
never used for training. Safety recall = share of truly urgent (gold urgency 5) complaints flagged urgency 5.
"""
import json
import os
import sys
import time
from collections import defaultdict

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import triage  # noqa: E402


def run(samples, label, fn):
    rows = []
    for i, s in enumerate(samples, 1):
        t = time.perf_counter()
        try:
            r = fn(s["text"])
        except Exception as e:  # a layer that refuses (low confidence, no model) counts as wrong
            r = {"department": None, "urgency": 0, "method": f"error: {type(e).__name__}"}
        rows.append({**s, "pred_department": r["department"], "pred_urgency": r["urgency"],
                     "method": r.get("method"), "seconds": time.perf_counter() - t})
        if i % 20 == 0 or i == len(samples):
            print(f"  {label}: {i}/{len(samples)}", flush=True)
    return rows


def metrics(rows):
    n = len(rows)
    urgent = [r for r in rows if r["gold_urgency"] == 5]
    not_urgent = [r for r in rows if r["gold_urgency"] < 5]
    return {"n": n,
            "dept_acc": sum(r["pred_department"] == r["gold_department"] for r in rows) / n,
            "urg_within_1": sum(abs((r["pred_urgency"] or 0) - r["gold_urgency"]) <= 1 for r in rows) / n,
            "safety_recall": sum(r["pred_urgency"] == 5 for r in urgent) / max(1, len(urgent)),
            "false_red": sum(r["pred_urgency"] == 5 for r in not_urgent),
            "avg_seconds": sum(r["seconds"] for r in rows) / n}


def main():
    samples = json.load(open(os.path.join(HERE, "eval_complaints.json"), encoding="utf-8"))
    layers = {}
    if "--no-llm" not in sys.argv and triage.llm_available():
        model = triage.llm_model()
        layers[f"LLM only ({model})"] = triage.ai
        layers[f"Full pipeline ({model} + NB + rules)"] = lambda t: triage.classify(t, True, True)
    else:
        print("LLM not available: skipping (start Ollama and `ollama pull gemma3:4b` to include it)")
    layers["Naive Bayes + rules (offline pipeline)"] = lambda t: triage.classify(t, False, True)
    layers["Keyword rules only"] = lambda t: triage.classify(t, False, False)

    results, by_lang = {}, {}
    for name, fn in layers.items():
        rows = run(samples, name, fn)
        results[name] = metrics(rows)
        groups = defaultdict(list)
        for r in rows:
            groups[r["language"]].append(r)
        by_lang[name] = {lang: round(metrics(g)["dept_acc"], 3) for lang, g in sorted(groups.items())}
        results[name]["misclassified"] = [
            {"text": r["text"], "gold": r["gold_department"], "pred": r["pred_department"],
             "gold_urgency": r["gold_urgency"], "pred_urgency": r["pred_urgency"]}
            for r in rows if r["pred_department"] != r["gold_department"]]

    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    json.dump({"test_set": "eval_complaints.json", "n": len(samples), "layers": results, "dept_acc_by_language": by_lang},
              open(os.path.join(HERE, "results", "scorecard.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    langs = sorted({s["language"] for s in samples})
    md = ["| Layer | Department accuracy | Urgency ±1 | Safety recall | False red alerts | Time / complaint |",
          "|---|---|---|---|---|---|"]
    for name, m in results.items():
        md.append(f"| {name} | {m['dept_acc']:.0%} | {m['urg_within_1']:.0%} | {m['safety_recall']:.0%} | "
                  f"{m['false_red']} | {m['avg_seconds']:.3f} s |")
    md += ["", "Department accuracy by language:", "", "| Layer | " + " | ".join(langs) + " |",
           "|---|" + "---|" * len(langs)]
    for name, accs in by_lang.items():
        md.append(f"| {name} | " + " | ".join(f"{accs.get(l, 0):.0%}" for l in langs) + " |")
    open(os.path.join(HERE, "results", "scorecard.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")
    print("\n".join(md))
    print("\nSaved results/scorecard.json and results/scorecard.md")


if __name__ == "__main__":
    main()
