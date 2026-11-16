import json
import re
from pathlib import Path
from collections import defaultdict

from hybrid_index import (
    load_records, get_collection, hybrid_search, MyEmbedder
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "bonus/output"
BENCHMARK = OUT / "benchmark_dev_draft.json"

def infer_kind(question):
    q = question.casefold()

    if any(x in q for x in (
        "flowchart", "flow chart", "decision diagram"
    )):
        return "diagram"

    if any(x in q for x in (
        "chart", "graph", "figure 8-6",
        "population trend", "general trend"
    )):
        return "chart"

    if any(x in q for x in (
        "screenshot", "pictured", "picture",
        "visual interface", "gui library",
        "software interface", "notebook interface",
        "code editor"
    )):
        return "image"

    if any(x in q for x in (
        "operator", "membership", "sequence",
        "result type", "integer divided",
        "integer plus", "'in' and 'not in'"
    )):
        return "table"

    return None

questions = json.loads(
    BENCHMARK.read_text(encoding="utf-8-sig")
)

records = load_records()
collection = get_collection()
embedder = MyEmbedder("minilm")

results = []
print("=== BONUS DEV QUERY ROUTING EXPERIMENT ===", flush=True)

for q in questions:
    route = infer_kind(q["question"])
    gold = {g["record_id"] for g in q["gold"]}

    baseline = hybrid_search(
        q["question"], records, collection, embedder,
        top_k=5, depth=10
    )

    routed = hybrid_search(
        q["question"], records, collection, embedder,
        top_k=5, depth=10, kind=route
    )

    def score(hits):
        ids = [h["id"] for h in hits]
        return {
            "any_hit_at_3": bool(gold.intersection(ids[:3])),
            "all_hit_at_3": gold.issubset(set(ids[:3])),
            "any_hit_at_5": bool(gold.intersection(ids[:5])),
            "all_hit_at_5": gold.issubset(set(ids[:5])),
            "ids": ids
        }

    before = score(baseline)
    after = score(routed)

    record = {
        "query_id": q["id"],
        "question": q["question"],
        "type_for_analysis_only": q["type"],
        "query_inferred_kind": route,
        "gold_ids": sorted(gold),
        "before": before,
        "after": after,
        "provisional": True
    }

    results.append(record)

    print(
        q["id"],
        "route:", route or "all",
        "before H3:", int(before["any_hit_at_3"]),
        "after H3:", int(after["any_hit_at_3"]),
        "all gold H5:", int(after["all_hit_at_5"]),
        flush=True
    )

def totals(key, metric):
    return sum(int(r[key][metric]) for r in results)

summary = {
    "split": "dev",
    "cases": len(results),
    "baseline": {
        "any_hit_at_3": totals("before", "any_hit_at_3"),
        "all_hit_at_3": totals("before", "all_hit_at_3"),
        "all_hit_at_5": totals("before", "all_hit_at_5")
    },
    "query_routing": {
        "any_hit_at_3": totals("after", "any_hit_at_3"),
        "all_hit_at_3": totals("after", "all_hit_at_3"),
        "all_hit_at_5": totals("after", "all_hit_at_5")
    },
    "limitations": [
        "Development-only experiment",
        "Gold labels not yet human verified",
        "Heuristic routing may overfit DEV",
        "Visual records use manually authored captions",
        "No answer grounding or agent score measured"
    ]
}

(OUT / "dev_query_routing_traces.json").write_text(
    json.dumps(results, ensure_ascii=False, indent=2),
    encoding="utf-8"
)

(OUT / "dev_query_routing_summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2),
    encoding="utf-8"
)

print("\n=== BONUS ROUTING COMPARISON ===")
print("Original:", summary["baseline"])
print("Query routing:", summary["query_routing"])
print("Cases:", len(results))
print("PASS: DEV routing comparison completed")
