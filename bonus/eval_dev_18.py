import json
from collections import Counter, defaultdict
from pathlib import Path

from hybrid_index import (
    load_records, get_collection, hybrid_search, MyEmbedder
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "bonus/output"
SOURCE = OUT / "benchmark_dev_18_draft.json"
BENCHMARK = OUT / "benchmark_dev_18_v2.json"
RESULTS = OUT / "dev_18_comparison.json"

if BENCHMARK.exists() or RESULTS.exists():
    raise FileExistsError("DEV v2 files already exist; preserve them")

questions = json.loads(SOURCE.read_text(encoding="utf-8-sig"))
assert len(questions) == 18

# Replace duplicated DEV question with a different table cell.
q17 = next(q for q in questions if q["id"] == "bonus_dev_017")
assert q17["gold"][0]["record_id"] == "bonus_p118_table_1"
q17["question"] = (
    "According to the division table, what is the "
    "result type of float divided by complex?"
)
q17["expected_answer"] = "complex"
q17["review_notes"] = (
    "Replaced a duplicate question; evidence is "
    "the float row and complex column on PDF page 118."
)

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
        "integer plus", "'in' and 'not in'",
        "division table", "arithmetic type table"
    )):
        return "table"

    return None

assert len({q["id"] for q in questions}) == 18
assert len({q["question"].casefold() for q in questions}) == 18

records = load_records()
by_id = {r["id"]: r for r in records}

for q in questions:
    assert q["split"] == "dev"
    assert q["answerable"] is True
    assert q["gold"]
    for g in q["gold"]:
        assert g["record_id"] in by_id, q["id"]
        assert g["pdf_page"] == by_id[g["record_id"]]["pdf_page"]

BENCHMARK.write_text(
    json.dumps(questions, ensure_ascii=False, indent=2),
    encoding="utf-8"
)

collection = get_collection()
embedder = MyEmbedder("minilm")

def score(hits, gold):
    ids = [h["id"] for h in hits]
    return {
        "any_at_3": bool(gold.intersection(ids[:3])),
        "all_at_3": gold.issubset(set(ids[:3])),
        "all_at_5": gold.issubset(set(ids[:5])),
        "ids": ids
    }

rows = []
print("=== BONUS DEV 18 EVALUATION ===", flush=True)

for q in questions:
    gold = {g["record_id"] for g in q["gold"]}
    route = infer_kind(q["question"])

    original = hybrid_search(
        q["question"], records, collection, embedder,
        top_k=5, depth=10
    )

    routed = hybrid_search(
        q["question"], records, collection, embedder,
        top_k=5, depth=10, kind=route
    )

    before = score(original, gold)
    after = score(routed, gold)

    rows.append({
        "query_id": q["id"],
        "type": q["type"],
        "inferred_route": route,
        "gold_ids": sorted(gold),
        "original": before,
        "routed": after
    })

    print(
        q["id"],
        "type:", q["type"],
        "route:", route or "all",
        "original H3:", int(before["any_at_3"]),
        "routed H3:", int(after["any_at_3"]),
        "all H5:", int(after["all_at_5"]),
        flush=True
    )

def aggregate(method):
    return {
        "any_hit_at_3": sum(r[method]["any_at_3"] for r in rows),
        "all_hit_at_3": sum(r[method]["all_at_3"] for r in rows),
        "all_hit_at_5": sum(r[method]["all_at_5"] for r in rows)
    }

summary = {
    "split": "dev",
    "cases": len(rows),
    "original": aggregate("original"),
    "query_routing": aggregate("routed"),
    "types": dict(Counter(q["type"] for q in questions)),
    "human_verified": False,
    "note": (
        "DEV-only provisional retrieval evaluation. "
        "Gold and image descriptions still require review."
    )
}

RESULTS.write_text(
    json.dumps(
        {"summary": summary, "traces": rows},
        ensure_ascii=False, indent=2
    ),
    encoding="utf-8"
)

print("\n=== BONUS DEV 18 FINAL COMPARISON ===")
print("Original:", summary["original"])
print("Query routing:", summary["query_routing"])
print("Total:", summary["cases"])
print("Saved benchmark:", BENCHMARK)
print("Saved evaluation:", RESULTS)
print("PASS: DEV 18 evaluation completed")
