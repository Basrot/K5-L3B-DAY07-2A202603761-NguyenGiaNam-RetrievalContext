import sys
import json
from pathlib import Path
from collections import Counter

ROOT = Path.cwd()
BONUS = ROOT / "bonus"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(BONUS))

from hybrid_index_v2 import (
    load_records, get_collection, hybrid_search, MyEmbedder
)

OUT = BONUS / "output"
BENCHMARK = OUT / "benchmark_dev_18_reviewed.json"
DEST = OUT / "v2_dev_regression.json"

if DEST.exists():
    raise FileExistsError(
        "V2 DEV result already exists. Preserve the previous result."
    )

questions = json.loads(BENCHMARK.read_text(encoding="utf-8-sig"))
records = load_records()
collection = get_collection()
embedder = MyEmbedder("minilm")

assert len(questions) == 18
assert len(records) == 997
assert collection.count() == 997

def infer_kind(question):
    q = question.casefold()

    rules = [
        ("diagram", (
            "flowchart", "flow chart", "decision diagram"
        )),
        ("chart", (
            "chart", "graph", "figure 8-6",
            "population trend", "general trend"
        )),
        ("image", (
            "screenshot", "pictured", "picture",
            "visual interface", "gui library",
            "software interface", "notebook interface",
            "code editor"
        )),
        ("table", (
            "operator", "membership", "sequence",
            "result type", "integer divided",
            "integer plus", "'in' and 'not in'",
            "division table", "arithmetic type table"
        ))
    ]

    for kind, terms in rules:
        if any(term in q for term in terms):
            return kind

    return None

results = []
print("=== BONUS V2 DEV REGRESSION ===", flush=True)

for q in questions:
    gold = {g["record_id"] for g in q["gold"]}
    route = infer_kind(q["question"])

    hits = hybrid_search(
        question=q["question"],
        records=records,
        collection=collection,
        embedder=embedder,
        kind=route,
        top_k=5,
        depth=10
    )

    found = [h["id"] for h in hits]

    h3 = gold.issubset(set(found[:3]))
    h5 = gold.issubset(set(found[:5]))

    results.append({
        "id": q["id"],
        "type": q["type"],
        "route": route,
        "gold": sorted(gold),
        "retrieved": found,
        "all_hit_at_3": h3,
        "all_hit_at_5": h5
    })

    print(
        q["id"],
        "ALL@3:", int(h3),
        "ALL@5:", int(h5),
        "route:", route or "all",
        flush=True
    )

summary = {
    "split": "dev",
    "index_version": "v2",
    "records": 997,
    "N": 18,
    "all_hit_at_3": sum(r["all_hit_at_3"] for r in results),
    "all_hit_at_5": sum(r["all_hit_at_5"] for r in results),
    "retrieval_only": True,
    "results": results
}

DEST.write_text(
    json.dumps(summary, ensure_ascii=False, indent=2),
    encoding="utf-8"
)

print("\n=== BONUS V2 DEV SCORE ===")
print("All Evidence Hit@3:", summary["all_hit_at_3"], "/18")
print("All Evidence Hit@5:", summary["all_hit_at_5"], "/18")
print("Saved:", DEST)
print("PASS: V2 DEV REGRESSION COMPLETED")

print("\n=== TEST DRAFT FILES ===")
for p in sorted(BONUS.glob("*TEST*.json")) + sorted(BONUS.glob("*test*.json")):
    print(p.name)
