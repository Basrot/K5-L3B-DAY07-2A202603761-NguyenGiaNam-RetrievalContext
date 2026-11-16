import json
from pathlib import Path
from collections import defaultdict

from hybrid_index import (
    load_records, get_collection, hybrid_search, MyEmbedder
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "bonus/output"
BENCHMARK = OUT / "benchmark_dev_draft.json"

questions = json.loads(BENCHMARK.read_text(encoding="utf-8-sig"))
records = load_records()
collection = get_collection()
embedder = MyEmbedder("minilm")
by_id = {r["id"]: r for r in records}

print("=== TABLE MARKDOWN QUALITY ===")

for page in (118, 131, 132):
    path = OUT / f"table_page_{page}_markdown.md"
    if path.exists():
        print(f"\n--- PAGE {page} ---")
        content = path.read_text(encoding="utf-8-sig")
        lines = [
            line for line in content.splitlines()
            if line.strip().startswith("|")
        ]
        for line in lines[:9]:
            print(ascii(line[:180]))

print("\n=== BONUS RETRIEVAL DEV EVALUATION ===")

results = []
stats = defaultdict(lambda: {"hit1": 0, "hit3": 0, "hit5": 0, "N": 0})

for q in questions:
    gold_ids = {g["record_id"] for g in q["gold"]}
    assert gold_ids.issubset(by_id), q["id"]

    hits = hybrid_search(
        q["question"], records, collection, embedder,
        top_k=5, depth=10
    )

    found = [h["id"] for h in hits]
    hit1 = bool(gold_ids.intersection(found[:1]))
    hit3 = bool(gold_ids.intersection(found[:3]))
    hit5 = bool(gold_ids.intersection(found[:5]))

    kind = q["type"]
    stats[kind]["N"] += 1
    stats[kind]["hit1"] += int(hit1)
    stats[kind]["hit3"] += int(hit3)
    stats[kind]["hit5"] += int(hit5)

    results.append({
        "query_id": q["id"],
        "type": kind,
        "question": q["question"],
        "gold_ids": sorted(gold_ids),
        "hit_at_1": hit1,
        "hit_at_3": hit3,
        "hit_at_5": hit5,
        "retrieved": hits,
        "human_verified": False
    })

    print(
        q["id"], kind,
        "H1:", int(hit1),
        "H3:", int(hit3),
        "H5:", int(hit5),
        "top:", found[:3],
        flush=True
    )

n = len(results)
summary = {
    "split": "dev",
    "cases": n,
    "hit_at_1": sum(r["hit_at_1"] for r in results),
    "hit_at_3": sum(r["hit_at_3"] for r in results),
    "hit_at_5": sum(r["hit_at_5"] for r in results),
    "by_type": dict(stats),
    "provisional": True,
    "note": "Candidate gold record IDs; manual verification pending"
}

(OUT / "dev_retrieval_traces.json").write_text(
    json.dumps(results, ensure_ascii=False, indent=2),
    encoding="utf-8"
)

(OUT / "dev_retrieval_summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2),
    encoding="utf-8"
)

print("\n=== BONUS DEV RETRIEVAL SCORE ===")
print("Hit@1:", summary["hit_at_1"], "/", n)
print("Hit@3:", summary["hit_at_3"], "/", n)
print("Hit@5:", summary["hit_at_5"], "/", n)

for kind, s in stats.items():
    print(
        kind, ":", s["hit3"], "/", s["N"],
        "Hit@3"
    )

print("\nPASS: DEV retrieval evaluation completed")
print("NOTE: Provisional scores, not final bonus grade")
