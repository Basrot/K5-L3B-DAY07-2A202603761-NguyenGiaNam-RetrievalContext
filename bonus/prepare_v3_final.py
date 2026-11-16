import sys
import json
from pathlib import Path

ROOT = Path.cwd()
BONUS = ROOT / "bonus"
OUT = BONUS / "output"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(BONUS))

from hybrid_index_v3 import (
    load_records, get_collection, hybrid_search, MyEmbedder
)

def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))

def save_new(path, value):
    if path.exists():
        raise FileExistsError(f"File already exists: {path}")
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

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

print("=== BONUS V3 FINAL DEV REGRESSION ===", flush=True)

dev = read_json(OUT / "benchmark_dev_18_reviewed.json")
records = load_records()
collection = get_collection()
embedder = MyEmbedder("minilm")

assert len(dev) == 18
assert len(records) == 999
assert collection.count() == 999

results = []

for q in dev:
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

    retrieved = [h["id"] for h in hits]

    result = {
        "id": q["id"],
        "route": route,
        "gold": sorted(gold),
        "retrieved": retrieved,
        "all_at_3": gold.issubset(set(retrieved[:3])),
        "all_at_5": gold.issubset(set(retrieved[:5]))
    }
    results.append(result)

    print(
        q["id"],
        "ALL@3:", int(result["all_at_3"]),
        "ALL@5:", int(result["all_at_5"]),
        flush=True
    )

at3 = sum(r["all_at_3"] for r in results)
at5 = sum(r["all_at_5"] for r in results)

regression = {
    "split": "dev",
    "index_version": "v3",
    "records": 999,
    "all_evidence_at_3": at3,
    "all_evidence_at_5": at5,
    "total": 18,
    "results": results
}

save_new(OUT / "v3_dev_regression.json", regression)

print("\n=== DEV V3 RESULTS ===")
print(f"All Evidence Hit@3: {at3}/18")
print(f"All Evidence Hit@5: {at5}/18")

if at5 < 18:
    raise SystemExit(
        "STOP: DEV regression decreased. Review before TEST."
    )

print("PASS: DEV V3 REGRESSION")

print("\n=== PREPARE TEST V3 ===")

source = OUT / "benchmark_test_30_UNVERIFIED.json"
questions = read_json(source)

assert len(questions) == 30

by_id = {r["id"]: r for r in records}
dev_gold = {
    g["record_id"]
    for q in dev
    for g in q["gold"]
}

# Replace two easier questions BEFORE TEST evaluation.
replacements = {
    "bonus_test_candidate_009": {
        "id": "bonus_test_candidate_009",
        "split": "test",
        "type": "chart",
        "question": (
            "According to the bar chart in Figure 1-6, "
            "what information does the chart visualize?"
        ),
        "expected_answer": (
            "The distribution of student letter grades, "
            "with bars showing the number of students "
            "in each grade category."
        ),
        "answerable": True,
        "gold": [{
            "record_id": "bonus_p63_chart_0",
            "source_id": "python_wang_2023",
            "pdf_page": 63,
            "kind": "chart"
        }],
        "human_verified": False
    },
    "bonus_test_candidate_015": {
        "id": "bonus_test_candidate_015",
        "split": "test",
        "type": "diagram",
        "question": (
            "In the while-loop flowchart in Figure 2-4, "
            "what happens when the condition is false?"
        ),
        "expected_answer": (
            "Control exits the while loop without "
            "executing its body again."
        ),
        "answerable": True,
        "gold": [{
            "record_id": "bonus_p180_diagram_0",
            "source_id": "python_wang_2023",
            "pdf_page": 180,
            "kind": "diagram"
        }],
        "human_verified": False
    }
}

updated = []
replaced_ids = set()

for q in questions:
    qid = q["id"]
    if qid in replacements:
        updated.append(replacements[qid])
        replaced_ids.add(qid)
    else:
        updated.append(q)

assert replaced_ids == set(replacements)
assert len(updated) == 30
assert len({q["id"] for q in updated}) == 30

answerable = [q for q in updated if q["answerable"]]
unanswerable = [q for q in updated if not q["answerable"]]

assert len(answerable) == 24
assert len(unanswerable) == 6

for q in answerable:
    assert q["gold"], f"Missing gold for {q['id']}"
    for g in q["gold"]:
        rid = g["record_id"]
        assert rid in by_id, f"Gold not indexed: {rid}"
        assert rid not in dev_gold, f"DEV evidence reused: {rid}"
        assert (
            by_id[rid]["pdf_page"] == g["pdf_page"]
        ), f"Page mismatch: {rid}"

type_counts = {}
for q in answerable:
    kind = q["type"]
    type_counts[kind] = type_counts.get(kind, 0) + 1

for kind in (
    "text", "table", "image", "chart", "diagram", "multi_hop"
):
    assert type_counts.get(kind, 0) >= 1, (
        f"Missing TEST category: {kind}"
    )

destination = OUT / "benchmark_test_30_v3_UNVERIFIED.json"
save_new(destination, updated)

print("TEST answerable:", len(answerable))
print("TEST unanswerable:", len(unanswerable))
print("TEST distribution:", type_counts)
print("All Gold IDs exist in V3: YES")
print("DEV evidence reused: NO")
print("Saved:", destination)
print("TEST scoring started: NO")
print("Human review required: YES")
print("PASS: TEST V3 BENCHMARK PREPARED")
