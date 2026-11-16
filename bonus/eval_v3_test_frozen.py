import sys
import json
import hashlib
import shutil
from pathlib import Path
from collections import defaultdict

ROOT = Path.cwd()
BONUS = ROOT / "bonus"
OUT = BONUS / "output"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(BONUS))

from hybrid_index_v3 import (
    load_records,
    get_collection,
    hybrid_search,
    MyEmbedder
)

source = OUT / "benchmark_test_30_v3_UNVERIFIED.json"
frozen = OUT / "benchmark_test_30_v3_FROZEN.json"
result_path = OUT / "v3_test_retrieval_first_run.json"

def load_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))

def save_new_json(path, data):
    with path.open("x", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def infer_kind(question):
    # Same fixed routing rules as DEV V3.
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

assert source.exists(), "TEST V3 benchmark missing"

questions = load_json(source)
answerable = [q for q in questions if q["answerable"]]
unanswerable = [q for q in questions if not q["answerable"]]

assert len(questions) == 30
assert len(answerable) == 24
assert len(unanswerable) == 6
assert len({q["id"] for q in questions}) == 30

# Do not overwrite the frozen benchmark or the first-run result.
assert not frozen.exists(), "Frozen TEST already exists"
assert not result_path.exists(), "First TEST result already exists"

records = load_records()
collection = get_collection()
embedder = MyEmbedder("minilm")

assert len(records) == 999
assert collection.count() == 999

record_ids = {r["id"] for r in records}

for q in answerable:
    assert q["gold"], f"Missing gold: {q['id']}"
    for item in q["gold"]:
        assert item["record_id"] in record_ids

# Freeze BEFORE running TEST.
shutil.copyfile(source, frozen)
sha = hashlib.sha256(frozen.read_bytes()).hexdigest()

print("=== BONUS TEST FROZEN ===")
print("Benchmark SHA256:", sha)
print("Answerable:", len(answerable))
print("Unanswerable:", len(unanswerable))
print("Index: bonus_multimodal_v3")
print("Records:", collection.count())
print("Retrieval Top K: 5")
print("Candidate depth: 10")
print("Routing: frozen DEV V3 rules")
print("Human absence audit: provisional")

results = []
by_type = defaultdict(list)

print("\n=== BONUS TEST RETRIEVAL FIRST RUN ===", flush=True)

for q in answerable:
    route = infer_kind(q["question"])
    gold = {g["record_id"] for g in q["gold"]}

    hits = hybrid_search(
        question=q["question"],
        records=records,
        collection=collection,
        embedder=embedder,
        kind=route,
        top_k=5,
        depth=10
    )

    ids = [h["id"] for h in hits]

    hit3 = gold.issubset(set(ids[:3]))
    hit5 = gold.issubset(set(ids[:5]))

    row = {
        "id": q["id"],
        "type": q["type"],
        "question": q["question"],
        "route": route,
        "gold": sorted(gold),
        "retrieved_top5": ids,
        "all_evidence_at_3": hit3,
        "all_evidence_at_5": hit5
    }

    results.append(row)
    by_type[q["type"]].append(row)

    print(
        q["id"],
        "|", q["type"],
        "| route:", route or "all",
        "| ALL@3:", int(hit3),
        "| ALL@5:", int(hit5),
        flush=True
    )

at3 = sum(r["all_evidence_at_3"] for r in results)
at5 = sum(r["all_evidence_at_5"] for r in results)

summary = {
    "split": "test",
    "run_type": "first_frozen_retrieval_evaluation",
    "index": "bonus_multimodal_v3",
    "corpus_records": 999,
    "benchmark_sha256": sha,
    "answerable": 24,
    "unanswerable_not_scored": 6,
    "top_k": 5,
    "candidate_depth": 10,
    "all_evidence_at_3": at3,
    "all_evidence_at_5": at5,
    "grounded_answer_scored": False,
    "citations_scored": False,
    "abstention_scored": False,
    "human_absence_audit_complete": False,
    "by_type": {
        kind: {
            "total": len(rows),
            "all_evidence_at_5": sum(
                r["all_evidence_at_5"] for r in rows
            )
        }
        for kind, rows in sorted(by_type.items())
    },
    "results": results
}

save_new_json(result_path, summary)

print("\n=== BONUS TEST RETRIEVAL SCORE ===")
print(f"All Evidence Hit@3: {at3}/24")
print(f"All Evidence Hit@5: {at5}/24")

for kind, item in summary["by_type"].items():
    print(
        f"{kind}: "
        f"{item['all_evidence_at_5']}/{item['total']}"
    )

print("\nUnanswerable: 6 (not yet graded)")
print("Agent scoring: NOT RUN")
print("Saved:", result_path)
print("PASS: FIRST FROZEN TEST RETRIEVAL RUN COMPLETED")
