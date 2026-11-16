import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BONUS = ROOT / "bonus"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(BONUS))

from hybrid_index_v6 import (
    load_records,
    get_collection,
    hybrid_search,
    MyEmbedder,
)

OUT = BONUS / "output"
BENCHMARK = OUT / "benchmark_dev_21_v4.json"
RESULT = OUT / "v6/dev_retrieval_first_run.json"

assert BENCHMARK.exists(), "Missing DEV V4 benchmark"
assert not RESULT.exists(), "DEV V4 result already exists"

questions = json.loads(
    BENCHMARK.read_text(encoding="utf-8-sig")
)

records = load_records()
by_id = {r["id"]: r for r in records}

assert len(questions) == 21
assert len(records) == 1006
assert len({q["id"] for q in questions}) == 21

answerable = [q for q in questions if q["answerable"]]
unanswerable = [q for q in questions if not q["answerable"]]

assert len(answerable) == 18
assert len(unanswerable) == 3

for q in answerable:
    assert q["gold"], q["id"]
    for gold in q["gold"]:
        assert gold["record_id"] in by_id, q["id"]

for q in unanswerable:
    assert q["gold"] == [], q["id"]


def infer_kind(question):
    q = question.casefold()

    rules = [
        (
            "diagram",
            (
                "flowchart",
                "flow chart",
                "decision diagram",
            ),
        ),
        (
            "chart",
            (
                "chart",
                "graph",
                "figure 8-6",
                "population trend",
                "general trend",
                "square-root curve",
            ),
        ),
        (
            "image",
            (
                "screenshot",
                "pictured",
                "picture",
                "visual interface",
                "gui library",
                "software interface",
                "notebook interface",
                "code editor",
            ),
        ),
        (
            "table",
            (
                "operator",
                "membership",
                "sequence",
                "result type",
                "integer divided",
                "integer plus",
                "'in' and 'not in'",
                "division table",
                "arithmetic type table",
            ),
        ),
    ]

    for kind, patterns in rules:
        if any(pattern in q for pattern in patterns):
            return kind

    return None


def evaluate_hits(hits, gold):
    ids = [h["id"] for h in hits]

    return {
        "any_hit_at_3": bool(gold & set(ids[:3])),
        "all_hit_at_3": gold <= set(ids[:3]),
        "all_hit_at_5": gold <= set(ids[:5]),
        "retrieved_ids": ids,
    }


collection = get_collection()

assert collection.count() == len(records), (
    f"Index count mismatch: {collection.count()}"
)

embedder = MyEmbedder("minilm")

rows = []
print("=== BONUS V6 DEV RETRIEVAL ===", flush=True)
print("Corpus:", len(records), flush=True)
print("Answerable:", len(answerable), flush=True)
print("Unanswerable:", len(unanswerable), flush=True)

for question in questions:
    gold = {
        g["record_id"]
        for g in question["gold"]
    }

    route = infer_kind(question["question"])

    original_hits = hybrid_search(
        question["question"],
        records,
        collection,
        embedder,
        top_k=5,
        depth=10,
    )

    routed_hits = hybrid_search(
        question["question"],
        records,
        collection,
        embedder,
        top_k=5,
        depth=10,
        kind=route,
    )

    row = {
        "query_id": question["id"],
        "type": question["type"],
        "answerable": question["answerable"],
        "question": question["question"],
        "route": route,
        "gold_ids": sorted(gold),
        "original_hits": original_hits,
        "routed_hits": routed_hits,
    }

    if question["answerable"]:
        row["original"] = evaluate_hits(
            original_hits, gold
        )
        row["routed"] = evaluate_hits(
            routed_hits, gold
        )
        print(
            question["id"],
            "| type:", question["type"],
            "| route:", route or "all",
            "| original H5:",
            int(row["original"]["all_hit_at_5"]),
            "| routed H5:",
            int(row["routed"]["all_hit_at_5"]),
            flush=True,
        )
    else:
        row["retrieval_score"] = None
        row["note"] = (
            "Unanswerable: retrieval hits are not scored. "
            "Agent abstention must be evaluated separately."
        )
        print(
            question["id"],
            "| UNANSWERABLE",
            "| route:", route or "all",
            "| no retrieval hit score",
            flush=True,
        )

    rows.append(row)


def summarize(method):
    scored = [
        r for r in rows if r["answerable"]
    ]

    return {
        "denominator": len(scored),
        "any_hit_at_3": sum(
            int(r[method]["any_hit_at_3"])
            for r in scored
        ),
        "all_hit_at_3": sum(
            int(r[method]["all_hit_at_3"])
            for r in scored
        ),
        "all_hit_at_5": sum(
            int(r[method]["all_hit_at_5"])
            for r in scored
        ),
    }


by_type = {}

for kind in sorted({q["type"] for q in answerable}):
    subset = [
        r for r in rows
        if r["answerable"] and r["type"] == kind
    ]

    by_type[kind] = {
        "total": len(subset),
        "original_all_hit_at_5": sum(
            int(r["original"]["all_hit_at_5"])
            for r in subset
        ),
        "routed_all_hit_at_5": sum(
            int(r["routed"]["all_hit_at_5"])
            for r in subset
        ),
    }

summary = {
    "split": "dev",
    "version": "v6",
    "corpus_records": len(records),
    "benchmark_cases": len(questions),
    "answerable_cases": len(answerable),
    "unanswerable_cases": len(unanswerable),
    "original": summarize("original"),
    "routed": summarize("routed"),
    "by_type": by_type,
    "human_review_status": (
        "18 answerable sources reviewed; "
        "3 unanswerable labels provisional"
    ),
    "note": (
        "DEV-only retrieval evaluation. "
        "Unanswerables excluded from Hit denominators. "
        "TEST V3 remains frozen."
    ),
}

RESULT.parent.mkdir(parents=True, exist_ok=True)

with RESULT.open("x", encoding="utf-8") as f:
    json.dump(
        {"summary": summary, "traces": rows},
        f,
        ensure_ascii=False,
        indent=2,
    )

print("\n=== DEV V4 FINAL SUMMARY ===")
print("Original:", summary["original"])
print("Routed:", summary["routed"])

print("\n=== BY TYPE ===")
for kind, metrics in by_type.items():
    print(kind, metrics)

print("\n=== UNANSWERABLE ===")
print(
    len(unanswerable),
    "cases excluded from retrieval scoring"
)

print("\nSaved:", RESULT)
print("PASS: DEV V4 RETRIEVAL EVALUATED")
