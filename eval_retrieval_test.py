import json
import time
from pathlib import Path

from starter.baseline import TfidfIndex, hit, pack_context
from run_gemini_agent import HybridRetriever
from src.seed_first_agent import load_chunks, pack_seed_first

ROOT = Path("starter")
OUT = ROOT / "results/retrieval-test-final"
OUTPUT = OUT / "comparison.json"

if OUTPUT.exists():
    raise RuntimeError(
        "TEST retrieval result already exists. "
        "Preserve the original run."
    )

data = json.loads(
    (ROOT / "data/converted/questions_full.json")
    .read_text(encoding="utf-8")
)
questions = data["questions"] if isinstance(data, dict) else data

test = [
    q for q in questions
    if q["split"] == "test" and q["answerable"]
]
assert len(test) == 5

chunks_by_id = load_chunks()
chunks = list(chunks_by_id.values())

tfidf = TfidfIndex(chunks)
hybrid = HybridRetriever()

configs = [
    ("baseline_tfidf", 800),
    ("hybrid_seedfirst", 2400),
]

def evaluate_result(results, context, ids, gold):
    marked = [
        bool(hit(chunks_by_id[r["id"]], gold))
        for r in results
    ]

    return {
        "hit1": bool(marked and marked[0]),
        "hit3": any(marked),
        "context_hit": any(
            hit(chunks_by_id[cid], gold)
            for cid in ids
        ),
    }

all_traces = []
summaries = {}

print("=== RETRIEVAL-ONLY TEST ===", flush=True)
print("Answerable questions:", len(test), flush=True)

for name, budget in configs:
    records = []
    print("\nCONFIG:", name, flush=True)

    for q in test:
        start = time.perf_counter()

        if name == "baseline_tfidf":
            results = tfidf.search(
                q["question"], top_k=3
            )
            context, ids = pack_context(
                results, budget=budget
            )
        else:
            results = hybrid.search(
                q["question"],
                filters=None,
                top_k=3
            )
            context, ids = pack_seed_first(
                results,
                chunks_by_id,
                budget=budget
            )

        elapsed = (time.perf_counter() - start) * 1000

        assert len(results) <= 3
        assert len(context) <= budget
        assert len(set(ids)) == len(ids)
        assert all(cid in chunks_by_id for cid in ids)

        metric = evaluate_result(
            results, context, ids, q["gold"]
        )

        row = {
            "query_id": q["id"],
            "strategy": name,
            "split": "test",
            "budget_chars": budget,
            "results": results,
            "context_ids": ids,
            "context": context,
            "context_chars": len(context),
            "elapsed_ms": round(elapsed, 3),
            **metric
        }

        records.append(row)
        all_traces.append(row)

        print(
            q["id"],
            "Hit@1:", int(metric["hit1"]),
            "Hit@3:", int(metric["hit3"]),
            "Context:", int(metric["context_hit"]),
            flush=True
        )

    n = len(records)

    summaries[name] = {
        "n": n,
        "hit1": sum(r["hit1"] for r in records),
        "hit3": sum(r["hit3"] for r in records),
        "context_hit": sum(r["context_hit"] for r in records),
        "avg_latency_ms": round(
            sum(r["elapsed_ms"] for r in records) / n, 3
        ),
        "avg_context_chars": round(
            sum(r["context_chars"] for r in records) / n, 2
        ),
        "budget_chars": budget
    }

OUT.mkdir(parents=True, exist_ok=True)

(OUT / "traces.json").write_text(
    json.dumps(all_traces, ensure_ascii=False, indent=2),
    encoding="utf-8"
)

OUTPUT.write_text(
    json.dumps(summaries, ensure_ascii=False, indent=2),
    encoding="utf-8"
)

print("\n=== FINAL RETRIEVAL TEST COMPARISON ===")
for name, s in summaries.items():
    print("\n", name)
    print("Hit@1:", f"{s['hit1']}/{s['n']}")
    print("Hit@3:", f"{s['hit3']}/{s['n']}")
    print("Context Hit:", f"{s['context_hit']}/{s['n']}")
    print("Avg latency (ms):", s["avg_latency_ms"])
    print("Avg context chars:", s["avg_context_chars"])

print("\nSaved:", OUTPUT)
print("PASS: Retrieval-only TEST completed")
