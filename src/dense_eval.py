import os
import sys
import json
import time
import statistics
from pathlib import Path

import chromadb

ROOT = Path(os.environ["DAY6_ROOT"])
sys.path.insert(0, str(ROOT / "starter"))

from src.embeddings import MyEmbedder
from starter.baseline import hit, pack_context, validate

chunks = json.loads(
    Path("starter/data/converted/chunks.json")
    .read_text(encoding="utf-8")
)

questions = json.loads(
    Path("starter/data/converted/questions.json")
    .read_text(encoding="utf-8")
)

dev = [q for q in questions if q["split"] == "dev"]
validate(chunks, dev)

chunk_by_id = {c["id"]: c for c in chunks}

embedder = MyEmbedder("minilm")

client = chromadb.PersistentClient(
    path=str(ROOT / "starter/data/chroma_db")
)
collection = client.get_collection("lab06_chunks")

traces = []

for q in dev:
    start = time.perf_counter()

    vector = embedder.embed_query(q["question"])

    raw = collection.query(
        query_embeddings=[vector],
        n_results=3,
        where={"strategy": "by_paragraph"},
        include=["distances"]
    )

    results = []

    for cid, distance in zip(
        raw["ids"][0],
        raw["distances"][0]
    ):
        if cid not in chunk_by_id:
            raise ValueError(f"Unknown chunk: {cid}")

        c = chunk_by_id[cid].copy()
        c["score"] = -float(distance)
        c["strategy"] = "dense"
        c["hit"] = hit(c, q["gold"])
        results.append(c)

    context, ids = pack_context(results, budget=800)

    elapsed = (time.perf_counter() - start) * 1000

    traces.append({
        "query_id": q["id"],
        "split": "dev",
        "question": q["question"],
        "strategy": "dense",
        "top_k": 3,
        "budget_chars": 800,
        "gold": q["gold"],
        "results": results,
        "context": context,
        "context_ids": ids,
        "context_chars": len(context),
        "elapsed_ms": elapsed,
        "hit1": bool(results and results[0]["hit"]),
        "hit3": any(r["hit"] for r in results),
        "context_hit": any(
            r["hit"] and r["id"] in ids for r in results
        )
    })

    print(
        q["id"],
        "Hit@1:", traces[-1]["hit1"],
        "Hit@3:", traces[-1]["hit3"]
    )

summary = {
    "strategy": "dense",
    "split": "dev",
    "n": len(traces),
    "hit1": sum(t["hit1"] for t in traces),
    "hit3": sum(t["hit3"] for t in traces),
    "context_hit": sum(t["context_hit"] for t in traces),
    "avg_context_chars": statistics.mean(
        t["context_chars"] for t in traces
    ),
    "median_ms": statistics.median(
        t["elapsed_ms"] for t in traces
    ),
    "service_cost_usd": 0
}

out = Path("starter/results/dev-dense")
out.mkdir(parents=True, exist_ok=True)

for name, data in [
    ("summary.json", summary),
    ("traces.json", traces)
]:
    (out / name).write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

print("\n=== DENSE SUMMARY ===")
print(json.dumps(summary, ensure_ascii=False, indent=2))
