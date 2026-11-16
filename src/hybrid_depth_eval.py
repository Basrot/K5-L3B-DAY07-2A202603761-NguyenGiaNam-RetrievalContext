import os
import sys
import json
import time
import statistics
from pathlib import Path

import chromadb

from starter.baseline import TfidfIndex, hit, pack_context

DAY6 = Path(os.environ["DAY6_ROOT"])
sys.path.insert(0, str(DAY6 / "starter"))
from src.embeddings import MyEmbedder

chunks = json.loads(
    Path("starter/data/converted/chunks.json")
    .read_text(encoding="utf-8")
)
questions = json.loads(
    Path("starter/data/converted/questions.json")
    .read_text(encoding="utf-8")
)

dev = [q for q in questions if q["split"] == "dev"]
by_id = {c["id"]: c for c in chunks}

tfidf = TfidfIndex(chunks)
embedder = MyEmbedder("minilm")

client = chromadb.PersistentClient(
    path=str(DAY6 / "starter/data/chroma_db")
)
collection = client.get_collection("lab06_chunks")

depths = [3, 5, 10]
all_results = []

out = Path("starter/results/hybrid-experiments")
out.mkdir(parents=True, exist_ok=True)

for depth in depths:
    traces = []

    for q in dev:
        start = time.perf_counter()

        lexical = tfidf.search(q["question"], top_k=depth)

        vector = embedder.embed_query(q["question"])

        raw = collection.query(
            query_embeddings=[vector],
            n_results=depth,
            where={"strategy": "by_paragraph"},
            include=["distances"]
        )

        dense = []
        for cid in raw["ids"][0]:
            if cid not in by_id:
                raise ValueError(f"Unknown chunk: {cid}")
            dense.append(by_id[cid])

        scores = {}
        candidates = {}

        for source in (lexical, dense):
            for rank, chunk in enumerate(source, 1):
                cid = chunk["id"]
                candidates[cid] = chunk
                scores[cid] = (
                    scores.get(cid, 0.0)
                    + 1.0 / (60 + rank)
                )

        ranked = sorted(
            candidates.values(),
            key=lambda c: (-scores[c["id"]], c["id"])
        )

        results = []
        for c in ranked[:3]:
            item = c.copy()
            item["score"] = scores[c["id"]]
            item["strategy"] = "hybrid_rrf"
            item["hit"] = hit(item, q["gold"])
            results.append(item)

        context, ids = pack_context(results, budget=1600)
        elapsed = (time.perf_counter() - start) * 1000

        # Bằng chứng có xuất hiện trong tập ứng viên?
        candidate_hit = any(
            hit(c, q["gold"]) for c in ranked
        )

        trace = {
            "query_id": q["id"],
            "split": "dev",
            "question": q["question"],
            "strategy": "hybrid_rrf",
            "candidate_depth": depth,
            "top_k": 3,
            "budget_chars": 1600,
            "gold": q["gold"],
            "results": results,
            "context": context,
            "context_ids": ids,
            "context_chars": len(context),
            "elapsed_ms": elapsed,
            "candidate_hit": candidate_hit,
            "hit1": bool(results and results[0]["hit"]),
            "hit3": any(r["hit"] for r in results),
            "context_hit": any(
                r["hit"] and r["id"] in ids
                for r in results
            )
        }
        traces.append(trace)

    n = len(traces)

    summary = {
        "strategy": "hybrid_rrf",
        "split": "dev",
        "n": n,
        "candidate_depth": depth,
        "rrf_k": 60,
        "top_k": 3,
        "budget_chars": 1600,
        "candidate_hit": sum(t["candidate_hit"] for t in traces),
        "hit1": sum(t["hit1"] for t in traces),
        "hit3": sum(t["hit3"] for t in traces),
        "context_hit": sum(t["context_hit"] for t in traces),
        "avg_context_chars": round(
            statistics.mean(t["context_chars"] for t in traces), 2
        ),
        "median_ms": round(
            statistics.median(t["elapsed_ms"] for t in traces), 3
        ),
        "service_cost_usd": 0
    }

    folder = out / f"depth-{depth}"
    folder.mkdir(parents=True, exist_ok=True)

    for filename, data in [
        ("summary.json", summary),
        ("traces.json", traces)
    ]:
        (folder / filename).write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )

    all_results.append(summary)

    print(f"\n=== HYBRID DEPTH {depth} ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

(out / "comparison.json").write_text(
    json.dumps(all_results, ensure_ascii=False, indent=2),
    encoding="utf-8"
)

print("\nPASS: All hybrid experiments completed!")
