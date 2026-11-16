import json
import time
import traceback
from pathlib import Path

import run_gemini_agent as agent
from src.seed_first_agent import run_agent_seedfirst

BASE = Path.cwd()
benchmark_path = BASE / "starter/data/converted/questions_full.json"
out = BASE / "starter/results/agent-test-seedfirst-final"

data = json.loads(benchmark_path.read_text(encoding="utf-8"))
questions = data["questions"] if isinstance(data, dict) else data
cases = [q for q in questions if q["split"] == "test"]

assert len(cases) == 7
assert sum(q["answerable"] for q in cases) == 5
assert sum(not q["answerable"] for q in cases) == 2
assert len({q["id"] for q in cases}) == 7

out.mkdir(parents=True, exist_ok=True)

# Prevent accidental overwriting of the holdout run.
if list(out.glob("q*_trace.json")):
    raise RuntimeError(
        "TEST traces already exist. Preserve the first run; "
        "do not overwrite or rerun automatically."
    )

config = {
    "split": "test",
    "top_k": 3,
    "max_searches": 3,
    "budget": 2400,
    "context_engineering": "seed_first_neighbor_expansion",
    "retrieval": "hybrid_rrf",
    "note": "Frozen DEV configuration; first TEST evaluation"
}

(out / "config.json").write_text(
    json.dumps(config, indent=2), encoding="utf-8"
)

traces = []

print("=== FINAL TEST HOLDOUT ===", flush=True)
print("Questions:", len(cases), flush=True)

for index, q in enumerate(cases, 1):
    qid = q["id"]
    print(
        f"\n[{index}/7] {qid}: {q['question']}",
        flush=True
    )

    start = time.perf_counter()

    try:
        controller = agent.GeminiController()
        retriever = agent.HybridRetriever()

        result = run_agent_seedfirst(
            question=q["question"],
            controller=controller,
            retriever=retriever,
            max_searches=3,
            top_k=3,
            budget=2400
        )

        result["query_id"] = qid
        result["model"] = controller.model
        result["model_calls"] = controller.calls
        result["usage"] = controller.usage
        result["elapsed_ms"] = round(
            (time.perf_counter() - start) * 1000, 2
        )

        assert result["status"] in ("answered", "abstain")

        for event in result["events"]:
            if event.get("event") == "search":
                assert len(event["context"]) <= 2400

        if result["status"] == "answered":
            searches = [
                e for e in result["events"]
                if e.get("event") == "search"
            ]
            assert searches
            assert result["citations"]
            assert all(
                cid in searches[-1]["context_ids"]
                for cid in result["citations"]
            )

    except Exception as exc:
        result = {
            "query_id": qid,
            "status": "error",
            "answer": "",
            "citations": [],
            "events": [],
            "error": str(exc),
            "error_type": type(exc).__name__,
            "traceback": traceback.format_exc(),
            "elapsed_ms": round(
                (time.perf_counter() - start) * 1000, 2
            )
        }

    traces.append(result)

    (out / f"{qid}_trace.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print("Status:", result["status"], flush=True)
    print("Citations:", len(result["citations"]), flush=True)

    # Persist progress even if a later model call fails.
    (out / "traces.json").write_text(
        json.dumps(traces, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

summary = {
    "split": "test",
    "total": len(traces),
    "answered": sum(t["status"] == "answered" for t in traces),
    "abstain": sum(t["status"] == "abstain" for t in traces),
    "errors": sum(t["status"] == "error" for t in traces),
    "note": "Behavioral results, pending human review"
}

(out / "summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2),
    encoding="utf-8"
)

print("\n=== TEST SUMMARY ===")
print(json.dumps(summary, indent=2))
print("Saved:", out)
