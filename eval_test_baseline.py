import json
import time
import traceback
from pathlib import Path

from run_gemini_agent import GeminiController
from starter.baseline import TfidfIndex
from agentic import run_agent

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "starter/results/agent-test-baseline"
QUESTIONS = ROOT / "starter/data/converted/questions_full.json"
CHUNKS = ROOT / "starter/data/converted/chunks.json"


class BaselineRetriever:
    def __init__(self):
        self.chunks = json.loads(CHUNKS.read_text(encoding="utf-8"))
        self.index = TfidfIndex(self.chunks)

    def search(self, query, *, filters=None, top_k=3):
        results = self.index.search(query, top_k=top_k)
        if filters:
            results = [
                c for c in results
                if all(c.get(k) == v for k, v in filters.items())
            ]
        return results


def main():
    if OUT.exists() and any(OUT.iterdir()):
        raise RuntimeError(
            "Baseline TEST output already exists. "
            "Preserve the previous run."
        )

    data = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    questions = data["questions"] if isinstance(data, dict) else data
    cases = [q for q in questions if q["split"] == "test"]

    assert len(cases) == 7
    assert sum(q["answerable"] for q in cases) == 5
    assert sum(not q["answerable"] for q in cases) == 2

    retriever = BaselineRetriever()
    OUT.mkdir(parents=True)

    config = {
        "split": "test",
        "retriever": "tfidf",
        "agent_loop": "starter.agentic.run_agent",
        "controller": "GeminiController",
        "top_k": 3,
        "budget": 800,
        "max_searches": 3,
        "note": "Baseline comparison; no TEST-driven tuning"
    }

    (OUT / "config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    traces = []

    print("=== AGENT BASELINE TEST ===", flush=True)

    for i, q in enumerate(cases, 1):
        qid = q["id"]
        print(f"\n[{i}/7] {qid}", flush=True)
        start = time.perf_counter()

        try:
            controller = GeminiController()

            result = run_agent(
                question=q["question"],
                controller=controller,
                retriever=retriever,
                top_k=3,
                budget=800,
                max_searches=3
            )

            result.update({
                "query_id": qid,
                "model": getattr(
                    controller, "actual_model", controller.model
                ),
                "model_calls": controller.calls,
                "usage": controller.usage,
                "elapsed_ms": round(
                    (time.perf_counter() - start) * 1000, 2
                )
            })

        except Exception as exc:
            result = {
                "query_id": qid,
                "status": "error",
                "answer": "",
                "citations": [],
                "events": [],
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "elapsed_ms": round(
                    (time.perf_counter() - start) * 1000, 2
                )
            }

        traces.append(result)

        (OUT / f"{qid}_trace.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )

        (OUT / "traces.json").write_text(
            json.dumps(traces, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )

        print("Status:", result["status"], flush=True)
        print("Citations:", len(result["citations"]), flush=True)
        print("Elapsed ms:", result["elapsed_ms"], flush=True)

    summary = {
        "split": "test",
        "cases": len(traces),
        "answered": sum(t["status"] == "answered" for t in traces),
        "abstain": sum(t["status"] == "abstain" for t in traces),
        "errors": sum(t["status"] == "error" for t in traces),
        "model_calls": sum(t.get("model_calls", 0) for t in traces),
        "avg_elapsed_ms": round(
            sum(t["elapsed_ms"] for t in traces) / len(traces), 2
        ),
        "note": "Human grounding and citation review still required"
    }

    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print("\n=== AGENT BASELINE TEST SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print("Saved:", OUT)


if __name__ == "__main__":
    main()
