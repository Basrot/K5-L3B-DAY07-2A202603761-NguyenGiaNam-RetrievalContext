import argparse
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BONUS = ROOT / "bonus"
OUT = BONUS / "output"
RUN_DIR = OUT / "v3_agent_first_run"
FROZEN = OUT / "benchmark_test_30_v3_FROZEN.json"
RETRIEVAL_RUN = OUT / "v3_test_retrieval_first_run.json"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(BONUS))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

# Keep these imports AFTER loading DAY6_ROOT and .env.
from run_gemini_agent import GeminiController
from hybrid_index_v3 import (
    load_records,
    get_collection,
    hybrid_search,
    MyEmbedder
)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def save_new_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)


# Exactly the rules used for the frozen TEST Retrieval run.
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


def run_case(q, records, by_id, collection, embedder):
    controller = GeminiController()
    history = []
    queries = []
    events = []
    evidence = {}
    started = time.perf_counter()

    answer = ""
    citations = []
    status = "protocol_error"
    problem = ""

    # Gemini controller permits a maximum of two completed searches.
    # One final answer decision is then allowed.
    for step in range(3):
        action = controller.decide(q["question"], history)
        events.append({
            "event": "model_action",
            "step": step + 1,
            "action": action
        })

        if action.get("action") == "search":
            if len(queries) >= 2:
                status = "search_budget_exceeded"
                problem = "Model requested more than two searches"
                break

            query = str(action.get("query") or "").strip()
            if not query or query in queries:
                status = "invalid_search"
                problem = "Empty or repeated search query"
                break

            # Frozen DEV routing policy, applied to the search query.
            route = infer_kind(query)

            hits = hybrid_search(
                question=query,
                records=records,
                collection=collection,
                embedder=embedder,
                kind=route,
                top_k=5,
                depth=10
            )

            queries.append(query)
            found_ids = []

            for hit in hits:
                rid = hit["id"]
                if rid not in by_id:
                    raise ValueError(f"Unknown evidence ID: {rid}")

                found_ids.append(rid)
                evidence[rid] = by_id[rid]

            # Context contains the COMPLETE indexed evidence text,
            # not the 220-character retrieval preview.
            context_parts = []
            for rid, record in evidence.items():
                context_parts.append(
                    f"[{rid}]\n"
                    f"PDF PAGE: {record['pdf_page']}\n"
                    f"KIND: {record['kind']}\n"
                    f"TEXT:\n{record['text']}"
                )

            context = "\n\n---\n\n".join(context_parts)

            events.append({
                "event": "search",
                "query": query,
                "route": route,
                "retrieved_ids": found_ids,
                "context_ids": list(evidence),
                "context_chars": len(context)
            })

            history.append({
                "queries": list(queries),
                "context": context,
                "context_ids": list(evidence)
            })

            continue

        if action.get("action") == "answer":
            answer = action.get("answer", "")
            citations = action.get("citations", [])

            if not isinstance(answer, str):
                status = "protocol_error"
                problem = "Answer is not a string"
                break

            if not isinstance(citations, list) or not all(
                isinstance(c, str) for c in citations
            ):
                status = "protocol_error"
                problem = "Invalid citation list"
                citations = []
                break

            if not queries:
                status = "protocol_error"
                problem = "Model answered without searching"
                break

            status = "answered" if answer.strip() else "abstained"
            break

        status = "protocol_error"
        problem = f"Unknown action: {action.get('action')}"
        break

    citation_ids_valid = all(
        rid in evidence for rid in citations
    )

    citation_markers_present = all(
        f"[{rid}]" in answer for rid in citations
    )

    citation_format_valid = (
        citation_ids_valid
        and citation_markers_present
        and (
            (status == "answered" and bool(citations))
            or (status == "abstained" and not citations)
        )
    )

    # Structural validation only. It does NOT establish whether
    # a cited passage supports the actual answer.
    return {
        "query_id": q["id"],
        "type": q["type"],
        "question": q["question"],
        "answerable_gold_label": q["answerable"],
        "expected_answer_for_manual_review": q.get("expected_answer"),
        "gold_record_ids_for_manual_review": [
            g["record_id"] for g in q.get("gold", [])
        ],
        "status": status,
        "problem": problem,
        "answer": answer,
        "citations": citations,
        "citation_ids_valid": citation_ids_valid,
        "citation_markers_present": citation_markers_present,
        "citation_format_valid": citation_format_valid,
        "searches": len(queries),
        "queries": queries,
        "evidence": [
            {
                "id": rid,
                "pdf_page": r["pdf_page"],
                "kind": r["kind"],
                "text": r["text"],
                "image_path": r.get("image_path")
            }
            for rid, r in evidence.items()
        ],
        "events": events,
        "model": getattr(controller, "actual_model", controller.model),
        "model_calls": controller.calls,
        "usage": controller.usage,
        "elapsed_ms": round(
            (time.perf_counter() - started) * 1000, 2
        )
    }


def export_review(questions):
    rows = []

    for q in questions:
        path = RUN_DIR / "cases" / f"{q['id']}.json"

        if not path.exists():
            continue

        r = read_json(path)

        rows.append({
            "query_id": q["id"],
            "type": q["type"],
            "gold_answerable": q["answerable"],
            "question": q["question"],
            "expected_answer": q.get("expected_answer", ""),
            "agent_status": r["status"],
            "agent_answer": r["answer"],
            "citations": "; ".join(r["citations"]),
            "retrieved_ids": "; ".join(
                item["id"] for item in r.get("evidence", [])
            ),
            "citation_format_valid": r["citation_format_valid"],
            "model": r["model"],
            "model_calls": r["model_calls"],
            "elapsed_ms": r["elapsed_ms"],
            "answer_correct_0_1_2": "",
            "citation_support_verified": "",
            "abstention_correct": "",
            "human_notes": ""
        })

    destination = RUN_DIR / "agent_human_review.csv"

    # Derived review worksheet; preserve any existing manual edits.
    if not destination.exists() and rows:
        with destination.open(
            "x", encoding="utf-8-sig", newline=""
        ) as f:
            writer = csv.DictWriter(
                f, fieldnames=list(rows[0].keys())
            )
            writer.writeheader()
            writer.writerows(rows)

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=["check", "run"], required=True
    )
    args = parser.parse_args()

    assert "DAY6_ROOT" in os.environ, (
        "Set $env:DAY6_ROOT before running"
    )
    assert FROZEN.exists(), "Frozen TEST benchmark missing"
    assert RETRIEVAL_RUN.exists(), "First retrieval run missing"

    benchmark_sha = hashlib.sha256(FROZEN.read_bytes()).hexdigest()
    first_run = read_json(RETRIEVAL_RUN)

    assert benchmark_sha == first_run["benchmark_sha256"], (
        "Benchmark SHA256 mismatch"
    )

    questions = read_json(FROZEN)
    assert len(questions) == 30
    assert sum(q["answerable"] for q in questions) == 24

    records = load_records()
    collection = get_collection()

    assert len(records) == 999
    assert collection.count() == 999

    print("=== BONUS V3 AGENT PREFLIGHT ===")
    print("Benchmark SHA256:", benchmark_sha)
    print("Questions:", len(questions))
    print("Indexed records:", collection.count())
    print("Model configured:", os.environ.get("GEMINI_MODEL"))
    print("API key configured:", bool(os.environ.get("GEMINI_API_KEY")))
    print("First Retrieval Hit@5:",
          first_run["all_evidence_at_5"], "/24")
    print("Routing: frozen DEV policy")
    print("Agent results folder:", RUN_DIR)

    if args.mode == "check":
        print("PASS: AGENT PREFLIGHT COMPLETED")
        print("Gemini calls made: 0")
        return

    assert os.environ.get("GEMINI_API_KEY"), "Missing Gemini API key"

    manifest = RUN_DIR / "manifest.json"

    if manifest.exists():
        previous = read_json(manifest)
        assert previous["benchmark_sha256"] == benchmark_sha, (
            "Existing run uses a different benchmark"
        )
    else:
        save_new_json(manifest, {
            "benchmark_sha256": benchmark_sha,
            "collection": "bonus_multimodal_v3",
            "records": 999,
            "top_k": 5,
            "depth": 10,
            "max_searches": 2,
            "routing": "frozen_dev_v3_rules",
            "evaluation_type": "first_agent_run",
            "human_absence_review": "provisional"
        })

    case_dir = RUN_DIR / "cases"
    case_dir.mkdir(parents=True, exist_ok=True)

    by_id = {r["id"]: r for r in records}
    embedder = MyEmbedder("minilm")

    for index, q in enumerate(questions, 1):
        destination = case_dir / f"{q['id']}.json"

        if destination.exists():
            print(
                f"[{index}/30] SKIP existing: {q['id']}",
                flush=True
            )
            continue

        print(
            f"\n[{index}/30] RUN: {q['id']}",
            flush=True
        )

        try:
            result = run_case(
                q, records, by_id, collection, embedder
            )
        except Exception as exc:
            result = {
                "query_id": q["id"],
                "type": q["type"],
                "question": q["question"],
                "status": "error",
                "problem": f"{type(exc).__name__}: {exc}",
                "answer": "",
                "citations": [],
                "citation_format_valid": False,
                "evidence": [],
                "model": "",
                "model_calls": None,
                "elapsed_ms": None
            }

        save_new_json(destination, result)

        print(
            "RESULT:", result["status"],
            "| citations:", result["citations"],
            flush=True
        )

    rows = export_review(questions)
    status_counts = {}

    for row in rows:
        status = row["agent_status"]
        status_counts[status] = status_counts.get(status, 0) + 1

    summary = {
        "benchmark_sha256": benchmark_sha,
        "total_cases": 30,
        "recorded_cases": len(rows),
        "status_counts": status_counts,
        "retrieval_first_run_hit_at_5": "16/24",
        "answer_accuracy": "pending_human_review",
        "citation_support": "pending_human_review",
        "abstention_accuracy": "provisional_pending_review",
        "results_directory": str(case_dir)
    }

    summary_file = RUN_DIR / "agent_run_summary.json"
    if not summary_file.exists():
        save_new_json(summary_file, summary)

    print("\n=== BONUS V3 AGENT FIRST RUN ===")
    print("Recorded:", len(rows), "/30")
    print("Status counts:", status_counts)
    print("Review CSV:", RUN_DIR / "agent_human_review.csv")
    print("Summary:", summary_file)
    print("NOTE: Human scoring is required.")


if __name__ == "__main__":
    main()
