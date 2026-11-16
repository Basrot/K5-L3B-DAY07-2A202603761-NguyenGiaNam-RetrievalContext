import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BONUS = ROOT / "bonus"
OUT = BONUS / "output"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(BONUS))

# Importing does not execute V3 main().
import run_v3_agent_test as agent
import hybrid_index_v6 as index

DEV = OUT / "benchmark_dev_21_v4.json"
TEST = OUT / "benchmark_test_30_v6_FROZEN.json"

EXPECTED_TEST_SHA = (
    "8afea6f54074f6dadf2559e985f6db52d8d9ed33b48eb2bf400d256868bf52fc"
)

def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))

def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def save_new(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

# Preserve the DEV V6 routing policy.
# Routing is applied to the Agent's search query,
# as in the original V3 run_case.
def infer_kind(question):
    q = question.casefold()
    rules = [
        ("diagram", (
            "flowchart", "flow chart", "decision diagram"
        )),
        ("chart", (
            "chart", "graph", "figure 8-6",
            "population trend", "general trend",
            "square-root curve"
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
        )),
    ]
    for kind, patterns in rules:
        if any(p in q for p in patterns):
            return kind
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=["check", "dev", "test"],
        required=True
    )
    args = parser.parse_args()

    assert DEV.exists()
    assert TEST.exists()
    assert sha256(TEST) == EXPECTED_TEST_SHA, (
        "Frozen TEST was modified"
    )

    records = index.load_records()
    collection = index.get_collection()

    assert len(records) == 1006
    assert collection.count() == 1006

    # Redirect only these two V3 module functions.
    # Agent reasoning, citation logic, and budgets remain unchanged.
    agent.hybrid_search = index.hybrid_search
    agent.infer_kind = infer_kind

    questions_path = (
        TEST if args.mode == "test" else DEV
    )

    questions = read_json(questions_path)

    assert len(questions) == (
        30 if args.mode == "test" else 21
    )

    print("=== BONUS V6 AGENT PREFLIGHT ===")
    print("Mode:", args.mode)
    print("Corpus:", len(records))
    print("Collection:", collection.name)
    print("Questions:", len(questions))
    print("Benchmark SHA256:", sha256(questions_path))
    print("Gemini model:", os.environ.get("GEMINI_MODEL"))
    print(
        "Gemini key configured:",
        bool(os.environ.get("GEMINI_API_KEY"))
    )
    print("Routing: DEV V6 rules")
    print("TEST SHA256 verified:", True)

    if args.mode == "check":
        print("PASS: V6 AGENT PREFLIGHT")
        print("Gemini calls: 0")
        return

    assert os.environ.get("GEMINI_API_KEY"), (
        "Missing GEMINI_API_KEY"
    )

    run_dir = OUT / "v6" / (
        "agent_test_first_run"
        if args.mode == "test"
        else "agent_dev_first_run"
    )
    case_dir = run_dir / "cases"

    manifest_path = run_dir / "manifest.json"
    manifest = {
        "mode": args.mode,
        "benchmark": questions_path.name,
        "benchmark_sha256": sha256(questions_path),
        "collection": collection.name,
        "corpus_records": len(records),
        "model": os.environ.get("GEMINI_MODEL"),
        "max_searches": 2,
        "top_k": 5,
        "depth": 10,
        "routing": "dev_v6_query_routing",
        "evaluation": "first_run",
        "unanswerable_labels": "provisional"
    }

    if manifest_path.exists():
        assert read_json(manifest_path) == manifest, (
            "Existing run manifest differs"
        )
    else:
        save_new(manifest_path, manifest)

    case_dir.mkdir(parents=True, exist_ok=True)

    by_id = {r["id"]: r for r in records}
    embedder = index.MyEmbedder("minilm")

    for i, q in enumerate(questions, 1):
        path = case_dir / f"{q['id']}.json"

        if path.exists():
            print(
                f"[{i}/{len(questions)}] SKIP {q['id']}",
                flush=True
            )
            continue

        print(
            f"[{i}/{len(questions)}] RUN {q['id']}",
            flush=True
        )

        try:
            result = agent.run_case(
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
                "evidence": [],
            }

        save_new(path, result)

        print(
            "RESULT:",
            result.get("status"),
            "| citations:",
            result.get("citations"),
            flush=True
        )

    rows = []

    for q in questions:
        result = read_json(
            case_dir / f"{q['id']}.json"
        )

        rows.append({
            "id": q["id"],
            "type": q["type"],
            "answerable": q["answerable"],
            "question": q["question"],
            "expected_answer": q["expected_answer"],
            "agent_status": result.get("status"),
            "agent_answer": result.get("answer", ""),
            "citations": json.dumps(
                result.get("citations", []),
                ensure_ascii=False
            ),
            "gold_ids": json.dumps(
                [g["record_id"] for g in q["gold"]],
                ensure_ascii=False
            ),
            "evidence": json.dumps(
                result.get("evidence", []),
                ensure_ascii=False
            ),
            "answer_correct": "",
            "citation_supported": "",
            "human_notes": ""
        })

    csv_path = run_dir / "human_review.csv"

    if not csv_path.exists():
        with csv_path.open(
            "x", encoding="utf-8-sig", newline=""
        ) as f:
            writer = csv.DictWriter(
                f, fieldnames=list(rows[0].keys())
            )
            writer.writeheader()
            writer.writerows(rows)

    statuses = {}
    for row in rows:
        status = str(row["agent_status"])
        statuses[status] = statuses.get(status, 0) + 1

    summary_path = run_dir / "summary.json"

    if not summary_path.exists():
        save_new(summary_path, {
            "mode": args.mode,
            "total_cases": len(questions),
            "recorded_cases": len(rows),
            "status_counts": statuses,
            "answer_accuracy": "pending_human_review",
            "citation_support": "pending_human_review",
            "unanswerable_accuracy": "provisional",
            "benchmark_sha256": sha256(questions_path)
        })

    print("\n=== BONUS V6 AGENT SUMMARY ===")
    print("Mode:", args.mode)
    print("Recorded:", len(rows), "/", len(questions))
    print("Status counts:", statuses)
    print("Review CSV:", csv_path)
    print("Summary:", summary_path)
    print("PASS: AGENT RUN RECORDED")
    print("NOTE: Answer/citation scoring pending human review")

if __name__ == "__main__":
    main()
