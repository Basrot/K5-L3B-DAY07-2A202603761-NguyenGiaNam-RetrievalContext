import json
import hashlib
from pathlib import Path

OUT = Path("bonus/output")
result_path = OUT / "v3_test_retrieval_first_run.json"
benchmark_path = OUT / "benchmark_test_30_v3_FROZEN.json"
corpus_path = OUT / "v3/multimodal_corpus.jsonl"
destination = OUT / "v3_test_failure_audit.txt"

assert not destination.exists(), "Audit file already exists"

run = json.loads(result_path.read_text(encoding="utf-8"))
benchmark = json.loads(
    benchmark_path.read_text(encoding="utf-8-sig")
)

sha = hashlib.sha256(benchmark_path.read_bytes()).hexdigest()
assert sha == run["benchmark_sha256"], "Frozen benchmark changed!"

records = {}
with corpus_path.open(encoding="utf-8") as f:
    for line in f:
        if line.strip():
            r = json.loads(line)
            records[r["id"]] = r

questions = {q["id"]: q for q in benchmark}
failed = [
    r for r in run["results"]
    if not r["all_evidence_at_5"]
]

lines = [
    "BONUS V3 - FROZEN TEST FAILURE AUDIT",
    f"Benchmark SHA256: {sha}",
    f"Failures: {len(failed)}/24",
    "Analysis only. No rerun or benchmark modification.",
    ""
]

for item in failed:
    q = questions[item["id"]]
    gold = set(item["gold"])
    hits = item["retrieved_top5"]

    lines.extend([
        "=" * 70,
        f"ID: {item['id']}",
        f"TYPE: {item['type']}",
        f"ROUTE: {item['route']}",
        f"QUESTION: {q['question']}",
        f"EXPECTED: {q['expected_answer']}",
        f"GOLD: {', '.join(sorted(gold))}",
        "",
        "TOP 5 RETRIEVED:"
    ])

    for i, rid in enumerate(hits, 1):
        r = records[rid]
        lines.append(
            f"{i}. {rid} | {r['kind']} | "
            f"PDF page {r['pdf_page']}"
        )

    lines.append("\nMISSING GOLD EVIDENCE:")

    for rid in sorted(gold - set(hits)):
        r = records[rid]
        lines.extend([
            f"ID: {rid}",
            f"KIND: {r['kind']}",
            f"PDF PAGE: {r['pdf_page']}",
            f"TEXT: {r['text'][:1800]}",
            ""
        ])

    if item["route"] is not None:
        incompatible = [
            rid for rid in gold
            if records[rid]["kind"] != item["route"]
        ]
        if incompatible:
            lines.append(
                "DIAGNOSIS: Gold kind incompatible "
                "with the chosen route."
            )
        else:
            lines.append(
                "DIAGNOSIS: Gold has compatible kind; "
                "ranking or content may be responsible."
            )
    else:
        lines.append(
            "DIAGNOSIS: No kind filter; inspect "
            "ranking and extracted evidence."
        )

    lines.append("")

destination.write_text(
    "\n".join(lines),
    encoding="utf-8"
)

print("=== BONUS TEST FAILURE AUDIT ===")
print("Frozen benchmark integrity: PASS")
print("Failed questions:", len(failed))
print("Saved:", destination)

for r in failed:
    print(r["id"], "| route:", r["route"] or "all")

print("PASS: FAILURE AUDIT CREATED")
