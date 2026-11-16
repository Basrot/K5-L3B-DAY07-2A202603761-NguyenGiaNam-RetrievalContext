import csv
import json
from pathlib import Path

root = Path.cwd()
folder = root / "bonus/output"

questions = json.loads(
    (folder / "benchmark_dev_18_v2.json").read_text(encoding="utf-8-sig")
)
evaluation = json.loads(
    (folder / "dev_18_comparison.json").read_text(encoding="utf-8-sig")
)

records = {}
for line in (folder / "multimodal_corpus.jsonl").read_text(
    encoding="utf-8-sig"
).splitlines():
    if line.strip():
        r = json.loads(line)
        records[r["id"]] = r

traces = {t["query_id"]: t for t in evaluation["traces"]}
assert len(questions) == 18
assert len(traces) == 18

rows = []
for q in questions:
    trace = traces[q["id"]]
    gold_ids = [g["record_id"] for g in q["gold"]]
    retrieved = trace["routed"]["ids"]

    evidence = []
    for rid in gold_ids:
        r = records[rid]
        evidence.append(
            "ID: " + rid +
            "\nPAGE: " + str(r["pdf_page"]) +
            "\nKIND: " + r["kind"] +
            "\nTEXT:\n" + r["text"][:1800]
        )

    rows.append({
        "query_id": q["id"],
        "type": q["type"],
        "question": q["question"],
        "expected_answer": q["expected_answer"],
        "gold_record_ids": "; ".join(gold_ids),
        "gold_pages": "; ".join(
            str(records[rid]["pdf_page"]) for rid in gold_ids
        ),
        "source_evidence": "\n\n".join(evidence),
        "retrieval_top5": "; ".join(retrieved[:5]),
        "all_gold_at_5": trace["routed"]["all_at_5"],
        "source_verified": "",
        "answer_verified": "",
        "review_reason": "",
    })

out = folder / "dev_18_human_review.csv"
if out.exists():
    raise FileExistsError("Review file already exists; not overwriting")

with out.open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

print("=== BONUS DEV HUMAN REVIEW ===")
print("Questions:", len(rows))
print("Gold IDs:", sum(len(q["gold"]) for q in questions))
print("All Gold @5:", sum(
    r["all_gold_at_5"] for r in rows
), "/", len(rows))
print("Source verified: 0 / 18")
print("Answer verified: 0 / 18")
print("Saved:", out)

print("\nVISUAL AND MULTI-HOP CASES TO CHECK:")
for r in rows:
    if r["type"] in ("image", "diagram", "chart", "multi_hop"):
        print(r["query_id"], r["type"], r["gold_pages"])

print("\nPASS: Review worksheet prepared")
