import json
from pathlib import Path
from collections import Counter

root = Path.cwd()
out = root / "bonus/output"

existing = json.loads(
    (out / "benchmark_dev_draft.json").read_text(encoding="utf-8")
)

records = {
    r["id"]: r
    for line in (out / "multimodal_corpus.jsonl")
        .read_text(encoding="utf-8").splitlines()
    if line.strip()
    for r in [json.loads(line)]
}

# New DEV candidates, based on source records already extracted.
new_cases = [
    (
        "text",
        "What are the main steps to calculate the sum of two numbers?",
        "Get two numbers, calculate their sum and print it.",
        ["bonus_p70_text_0"]
    ),
    (
        "text",
        "Which computational resources are mentioned when discussing algorithm complexity?",
        "CPU time and memory.",
        ["bonus_p25_text_1"]
    ),
    (
        "table",
        "According to the arithmetic type table, what is the result type of complex plus float?",
        "complex",
        ["bonus_p118_table_0"]
    ),
    (
        "table",
        "According to the division table, what is the result type of integer divided by integer?",
        "float",
        ["bonus_p118_table_1"]
    ),
    (
        "table",
        "Which Python operator joins two sequences?",
        "The plus operator (+).",
        ["bonus_p131_table_0"]
    )
]

assert len(existing) == 13
assert len(new_cases) == 5

draft = list(existing)

for number, (kind, question, answer, ids) in enumerate(
    new_cases, start=14
):
    assert all(rid in records for rid in ids)

    draft.append({
        "id": f"bonus_dev_{number:03}",
        "split": "dev",
        "type": kind,
        "question": question,
        "answerable": True,
        "expected_answer": answer,
        "gold": [
            {
                "record_id": rid,
                "source_id": records[rid]["source_id"],
                "pdf_page": records[rid]["pdf_page"],
                "kind": records[rid]["kind"]
            }
            for rid in ids
        ],
        "human_verified": False,
        "review_notes": "Pending source verification"
    })

assert len(draft) == 18
assert len({q["id"] for q in draft}) == 18

destination = out / "benchmark_dev_18_draft.json"
if destination.exists():
    raise FileExistsError(
        "18-question DEV draft already exists. Preserve it."
    )

destination.write_text(
    json.dumps(draft, ensure_ascii=False, indent=2),
    encoding="utf-8"
)

print("=== BONUS DEV 18 CANDIDATES ===")
print("Questions:", len(draft))
print("Types:", dict(Counter(q["type"] for q in draft)))
print("All evidence IDs exist: PASS")

print("\n=== NEW EVIDENCE TO VERIFY ===")
for q in draft[-5:]:
    print("\n", q["id"], q["question"])
    print("Expected:", q["expected_answer"])
    for gold in q["gold"]:
        record = records[gold["record_id"]]
        print("Record:", record["id"])
        print("Page:", record["pdf_page"])
        print("Excerpt:", ascii(record["text"][:500]))

print("\nSaved:", destination)
print("Human verified:", sum(q["human_verified"] for q in draft))
print("PASS: DEV 18 draft created")
