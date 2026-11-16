import csv, json
from pathlib import Path
from collections import Counter

root = Path("bonus/output/v6/agent_test_first_run")
source = root / "human_review.csv"
scored = root / "human_review_scored.csv"
summary = root / "human_scoring_summary.json"

assert source.exists()
assert not scored.exists()
assert not summary.exists()

wrong = {
    "bonus_v4_test_table_01",
    "bonus_v4_test_table_02",
    "bonus_v4_test_multi_hop_01",
    "bonus_v4_test_multi_hop_03",
}

citation_issue = {"bonus_v4_test_text_02"}

with source.open(encoding="utf-8-sig", newline="") as f:
    reader = csv.DictReader(f)
    fields = reader.fieldnames
    rows = list(reader)

assert len(rows) == 30

for row in rows:
    qid = row["id"]
    correct = qid not in wrong

    row["answer_correct"] = str(correct)

    if row["agent_status"] == "abstained":
        row["citation_supported"] = "N/A"
    else:
        row["citation_supported"] = str(
            correct and qid not in citation_issue
        )

    if row["type"] == "unanswerable":
        row["human_notes"] = (
            "Abstention correct under provisional "
            "unanswerable label."
        )
    elif qid in wrong:
        row["human_notes"] = (
            "Incorrect answer or false abstention."
        )
    elif qid in citation_issue:
        row["human_notes"] = (
            "Correct answer, but cited evidence "
            "does not directly support division."
        )
    else:
        row["human_notes"] = (
            "Answer and citation reviewed."
        )

with scored.open("x", encoding="utf-8-sig", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)

answerable = [r for r in rows if r["answerable"] == "True"]
unanswerable = [r for r in rows if r["answerable"] == "False"]
correct_answers = [
    r for r in answerable if r["answer_correct"] == "True"
]
supported = [
    r for r in correct_answers
    if r["citation_supported"] == "True"
]

result = {
    "test_cases": 30,
    "answerable_correct": len(correct_answers),
    "answerable_total": len(answerable),
    "answer_accuracy": len(correct_answers) / len(answerable),
    "provisional_unanswerable_correct": sum(
        r["agent_status"] == "abstained"
        for r in unanswerable
    ),
    "provisional_unanswerable_total": len(unanswerable),
    "correct_citation_support": len(supported),
    "correct_answer_citation_denominator": len(correct_answers),
    "total_correct_provisional": sum(
        r["answer_correct"] == "True" for r in rows
    ),
    "total": len(rows),
    "status_counts": dict(Counter(r["agent_status"] for r in rows)),
    "scoring_method": "manual_review_of_first_run",
    "unanswerable_labels": "provisional",
    "note": "Original first-run files preserved."
}

with summary.open("x", encoding="utf-8") as f:
    json.dump(result, f, indent=2, ensure_ascii=False)

print("=== BONUS V6 TEST SCORING ===")
print("Answer accuracy:", len(correct_answers), "/", len(answerable))
print("Citation supported:", len(supported), "/", len(correct_answers))
print("Provisional total correct:", result["total_correct_provisional"], "/30")
print("Saved:", scored)
print("Saved:", summary)
print("PASS: TEST SCORING RECORDED")
