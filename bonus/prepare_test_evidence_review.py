import csv
import json
import re
from pathlib import Path

ROOT = Path.cwd()
OUT = ROOT / "bonus/output"
BENCHMARK = OUT / "benchmark_test_30_UNVERIFIED.json"
CORPUS = OUT / "v2/multimodal_corpus.jsonl"

questions = json.loads(BENCHMARK.read_text(encoding="utf-8-sig"))
records = {
    r["id"]: r
    for line in CORPUS.read_text(encoding="utf-8").splitlines()
    if line.strip()
    for r in [json.loads(line)]
}

assert len(questions) == 30
assert len(records) == 997

terms = {
    1: ["free-threading", "free threading", "no-gil",
        "gil", "global interpreter lock", "python 3.13"],
    2: ["experimental jit", "just-in-time", "jit compiler",
        "python 3.13"],
    3: ["type parameters", "type parameter", "pep 695",
        "python 3.12"],
    4: ["asyncio.taskgroup", "taskgroup", "exceptiongroup",
        "exception group"],
    5: ["template string literal", "template strings",
        "python 3.14", "t-string"],
    6: ["cpython 3.14", "python 3.14 release",
        "3.14 release date"],
}

answer_rows = []
absence_rows = []

for q in questions:
    qid = q["id"]

    if q["answerable"]:
        evidence = []
        for g in q["gold"]:
            rid = g["record_id"]
            r = records[rid]
            evidence.append({
                "id": rid,
                "page": r["pdf_page"],
                "kind": r["kind"],
                "text": r["text"],
                "image_path": r.get("image_path")
            })

        answer_rows.append({
            "query_id": qid,
            "type": q.get("type", ""),
            "question": q["question"],
            "expected_answer": q["expected_answer"],
            "gold_ids": "; ".join(e["id"] for e in evidence),
            "gold_pages": "; ".join(
                str(e["page"]) for e in evidence
            ),
            "source_evidence": json.dumps(
                evidence, ensure_ascii=False
            ),
            "source_verified": "",
            "answer_verified": "",
            "review_reason": ""
        })

    else:
        number = int(qid.rsplit("_", 1)[-1])
        keywords = terms[number]
        matches = []

        for r in records.values():
            content = r["text"].casefold()
            found = [
                term for term in keywords
                if re.search(
                    r"(?<!\w)" + re.escape(term) + r"(?!\w)",
                    content
                )
            ]
            if found:
                matches.append({
                    "id": r["id"],
                    "page": r["pdf_page"],
                    "terms": found,
                    "excerpt": r["text"][:400]
                })

        absence_rows.append({
            "query_id": qid,
            "question": q["question"],
            "matched_records": len(matches),
            "keyword_matches": json.dumps(
                matches[:15], ensure_ascii=False
            ),
            "absence_verified": "",
            "review_reason": ""
        })

assert len(answer_rows) == 24
assert len(absence_rows) == 6

def save_csv(path, rows):
    if path.exists():
        raise FileExistsError(
            f"File already exists: {path}"
        )
    with path.open(
        "w", newline="", encoding="utf-8-sig"
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=list(rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(rows)

answer_file = OUT / "test_24_gold_review.csv"
absence_file = OUT / "test_6_absence_review.csv"

save_csv(answer_file, answer_rows)
save_csv(absence_file, absence_rows)

print("=== BONUS TEST FINAL EVIDENCE AUDIT ===")
print("Corpus records:", len(records))
print("Answerable review:", len(answer_rows))
print("Unanswerable review:", len(absence_rows))

for r in absence_rows:
    print(
        r["query_id"],
        "keyword matching records:",
        r["matched_records"]
    )

print("\nSaved:", answer_file)
print("Saved:", absence_file)
print("TEST retrieval/agent scoring started: NO")
print("PASS: TEST EVIDENCE REVIEW PACKAGE READY")
