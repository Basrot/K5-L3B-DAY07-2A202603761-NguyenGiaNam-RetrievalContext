import json
from pathlib import Path

root = Path.cwd()
bonus = root / "bonus"
out = bonus / "output"

candidates = sorted(
    bonus.glob("*24*answerable*.json")
) + sorted(
    out.glob("*24*answerable*.json")
)

print("=== BONUS TEST BENCHMARK VALIDATION ===")

if not candidates:
    print("ERROR: No 24-answerable draft found.")
    print("JSON files in bonus/:")
    for p in sorted(bonus.glob("*.json")):
        print(" ", p)
    raise SystemExit(1)

if len(candidates) != 1:
    print("Multiple candidates found:")
    for p in candidates:
        print(" ", p)
    raise SystemExit("Select a single draft before proceeding.")

source = candidates[0]
data = json.loads(source.read_text(encoding="utf-8-sig"))
questions = data["questions"] if isinstance(data, dict) else data

assert isinstance(questions, list)
assert len(questions) == 24, f"Expected 24, got {len(questions)}"

records = {
    r["id"]: r
    for line in (
        out / "v2/multimodal_corpus.jsonl"
    ).read_text(encoding="utf-8").splitlines()
    if line.strip()
    for r in [json.loads(line)]
}

dev = json.loads(
    (out / "benchmark_dev_18_reviewed.json")
    .read_text(encoding="utf-8-sig")
)

dev_questions = {q["question"].strip().casefold() for q in dev}
dev_gold = {
    g["record_id"] for q in dev for g in q["gold"]
}

errors = []
seen_ids = set()
answerable = []

for q in questions:
    qid = q.get("id", "<missing>")
    if qid in seen_ids:
        errors.append(f"{qid}: duplicate ID")
    seen_ids.add(qid)

    if not q.get("question") or not q.get("expected_answer"):
        errors.append(f"{qid}: missing question or expected_answer")

    if q.get("question", "").strip().casefold() in dev_questions:
        errors.append(f"{qid}: duplicates DEV question")

    if q.get("answerable") is not True:
        errors.append(f"{qid}: must be answerable")

    gold = q.get("gold", [])
    if not gold:
        errors.append(f"{qid}: missing Gold Evidence")
        continue

    for g in gold:
        rid = g.get("record_id")
        record = records.get(rid)
        if record is None:
            errors.append(f"{qid}: missing record {rid}")
            continue
        if g.get("pdf_page") != record["pdf_page"]:
            errors.append(f"{qid}: page mismatch for {rid}")
        if rid in dev_gold:
            errors.append(f"{qid}: Gold ID reused from DEV: {rid}")

    answerable.append(q)

print("Draft:", source)
print("TEST answerable:", len(answerable))
print("Indexed V2:", len(records))
print("Validation issues:", len(errors))

for problem in errors[:30]:
    print("ISSUE:", problem)

if errors:
    print("STOP: Fix draft issues before TEST scoring.")
    raise SystemExit(1)

# These six cases are provisional absence candidates.
missing_cases = [
    ("How is Python 3.13 free-threading configured to disable the GIL?",
     "free-threading, GIL, Python 3.13"),
    ("How is the experimental JIT enabled in Python 3.13?",
     "experimental JIT, Python 3.13"),
    ("What is the syntax for generic type parameters introduced in Python 3.12?",
     "generic type parameters, Python 3.12"),
    ("How does asyncio.TaskGroup aggregate simultaneous failures?",
     "asyncio.TaskGroup exception handling"),
    ("What changed in the Python 3.14 template string literal feature?",
     "template string literals, Python 3.14"),
    ("According to this textbook, what is the official CPython 3.14 release date?",
     "CPython 3.14 release date"),
]

unanswerable = []
for i, (question, topic) in enumerate(missing_cases, 1):
    unanswerable.append({
        "id": f"bonus_test_abs_{i:03}",
        "split": "test",
        "type": "unanswerable",
        "question": question,
        "answerable": False,
        "expected_behavior": "search_and_abstain",
        "gold": [],
        "absence_reason": (
            "Candidate for absence review: " + topic
        ),
        "human_verified": False
    })

for q in answerable:
    q["split"] = "test"

combined = answerable + unanswerable
assert len(combined) == 30
assert len({q["id"] for q in combined}) == 30

destination = out / "benchmark_test_30_UNVERIFIED.json"
if destination.exists():
    raise FileExistsError("TEST draft already exists; preserve it.")

destination.write_text(
    json.dumps(combined, ensure_ascii=False, indent=2),
    encoding="utf-8"
)

print("\n=== BONUS TEST 30 DRAFT ===")
print("Answerable:", 24)
print("Unanswerable candidates:", 6)
print("Total:", len(combined))
print("Gold IDs exist: PASS")
print("TEST scoring started: NO")
print("Human absence review: REQUIRED")
print("Saved:", destination)
print("PASS: TEST 30 DRAFT CREATED")
