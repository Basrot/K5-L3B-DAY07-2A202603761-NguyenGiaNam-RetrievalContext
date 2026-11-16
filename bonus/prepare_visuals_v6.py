import json
import os
from collections import Counter
from pathlib import Path

import pymupdf

root = Path.cwd()
bonus = root / "bonus"
out = bonus / "output"

old_corpus = out / "v5/multimodal_corpus.jsonl"
new_corpus = out / "v6/multimodal_corpus.jsonl"

old_test = out / "benchmark_test_30_v5_REVIEW_DRAFT.json"
new_test = out / "benchmark_test_30_v6_DRAFT.json"

old_index = bonus / "hybrid_index_v5.py"
new_index = bonus / "hybrid_index_v6.py"

pdf = (
    Path(os.environ["DAY6_ROOT"])
    / "starter/data/python_textbook.pdf"
)

for path in (old_corpus, old_test, old_index, pdf):
    assert path.exists(), f"Missing: {path}"

for path in (new_corpus, new_test, new_index):
    assert not path.exists(), f"Already exists: {path}"

records = [
    json.loads(line)
    for line in old_corpus.read_text(
        encoding="utf-8-sig"
    ).splitlines()
    if line.strip()
]

cases = json.loads(
    old_test.read_text(encoding="utf-8-sig")
)

assert len(records) == 1001
assert len(cases) == 30

# Five newly reviewed diagrams and their TEST questions.
items = [
    {
        "page": 19,
        "figure": "Figure 1-2",
        "question": (
            "In the Von Neumann architecture diagram, "
            "which two labeled units appear inside "
            "the Central Processing Unit?"
        ),
        "answer": (
            "Control Unit and Arithmetic/Logic Unit."
        ),
        "caption": (
            "Figure 1-2: Von Neumann architecture. "
            "The Central Processing Unit contains "
            "a Control Unit and an Arithmetic/Logic Unit. "
            "The diagram also shows Input, Output, "
            "Memory, and Storage with connecting arrows."
        )
    },
    {
        "page": 71,
        "figure": "Figure 1-7",
        "question": (
            "In Figure 1-7, what calculation takes "
            "place after Input n1 and Input n2, "
            "but before Print s?"
        ),
        "answer": "s = n1 + n2.",
        "caption": (
            "Figure 1-7: Flowchart of a simple algorithm. "
            "Start -> Input n1 -> Input n2 -> "
            "s = n1 + n2 -> Print s -> Stop."
        )
    },
    {
        "page": 72,
        "figure": "Figure 1-8",
        "question": (
            "In Figure 1-8, which condition is "
            "checked to decide whether the sum "
            "accumulation steps repeat?"
        ),
        "answer": "i <= 10000.",
        "caption": (
            "Figure 1-8: Summation algorithm flowchart. "
            "Initialize i = 1 and s = 0. Update "
            "s = s + i and i = i + 1. "
            "Check i <= 10000. When true, "
            "repeat the update; otherwise print s."
        )
    },
    {
        "page": 89,
        "figure": "Figure 2-1",
        "question": (
            "In Figure 2-1, what value is stored "
            "in the memory location referenced "
            "by the variable grade?"
        ),
        "answer": "98.",
        "caption": (
            "Figure 2-1: The variable grade and "
            "its memory location. An arrow from "
            "grade points to a memory box "
            "containing the value 98."
        )
    },
    {
        "page": 90,
        "figure": "Figure 2-3",
        "question": (
            "In Figure 2-3, after x changes its "
            "reference, which values do x and y "
            "refer to respectively?"
        ),
        "answer": "x refers to Smith; y refers to John.",
        "caption": (
            "Figure 2-3: Now X and Y point to "
            "different memory locations. "
            "X points to Smith and Y points "
            "to John."
        )
    }
]

existing = {r["id"] for r in records}
new_records = []

for item in items:
    page = item["page"]
    rid = f"bonus_p{page}_diagram_0"

    assert rid not in existing

    new_records.append({
        "id": rid,
        "source_id": "python_wang_2023",
        "pdf_page": page,
        "kind": "diagram",
        "text": item["caption"],
        "image_path": f"visual_pages/page_{page}.png",
        "caption_origin": "manual_pdf_visual_review"
    })

assert len(new_records) == 5

# Replace five old image/diagram cases.
old_visual = [
    q for q in cases
    if q["type"] in ("image", "diagram")
]

assert len(old_visual) == 5

for index, (case, item) in enumerate(
    zip(old_visual, items), 1
):
    page = item["page"]

    case["id"] = (
        f"bonus_v6_test_diagram_{index:02d}"
    )
    case["type"] = "diagram"
    case["question"] = item["question"]
    case["expected_answer"] = item["answer"]
    case["answerable"] = True

    case["gold"] = [{
        "record_id": f"bonus_p{page}_diagram_0",
        "source_id": "python_wang_2023",
        "pdf_page": page,
        "kind": "diagram"
    }]

    case["human_verified"] = False
    case["review_notes"] = (
        f"{item['figure']} visually inspected in PDF. "
        "Draft awaiting final independent review."
    )

combined = records + new_records
assert len(combined) == 1006
assert len({r["id"] for r in combined}) == 1006
assert len({q["id"] for q in cases}) == 30

counts = Counter(q["type"] for q in cases)

assert counts["table"] == 6
assert counts["diagram"] == 5
assert counts["chart"] == 5
assert counts["multi_hop"] == 4
assert counts["text"] == 4
assert counts["unanswerable"] == 6

# Check against all earlier DEV and V3 TEST Gold IDs.
dev = json.loads(
    (out / "benchmark_dev_21_v4.json")
    .read_text(encoding="utf-8-sig")
)
v3 = json.loads(
    (out / "benchmark_test_30_v3_FROZEN.json")
    .read_text(encoding="utf-8-sig")
)

old_gold = {
    g["record_id"]
    for q in dev + v3
    for g in q.get("gold", [])
}

for record in new_records:
    assert record["id"] not in old_gold

# Validate the index script before writing.
script = old_index.read_text(encoding="utf-8-sig")

assert "bonus/output/v5/" in script
assert "bonus_multimodal_v5" in script
assert "assert len(records) == 1001" in script

script = script.replace(
    "bonus/output/v5/",
    "bonus/output/v6/"
).replace(
    "bonus_multimodal_v5",
    "bonus_multimodal_v6"
).replace(
    "assert len(records) == 1001",
    "assert len(records) == 1006"
)

new_corpus.parent.mkdir(parents=True, exist_ok=True)
image_dir = new_corpus.parent / "visual_pages"
image_dir.mkdir(parents=True, exist_ok=True)

with pymupdf.open(pdf) as doc:
    for item in items:
        page_no = item["page"]
        pix = doc[page_no - 1].get_pixmap(
            matrix=pymupdf.Matrix(1.8, 1.8),
            alpha=False
        )
        pix.save(str(image_dir / f"page_{page_no}.png"))

with new_corpus.open("x", encoding="utf-8") as f:
    for record in combined:
        f.write(
            json.dumps(record, ensure_ascii=False) + "\n"
        )

with new_test.open("x", encoding="utf-8") as f:
    json.dump(cases, f, ensure_ascii=False, indent=2)

new_index.write_text(script, encoding="utf-8")

print("=== BONUS V6 PREPARATION ===")
print("Corpus:", len(combined))
print("New diagrams:", len(new_records))
print("TEST questions:", len(cases))
print("Types:", dict(counts))
print("Previous DEV/V3 Gold overlap for new diagrams: 0")
print("Corpus:", new_corpus)
print("Benchmark:", new_test)
print("PASS: BONUS V6 PREPARED")
print("NOTE: TEST IS STILL A DRAFT")
