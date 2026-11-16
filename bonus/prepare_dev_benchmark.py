import json
import os
from pathlib import Path
import pymupdf4llm

root = Path.cwd()
day6 = Path(os.environ["DAY6_ROOT"])
pdf = day6 / "starter/data/python_textbook.pdf"
assert pdf.exists(), "PDF not found"

out = root / "bonus/output"
out.mkdir(parents=True, exist_ok=True)

print("=== TABLE EXTRACTION QUALITY ===", flush=True)

table_pages = [118, 131, 132, 259, 361, 459]
for page in table_pages:
    try:
        markdown = pymupdf4llm.to_markdown(
            str(pdf),
            pages=[page - 1]
        )
        path = out / f"table_page_{page}_markdown.md"
        path.write_text(markdown, encoding="utf-8")

        print(
            f"PAGE {page}:",
            len(markdown), "characters",
            "| markdown tables:",
            sum(line.startswith("|") for line in markdown.splitlines()),
            "| saved:", path.name,
            flush=True
        )
    except Exception as exc:
        print("TABLE ERROR:", page, repr(exc), flush=True)

records = {}
with (out / "multimodal_corpus.jsonl").open(encoding="utf-8") as f:
    for line in f:
        if line.strip():
            r = json.loads(line)
            records[r["id"]] = r

# Curated preliminary questions, not verified gold labels.
seeds = [
    ("image", "What traditional calculating device is pictured?",
     "A traditional abacus.", ["bonus_p17_image_0"]),
    ("image", "Which notebook interface is shown in the screenshot?",
     "Jupyter Notebook in a web browser.", ["bonus_p42_image_0"]),
    ("image", "Which code editor appears in the screenshot?",
     "Visual Studio Code.", ["bonus_p46_image_0"]),
    ("image", "What GUI library is illustrated by the application screenshot?",
     "Tkinter.", ["bonus_p480_image_0"]),
    ("image", "What kind of software interface is illustrated on page 481?",
     "A GUI application containing widgets.", ["bonus_p481_image_0"]),
    ("diagram", "What conditional structure is illustrated by the flowcharts?",
     "An if-elif-else conditional structure.", ["bonus_p193_diagram_0"]),
    ("chart", "What general trend is illustrated by Figure 8-6?",
     "Changes in world and regional population over time.",
     ["bonus_p469_chart_0"]),
    ("table", "Which sequence operator repeats a sequence multiple times?",
     "The multiplication operator (*).", ["bonus_p131_table_0"]),
    ("table", "What does Python's 'in' membership operator check?",
     "Whether a value occurs in a sequence.", ["bonus_p131_table_1"]),
    ("table", "What does the 'not in' operator check?",
     "Whether a value is absent from a sequence.", ["bonus_p132_table_0"]),
    ("table", "What result type does integer divided by integer produce in the division table?",
     "float.", ["bonus_p118_table_1"]),
    ("table", "What result type is shown for integer plus float?",
     "float.", ["bonus_p118_table_0"]),
    ("multi_hop", "How do 'in' and 'not in' differ in Python?",
     "'in' checks membership; 'not in' checks non-membership.",
     ["bonus_p131_table_1", "bonus_p132_table_0"]),
]

draft = []
for i, (kind, question, answer, ids) in enumerate(seeds, 1):
    missing = [rid for rid in ids if rid not in records]
    if missing:
        raise ValueError(f"Missing evidence for q{i}: {missing}")

    draft.append({
        "id": f"bonus_dev_{i:03}",
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
        "review_notes": "Pending evidence and answer verification"
    })

path = out / "benchmark_dev_draft.json"
if path.exists():
    raise FileExistsError("Draft already exists; not overwriting")

path.write_text(
    json.dumps(draft, ensure_ascii=False, indent=2),
    encoding="utf-8"
)

print("\n=== BONUS DEV BENCHMARK DRAFT ===")
print("Questions:", len(draft))
print("Evidence IDs:", sum(len(q["gold"]) for q in draft))
print("All source IDs exist: PASS")
print("Human verified: 0")
print("Saved:", path)
print("PASS: DRAFT READY")
