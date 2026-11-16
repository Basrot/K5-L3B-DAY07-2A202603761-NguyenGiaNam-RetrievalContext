import csv
import json
import os
from pathlib import Path

import pymupdf
from PIL import Image, ImageDraw

root = Path.cwd()
out = root / "bonus/output"
package = json.loads(
    (out / "test_source_candidates.json").read_text(encoding="utf-8")
)

records = {
    item["id"]: item
    for line in (out / "multimodal_corpus.jsonl")
        .read_text(encoding="utf-8").splitlines()
    if line.strip()
    for item in [json.loads(line)]
}

# Make a source review worksheet with full extracted text.
rows = []

for item in package["answerable_source_candidates"]:
    r = records[item["record_id"]]

    rows.append({
        "record_id": r["id"],
        "kind": r["kind"],
        "pdf_page": r["pdf_page"],
        "full_source_text": r["text"],
        "usable_for_question": "",
        "proposed_question": "",
        "expected_answer": "",
        "review_notes": ""
    })

assert len(rows) == 24

csv_path = out / "test_source_review.csv"

if csv_path.exists():
    raise FileExistsError("Review CSV exists; not overwriting")

with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

# Render six visual candidate pages for human review.
day6 = Path(os.environ["DAY6_ROOT"])
pdf = day6 / "starter/data/python_textbook.pdf"

if not pdf.exists():
    pdf = root / "starter/data/python_textbook.pdf"

assert pdf.exists(), "PDF not found"

pages = [53, 212, 213, 214, 458, 498]
doc = pymupdf.open(pdf)

cols = 3
cell_w, cell_h = 460, 660
sheet = Image.new("RGB", (cols * cell_w, 2 * cell_h), "white")
draw = ImageDraw.Draw(sheet)

for i, page_no in enumerate(pages):
    page = doc[page_no - 1]
    pix = page.get_pixmap(matrix=pymupdf.Matrix(1.1, 1.1))
    image = Image.frombytes(
        "RGB", (pix.width, pix.height), pix.samples
    )
    image.thumbnail((430, 610))

    x = (i % cols) * cell_w + 15
    y = (i // cols) * cell_h + 35

    sheet.paste(image, (x, y))
    draw.text((x, y - 24), f"PDF PAGE {page_no}", fill="black")

doc.close()

image_path = out / "test_visual_candidates.jpg"

if image_path.exists():
    raise FileExistsError("Preview exists; not overwriting")

sheet.save(image_path, quality=92)

print("=== BONUS TEST REVIEW PACKAGE ===")
print("Source records:", len(rows))
print("Visual pages:", pages)
print("CSV:", csv_path)
print("Image:", image_path)
print("PASS: TEST review package ready")
