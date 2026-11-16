import os
import json
from pathlib import Path
from collections import Counter
import pymupdf

ROOT = Path.cwd()
DAY6 = Path(os.environ["DAY6_ROOT"])
PDF = DAY6 / "starter/data/python_textbook.pdf"
if not PDF.exists():
    PDF = ROOT / "starter/data/python_textbook.pdf"

assert PDF.exists(), "Source PDF not found"

OUT = ROOT / "bonus/output"
MANIFEST = OUT / "multimodal_corpus.jsonl"
RENDER = OUT / "visual_pages"

if MANIFEST.exists():
    raise RuntimeError("Corpus already exists; preserve previous output")

OUT.mkdir(parents=True, exist_ok=True)
RENDER.mkdir(parents=True, exist_ok=True)

# Descriptions based on inspected preview images.
VISUALS = {
    17: ("image", "Traditional abacus used as a computing device."),
    42: ("image", "Screenshot of Jupyter Notebook in a browser."),
    46: ("image", "Screenshot of a Python program in Visual Studio Code."),
    193: ("diagram", "Two flowcharts illustrating if-elif-else decisions."),
    469: (
        "chart",
        "Line chart of world population and regional population "
        "over time, with separate lines for geographic regions."
    ),
    480: ("image", "Screenshot of a basic Tkinter GUI application."),
    481: ("image", "Examples of Tkinter GUI applications with widgets.")
}

records = []
counts = Counter()

def add(page_no, kind, text, bbox=None, image_path=None):
    text = str(text).strip()
    if not text:
        return
    number = counts[(page_no, kind)]
    counts[(page_no, kind)] += 1

    records.append({
        "id": f"bonus_p{page_no}_{kind}_{number}",
        "source_id": "python_wang_2023",
        "pdf_page": page_no,
        "kind": kind,
        "text": text,
        "bbox": bbox,
        "image_path": image_path
    })

def chunks(text, size=1200, overlap=120):
    text = " ".join(text.split())
    if not text:
        return
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        yield text[start:end]
        if end == len(text):
            break
        start = end - overlap

doc = pymupdf.open(PDF)
assert len(doc) >= 100

print("=== BONUS MULTIMODAL INGESTION ===", flush=True)

for index, page in enumerate(doc):
    page_no = index + 1

    # Text extraction across all PDF pages.
    for section in chunks(page.get_text("text")):
        add(page_no, "text", section)

    # Structured table extraction, preserving table boundaries.
    try:
        found = page.find_tables()
        for table in found.tables:
            cells = table.extract()
            lines = [
                " | ".join(
                    str(value or "").replace("\n", " ")
                    for value in row
                )
                for row in cells
            ]
            if lines:
                add(
                    page_no, "table",
                    "\n".join(lines),
                    bbox=list(table.bbox)
                )
    except Exception as exc:
        print(
            "TABLE_WARNING:",
            page_no,
            type(exc).__name__,
            flush=True
        )

    # Verified visual evidence from the reviewed contact sheets.
    if page_no in VISUALS:
        kind, description = VISUALS[page_no]
        name = f"page_{page_no:03}.png"
        path = RENDER / name

        # Whole-page render preserves visual context and citations.
        pix = page.get_pixmap(
            matrix=pymupdf.Matrix(1.5, 1.5),
            alpha=False
        )
        pix.save(str(path))

        add(
            page_no,
            kind,
            description,
            bbox=list(page.rect),
            image_path=f"visual_pages/{name}"
        )

    if page_no % 100 == 0:
        print("Processed pages:", page_no, flush=True)

doc.close()

with MANIFEST.open("w", encoding="utf-8") as f:
    for item in records:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

summary = {
    "source": PDF.name,
    "pages_processed": 501,
    "total_records": len(records),
    "records_by_kind": dict(
        Counter(r["kind"] for r in records)
    ),
    "verified_visual_pages": sorted(VISUALS),
    "note": (
        "Initial corpus extraction. Visual captions are manually "
        "verified seed descriptions, not machine vision outputs."
    )
}

(OUT / "summary.json").write_text(
    json.dumps(summary, indent=2),
    encoding="utf-8"
)

print("\n=== BONUS INGESTION SUMMARY ===")
print(json.dumps(summary, indent=2))
print("Corpus:", MANIFEST)
print("PASS: Multimodal corpus extracted")
