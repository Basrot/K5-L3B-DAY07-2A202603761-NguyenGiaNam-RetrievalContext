import json
import os
from pathlib import Path
import pymupdf

root = Path.cwd()
day6 = Path(os.environ["DAY6_ROOT"])
candidates = [
    day6 / "starter/data/python_textbook.pdf",
    root / "starter/data/python_textbook.pdf",
]
pdf = next((p for p in candidates if p.is_file()), None)
if pdf is None:
    raise FileNotFoundError("Source PDF not found")

doc = pymupdf.open(pdf)
records = []

for index, page in enumerate(doc):
    text = page.get_text("text")
    images = page.get_images(full=True)
    drawings = page.get_drawings()

    try:
        tables = page.find_tables().tables
        table_count = len(tables)
    except Exception:
        table_count = 0

    records.append({
        "pdf_page": index + 1,
        "text_chars": len(text.strip()),
        "images": len(images),
        "tables_detected": table_count,
        "vector_drawings": len(drawings),
        "text_preview": " ".join(text.split())[:160]
    })

out = root / "bonus/document_inventory.json"
out.write_text(
    json.dumps(records, ensure_ascii=False, indent=2),
    encoding="utf-8"
)

table_pages = sorted(
    (r for r in records if r["tables_detected"] > 0),
    key=lambda r: (-r["tables_detected"], r["pdf_page"])
)
image_pages = sorted(
    (r for r in records if r["images"] > 0),
    key=lambda r: (-r["images"], r["pdf_page"])
)
drawing_pages = sorted(
    (r for r in records if r["vector_drawings"] >= 12),
    key=lambda r: (-r["vector_drawings"], r["pdf_page"])
)

print("=== BONUS DOCUMENT AUDIT ===")
print("Total pages:", len(records))
print("Table candidate pages:", len(table_pages))
print("Image candidate pages:", len(image_pages))
print("Vector drawing candidate pages:", len(drawing_pages))

for label, items in [
    ("TABLE CANDIDATES", table_pages),
    ("IMAGE CANDIDATES", image_pages),
    ("VECTOR / POSSIBLE CHART CANDIDATES", drawing_pages)
]:
    print("\n===", label, "===")
    for r in items[:12]:
        print(
            "PDF page:", r["pdf_page"],
            "| tables:", r["tables_detected"],
            "| images:", r["images"],
            "| drawings:", r["vector_drawings"],
            "| preview:", ascii(r["text_preview"][:95])
        )

print("\nInventory:", out)
print("NOTE: Vector drawings are not automatically charts.")
print("PASS: Bonus source audit")
doc.close()
