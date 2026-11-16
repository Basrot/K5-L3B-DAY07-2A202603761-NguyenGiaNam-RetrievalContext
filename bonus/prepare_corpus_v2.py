import json
import os
from pathlib import Path
import pymupdf

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "bonus/output/multimodal_corpus.jsonl"
NEW = ROOT / "bonus/output/v2/multimodal_corpus.jsonl"
IMAGE_DIR = NEW.parent / "visual_pages"

assert OLD.exists(), "Original corpus missing"
assert not NEW.exists(), "V2 corpus already exists; do not overwrite"

records = [
    json.loads(line)
    for line in OLD.read_text(encoding="utf-8").splitlines()
    if line.strip()
]
assert len(records) == 994

visuals = {
    53: (
        "Athabasca University logo rendered in a Markdown image example."
    ),
    458: (
        "Figure 8-2 illustrates a blank Tkinter Canvas window."
    ),
    498: (
        "GUI examples with widgets and a login/register screen."
    )
}

day6 = Path(os.environ["DAY6_ROOT"])
pdf = day6 / "starter/data/python_textbook.pdf"
assert pdf.exists(), "Source PDF missing"

IMAGE_DIR.mkdir(parents=True, exist_ok=True)
doc = pymupdf.open(pdf)

existing = {r["id"] for r in records}

for page_no, description in visuals.items():
    rid = f"bonus_p{page_no}_image_0"
    assert rid not in existing, f"Duplicate ID: {rid}"

    page = doc[page_no - 1]
    filename = f"page_{page_no:03}.png"
    destination = IMAGE_DIR / filename

    pix = page.get_pixmap(
        matrix=pymupdf.Matrix(1.5, 1.5),
        alpha=False
    )
    pix.save(str(destination))

    records.append({
        "id": rid,
        "source_id": "python_wang_2023",
        "pdf_page": page_no,
        "kind": "image",
        "text": description,
        "bbox": list(page.rect),
        "image_path": f"visual_pages/{filename}",
        "caption_origin": "manually_reviewed_page_preview"
    })

    print("ADDED:", rid)

doc.close()

assert len(records) == 997
assert len({r["id"] for r in records}) == 997

NEW.parent.mkdir(parents=True, exist_ok=True)

with NEW.open("w", encoding="utf-8") as f:
    for record in records:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

# Make a separate V2 indexing script.
original = ROOT / "bonus/hybrid_index.py"
script = original.read_text(encoding="utf-8-sig")

replacements = {
    'bonus/output/multimodal_corpus.jsonl':
        'bonus/output/v2/multimodal_corpus.jsonl',
    'bonus/output/chroma_db':
        'bonus/output/v2/chroma_db',
    'bonus/output/index_fingerprint.json':
        'bonus/output/v2/index_fingerprint.json',
    'bonus_multimodal_v1':
        'bonus_multimodal_v2',
    'assert len(records) == 994':
        'assert len(records) == 997',
}

for old, new in replacements.items():
    assert old in script, f"Expected source not found: {old}"
    script = script.replace(old, new)

target = ROOT / "bonus/hybrid_index_v2.py"
assert not target.exists(), "V2 index script already exists"
target.write_text(script, encoding="utf-8")

print("\n=== BONUS CORPUS V2 ===")
print("Original records: 994")
print("Added visual records: 3")
print("Total records:", len(records))
print("V2 corpus:", NEW)
print("V2 index script:", target)
print("Original index preserved: YES")
print("PASS: BONUS CORPUS V2 READY")
