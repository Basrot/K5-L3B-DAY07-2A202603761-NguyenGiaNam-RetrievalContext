import os
from pathlib import Path

import pymupdf
from PIL import Image, ImageDraw

root = Path.cwd()
pdf_path = (
    Path(os.environ["DAY6_ROOT"])
    / "starter/data/python_textbook.pdf"
)

out = root / "bonus/output/visual_review"
out.mkdir(parents=True, exist_ok=True)

pages = [
    63, 66, 71, 72,
    177, 178, 179, 180,
    192, 196, 207, 468
]

doc = pymupdf.open(pdf_path)

cols = 3
rows = 4
cell_w = 550
cell_h = 790

sheet = Image.new(
    "RGB",
    (cols * cell_w, rows * cell_h),
    "white"
)

draw = ImageDraw.Draw(sheet)

for i, page_no in enumerate(pages):
    page = doc[page_no - 1]

    pix = page.get_pixmap(
        matrix=pymupdf.Matrix(1.4, 1.4),
        alpha=False
    )

    img = Image.frombytes(
        "RGB",
        (pix.width, pix.height),
        pix.samples
    )

    full_path = out / f"page_{page_no:03}.png"
    img.save(full_path)

    img.thumbnail((520, 740))

    x = (i % cols) * cell_w + 15
    y = (i // cols) * cell_h + 35

    sheet.paste(img, (x, y))
    draw.text(
        (x, y - 25),
        f"PDF PAGE {page_no}",
        fill="black"
    )

    print(
        f"Rendered page {page_no}",
        flush=True
    )

doc.close()

destination = out / "visual_candidates_12.jpg"
sheet.save(destination, quality=94)

print("\n=== BONUS VISUAL REVIEW ===")
print("Pages rendered:", len(pages))
print("Saved:", destination)
print("PASS: VISUAL CONTACT SHEET READY")
