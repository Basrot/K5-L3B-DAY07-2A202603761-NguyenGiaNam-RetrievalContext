from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import os
import pymupdf

root = Path.cwd()
day6 = Path(os.environ["DAY6_ROOT"])
candidates = [
    day6 / "starter/data/python_textbook.pdf",
    root / "starter/data/python_textbook.pdf",
]
pdf = next((p for p in candidates if p.is_file()), None)
assert pdf is not None, "PDF not found"

out = root / "bonus/previews"
out.mkdir(parents=True, exist_ok=True)

groups = {
    "possible_charts": [
        56, 193, 469, 486, 488, 428, 180, 385
    ],
    "images": [
        17, 42, 46, 53, 212, 458, 480, 481
    ],
    "tables": [
        118, 131, 132, 259, 361, 459
    ],
}

doc = pymupdf.open(pdf)

thumb_w, thumb_h = 390, 550
cell_w, cell_h = 410, 595
cols = 4

for category, pages in groups.items():
    rows = (len(pages) + cols - 1) // cols

    sheet = Image.new(
        "RGB",
        (cols * cell_w, rows * cell_h),
        "white"
    )
    draw = ImageDraw.Draw(sheet)

    for pos, page_no in enumerate(pages):
        page = doc[page_no - 1]

        pix = page.get_pixmap(
            matrix=pymupdf.Matrix(0.9, 0.9),
            alpha=False
        )

        image = Image.frombytes(
            "RGB",
            [pix.width, pix.height],
            pix.samples
        )
        image.thumbnail((thumb_w, thumb_h))

        x = (pos % cols) * cell_w + 10
        y = (pos // cols) * cell_h + 35

        sheet.paste(image, (x, y))

        draw.text(
            (x, y - 25),
            f"PDF PAGE {page_no}",
            fill="black"
        )

        draw.rectangle(
            [x, y, x + image.width, y + image.height],
            outline="#777777",
            width=2
        )

    destination = out / f"{category}_contact_sheet.jpg"
    sheet.save(destination, quality=88)

    print("CREATED:", destination)

doc.close()
print("PASS: All contact sheets created")
