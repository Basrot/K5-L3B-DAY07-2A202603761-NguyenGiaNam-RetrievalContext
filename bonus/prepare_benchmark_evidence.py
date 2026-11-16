import json
from pathlib import Path
from collections import Counter

root = Path.cwd()
source = root / "bonus/output/multimodal_corpus.jsonl"
target = root / "bonus/output/benchmark_evidence_candidates.json"

records = [
    json.loads(line)
    for line in source.read_text(encoding="utf-8").splitlines()
    if line.strip()
]

assert len(records) == 994
by_kind = {}
for record in records:
    by_kind.setdefault(record["kind"], []).append(record)

# Page selection is preliminary, not benchmark labeling.
selected_pages = {
    "table": [118, 131, 132, 193, 259, 361, 458, 459],
    "image": [17, 42, 46, 480, 481],
    "diagram": [193],
    "chart": [469],
    "text": [
        25, 70, 95, 125, 150, 175,
        200, 225, 250, 275, 300, 325,
        350, 375, 400, 425, 450, 475
    ],
}

candidates = []

for kind, pages in selected_pages.items():
    for page in pages:
        matching = [
            r for r in by_kind.get(kind, [])
            if r["pdf_page"] == page
        ]

        for r in matching[:2]:
            candidates.append({
                "record_id": r["id"],
                "source_id": r["source_id"],
                "pdf_page": r["pdf_page"],
                "kind": kind,
                "evidence_text": r["text"][:1400],
                "bbox": r.get("bbox"),
                "image_path": r.get("image_path"),
                "human_verified": False
            })

target.write_text(
    json.dumps(candidates, ensure_ascii=False, indent=2),
    encoding="utf-8"
)

print("=== BONUS BENCHMARK EVIDENCE ===")
print("Total candidate records:", len(candidates))

for kind in ("text", "table", "image", "diagram", "chart"):
    subset = [r for r in candidates if r["kind"] == kind]
    print("\nKIND:", kind, "COUNT:", len(subset))

    for item in subset[:5]:
        print("ID:", item["record_id"])
        print("PAGE:", item["pdf_page"])
        print("TEXT:", ascii(item["evidence_text"][:240]))
        print("---")

print("\nSaved:", target)
print("PASS: Evidence candidates prepared")
print("NOTE: All candidates require human verification.")
