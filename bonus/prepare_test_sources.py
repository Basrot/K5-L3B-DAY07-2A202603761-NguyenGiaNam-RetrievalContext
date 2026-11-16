import json
import shutil
from pathlib import Path

ROOT = Path.cwd()
OUT = ROOT / "bonus/output"
REVIEWED = OUT / "benchmark_dev_18_reviewed.json"
CONFIG = OUT / "frozen_dev_config.json"
CANDIDATES = OUT / "test_source_candidates.json"

if CONFIG.exists() or CANDIDATES.exists():
    raise FileExistsError(
        "Frozen configuration or candidate file already exists."
    )

questions = json.loads(REVIEWED.read_text(encoding="utf-8-sig"))
assert len(questions) == 18

records = [
    json.loads(line)
    for line in (OUT / "multimodal_corpus.jsonl")
        .read_text(encoding="utf-8").splitlines()
    if line.strip()
]

dev_pages = {
    g["pdf_page"]
    for q in questions
    for g in q["gold"]
}

# Freeze the implementation already tested on DEV.
config = {
    "experiment": "bonus_multimodal_v1",
    "dev_benchmark": "benchmark_dev_18_reviewed.json",
    "collection": "bonus_multimodal_v1",
    "embedding": "minilm",
    "embedding_dimension": 384,
    "retrieval": "tfidf_dense_rrf",
    "rrf_k": 60,
    "candidate_depth_per_source": 10,
    "query_routing": "bonus/eval_dev_18.py:infer_kind",
    "top_k": 5,
    "dev_all_evidence_hit_at_5": "18/18",
    "dev_review": (
        "18 sources reviewed; q007 answer revised "
        "and requires final confirmation"
    ),
    "test_used_for_tuning": False,
    "frozen": True
}

CONFIG.write_text(
    json.dumps(config, indent=2),
    encoding="utf-8"
)

def pick(kind, count, min_length):
    eligible = [
        r for r in records
        if r["kind"] == kind
        and r["pdf_page"] not in dev_pages
        and len(r["text"]) >= min_length
        and r["pdf_page"] > 20
    ]

    eligible.sort(key=lambda r: r["pdf_page"])

    chosen = []
    used_pages = set()

    for r in eligible:
        if r["pdf_page"] in used_pages:
            continue
        if chosen and r["pdf_page"] - chosen[-1]["pdf_page"] < 8:
            continue

        chosen.append(r)
        used_pages.add(r["pdf_page"])

        if len(chosen) >= count:
            break

    return chosen

text_candidates = pick("text", 12, 500)
table_candidates = pick("table", 12, 100)

# Possible visual sources outside the DEV visual evidence.
# These pages must be inspected before being labeled.
visual_pages = [53, 212, 213, 214, 458, 498]

visual_candidates = []
for page in visual_pages:
    page_text = next(
        (r["text"][:650] for r in records
         if r["pdf_page"] == page and r["kind"] == "text"),
        ""
    )

    visual_candidates.append({
        "pdf_page": page,
        "status": "visual_review_required",
        "preview": page_text,
        "note": "Not yet indexed as verified visual evidence"
    })

package = {
    "dev_pages_excluded": sorted(dev_pages),
    "answerable_source_candidates": [
        {
            "record_id": r["id"],
            "pdf_page": r["pdf_page"],
            "kind": r["kind"],
            "text": r["text"][:1200]
        }
        for r in text_candidates + table_candidates
    ],
    "visual_source_candidates": visual_candidates,
    "test_questions_written": 0,
    "test_gold_verified": 0,
    "note": (
        "Source selection only; no TEST results or "
        "answer labels have been generated."
    )
}

CANDIDATES.write_text(
    json.dumps(package, ensure_ascii=False, indent=2),
    encoding="utf-8"
)

print("=== BONUS TEST PREPARATION ===")
print("DEV questions:", len(questions))
print("DEV evidence pages excluded:", len(dev_pages))
print("Text candidates:", len(text_candidates))
print("Table candidates:", len(table_candidates))
print("Visual pages needing review:", len(visual_candidates))
print("Frozen config:", CONFIG)
print("Candidate file:", CANDIDATES)

print("\n=== TEST SOURCE PREVIEW ===")
for r in package["answerable_source_candidates"]:
    print(
        r["kind"],
        "page", r["pdf_page"],
        r["record_id"],
        ascii(r["text"][:140])
    )

print("\nPASS: DEV CONFIG FROZEN AND TEST SOURCES PREPARED")
