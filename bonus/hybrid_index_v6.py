import argparse
import hashlib
import json
import os
from pathlib import Path

import chromadb
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from run_gemini_agent import MyEmbedder

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "bonus/output/v6/multimodal_corpus.jsonl"
DB = ROOT / "bonus/output/v6/chroma_db"
FINGERPRINT = ROOT / "bonus/output/v6/index_fingerprint.json"
COLLECTION = "bonus_multimodal_v6"

def load_records():
    records = [
        json.loads(line)
        for line in DATA.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(records) == 1006, f"Unexpected records: {len(records)}"
    assert len({r["id"] for r in records}) == len(records)
    return records

def get_collection():
    client = chromadb.PersistentClient(path=str(DB))
    return client.get_or_create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine"}
    )

def create_index():
    records = load_records()
    digest = hashlib.sha256(DATA.read_bytes()).hexdigest()

    if FINGERPRINT.exists():
        stored = json.loads(FINGERPRINT.read_text(encoding="utf-8"))
        if stored["sha256"] != digest:
            raise RuntimeError(
                "Corpus changed. Do not reuse the old index."
            )
    else:
        FINGERPRINT.parent.mkdir(parents=True, exist_ok=True)
        FINGERPRINT.write_text(json.dumps({
            "sha256": digest,
            "embedding": "minilm",
            "dimension": 384,
            "collection": COLLECTION
        }, indent=2), encoding="utf-8")

    collection = get_collection()
    existing = set(collection.get(include=[])["ids"])
    embedder = MyEmbedder("minilm")

    print("=== BONUS DENSE INDEX ===", flush=True)
    print("Records:", len(records), flush=True)
    print("Already indexed:", len(existing), flush=True)

    pending = [r for r in records if r["id"] not in existing]

    for start in range(0, len(pending), 32):
        batch = pending[start:start + 32]
        vectors = []

        for record in batch:
            vector = np.asarray(
                embedder.embed_query(record["text"]),
                dtype=np.float32
            ).reshape(-1)

            if len(vector) != 384:
                raise ValueError(
                    f"Expected 384 dimensions; got {len(vector)}"
                )

            vectors.append(vector.tolist())

        collection.upsert(
            ids=[r["id"] for r in batch],
            documents=[r["text"] for r in batch],
            embeddings=vectors,
            metadatas=[{
                "source_id": r["source_id"],
                "pdf_page": r["pdf_page"],
                "kind": r["kind"],
                "image_path": r.get("image_path") or ""
            } for r in batch]
        )

        print(
            "Indexed:",
            min(start + len(batch), len(pending)),
            "/",
            len(pending),
            flush=True
        )

    print("Collection:", collection.name)
    print("Total:", collection.count())
    assert collection.count() == len(records)
    print("PASS: BONUS INDEX READY")

def hybrid_search(question, records, collection, embedder,
                  top_k=5, depth=10, kind=None):
    filtered = [
        r for r in records
        if kind is None or r["kind"] == kind
    ]

    tfidf = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        token_pattern=r"(?u)\b\w+\b"
    )
    corpus_matrix = tfidf.fit_transform(
        [r["text"] for r in filtered]
    )
    query_vector = tfidf.transform([question])
    lexical_scores = (
        corpus_matrix @ query_vector.T
    ).toarray().reshape(-1)

    lexical_order = [
        int(i)
        for i in np.argsort(-lexical_scores)[:depth]
        if lexical_scores[int(i)] > 0
    ]
    lexical_ids = [filtered[i]["id"] for i in lexical_order]

    vector = np.asarray(
        embedder.embed_query(question),
        dtype=np.float32
    ).reshape(-1).tolist()

    query_args = {
        "query_embeddings": [vector],
        "n_results": min(depth, len(filtered))
    }
    if kind is not None:
        query_args["where"] = {"kind": kind}

    dense_result = collection.query(**query_args)
    dense_ids = dense_result["ids"][0]

    scores = {}
    for ranking in (lexical_ids, dense_ids):
        for rank, rid in enumerate(ranking, 1):
            scores[rid] = scores.get(rid, 0.0) + 1 / (60 + rank)

    by_id = {r["id"]: r for r in records}
    ranked = sorted(
        scores,
        key=lambda rid: (-scores[rid], rid)
    )[:top_k]

    return [
        {
            "id": rid,
            "pdf_page": by_id[rid]["pdf_page"],
            "kind": by_id[rid]["kind"],
            "score": round(scores[rid], 6),
            "preview": by_id[rid]["text"][:220],
            "image_path": by_id[rid].get("image_path")
        }
        for rid in ranked
    ]

def smoke_test():
    records = load_records()
    collection = get_collection()
    assert collection.count() == len(records), (
        "Index incomplete. Run build first."
    )
    embedder = MyEmbedder("minilm")

    tests = [
        {
            "question": "What does Figure 8-6 show about world population?",
            "expected_page": 469,
            "kind": None
        },
        {
            "question": "Python sequence operators in Table 2-13",
            "expected_page": 131,
            "kind": None
        },
        {
            "question": "What is the if elif else flowchart?",
            "expected_page": 193,
            "kind": None
        },
        {
            "question": "Screenshot of Jupyter Notebook in a browser",
            "expected_page": 42,
            "kind": None
        }
    ]

    results = []
    print("=== BONUS HYBRID RETRIEVAL SMOKE TEST ===")

    for test in tests:
        hits = hybrid_search(
            test["question"],
            records,
            collection,
            embedder,
            top_k=10
        )

        matched = [
            h for h in hits
            if h["pdf_page"] == test["expected_page"]
        ]

        print("\nQUESTION:", test["question"])
        print("EXPECTED PAGE:", test["expected_page"])
        print("PAGE HIT@10:", bool(matched))

        for rank, hit in enumerate(hits[:5], 1):
            print(
                rank,
                "page =", hit["pdf_page"],
                "kind =", hit["kind"],
                "id =", hit["id"]
            )

        results.append({
            **test,
            "page_hit_at_10": bool(matched),
            "hits": hits
        })

    out = ROOT / "bonus/output/smoke_results.json"
    out.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print("\nPAGE HITS:", sum(
        r["page_hit_at_10"] for r in results
    ), "/", len(results))
    print("Saved:", out)
    print("PASS: SMOKE TEST COMPLETED")
    print("NOTE: Page hits are diagnostics, not bonus scores.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action", choices=["build", "smoke"]
    )
    args = parser.parse_args()

    if args.action == "build":
        create_index()
    else:
        smoke_test()
