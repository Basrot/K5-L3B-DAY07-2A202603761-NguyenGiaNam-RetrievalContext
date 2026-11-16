
from pathlib import Path
import importlib.util
import os
import chromadb
from starter.baseline import hit, pack_context, validate

class Day6Retriever:
    def __init__(self):
        root = Path(os.environ["DAY6_ROOT"])

        # Load embedding model đã dùng ở Lab 06
        embedding_file = root / "starter/src/embeddings.py"
        spec = importlib.util.spec_from_file_location(
            "day6_embeddings", embedding_file
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.embedder = module.MyEmbedder("minilm")

        # Mở vector database Lab 06
        db_path = root / "starter/data/chroma_db"
        if not db_path.is_dir():
            raise FileNotFoundError(db_path)

        client = chromadb.PersistentClient(path=str(db_path))
        self.collection = client.get_collection("lab06_chunks")

    def search(self, query, *, filters=None, top_k=3):
        vector = self.embedder.embed_query(query)

        # Strategy B của Lab 06
        conditions = [{"strategy": "by_paragraph"}]
        for key, value in (filters or {}).items():
            conditions.append({key: value})

        where = (
            conditions[0] if len(conditions) == 1
            else {"$and": conditions}
        )

        results = self.collection.query(
            query_embeddings=[vector],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"]
        )

        hits = []
        for id_, text, meta, dist in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ):
            hits.append({
                "id": id_,
                "text": text,
                "source_id": meta["document_id"],
                "source": meta["source"],
                "page": meta["page"],
                "metadata": meta,
                "score": 1 - dist
            })

        return hits
