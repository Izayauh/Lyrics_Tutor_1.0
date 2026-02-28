"""Chroma vector store integration with swappable embedding model config."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import chromadb
import numpy as np
from chromadb.utils import embedding_functions

from .config import EmbeddingConfig
from .schemas import ChunkRecord


def _safe_float(value: float) -> float:
    if np.isnan(value) or np.isinf(value):
        return 0.0
    return float(value)


class ChromaVectorStore:
    """Persists chunk vectors in local Chroma."""

    def __init__(self, chroma_dir: str, config: EmbeddingConfig):
        Path(chroma_dir).mkdir(parents=True, exist_ok=True)
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=config.model_name
        )
        self.client = chromadb.PersistentClient(path=chroma_dir)
        self.collection = self.client.get_or_create_collection(
            name=config.collection_name,
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert_chunks(self, chunks: List[ChunkRecord]) -> None:
        if not chunks:
            return
        ids = [chunk.id for chunk in chunks]
        docs = [chunk.text for chunk in chunks]
        metadatas = [
            {
                "source": chunk.source,
                "timestamp": chunk.timestamp_iso() or "",
                "emotion": chunk.emotion,
                "time_scope": chunk.time_scope,
                "intensity": int(chunk.intensity),
                "voice_mode": chunk.voice_mode,
            }
            for chunk in chunks
        ]
        self.collection.upsert(ids=ids, documents=docs, metadatas=metadatas)

    def count(self) -> int:
        return self.collection.count()

    def query_subset(self, query_text: str, candidate_ids: List[str], top_k: int = 10) -> List[Dict]:
        """Similarity search constrained to a candidate ID subset."""
        if not candidate_ids:
            return []

        # Chroma ID lookups are practical for v1-size candidate pools.
        records = self.collection.get(
            ids=candidate_ids,
            include=["embeddings", "documents", "metadatas"],
        )
        # Chroma returns lists or NumPy arrays. Using 'is not None' avoids ambiguous truth value checks.
        ids = records.get("ids", [])
        embeddings = records.get("embeddings")
        if embeddings is None or len(embeddings) == 0:
            embeddings = []
        docs = records.get("documents")
        if docs is None:
            docs = []
        metas = records.get("metadatas")
        if metas is None:
            metas = []

        if not ids or len(embeddings) == 0:
            return []

        query_vec = np.array(self.embedding_function([query_text])[0], dtype=np.float32)
        qnorm = np.linalg.norm(query_vec)
        if qnorm == 0:
            return []

        scored: List[Dict] = []
        for idx, emb in enumerate(embeddings):
            emb_vec = np.array(emb, dtype=np.float32)
            denom = float(np.linalg.norm(emb_vec) * qnorm)
            sim = 0.0 if denom == 0 else float(np.dot(query_vec, emb_vec) / denom)
            scored.append(
                {
                    "id": ids[idx],
                    "similarity": _safe_float(sim),
                    "text": docs[idx] if idx < len(docs) else "",
                    "metadata": metas[idx] if idx < len(metas) else {},
                }
            )

        scored.sort(key=lambda x: x["similarity"], reverse=True)
        return scored[: max(1, top_k)]

    def query_global(self, query_text: str, top_k: int = 10) -> List[Dict]:
        """Similarity search over the whole collection."""
        result = self.collection.query(
            query_texts=[query_text],
            n_results=max(1, top_k),
            include=["documents", "metadatas", "distances"],
        )
        ids = (result.get("ids") or [[]])[0]
        docs = (result.get("documents") or [[]])[0]
        metas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]

        out: List[Dict] = []
        for i, chunk_id in enumerate(ids):
            distance = distances[i] if i < len(distances) else 1.0
            similarity = _safe_float(1.0 - float(distance))
            out.append(
                {
                    "id": chunk_id,
                    "similarity": similarity,
                    "text": docs[i] if i < len(docs) else "",
                    "metadata": metas[i] if i < len(metas) else {},
                }
            )
        return out
