"""Hybrid retrieval: metadata/time filters + vector similarity + weighted ranking."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

from dateutil import parser as dt_parser

from .embeddings import ChromaVectorStore
from .storage import SQLiteMetadataStore


@dataclass
class RetrievalWeights:
    """Ranking weights for hybrid retrieval."""

    vector: float = 0.75
    emotion: float = 0.15
    recency: float = 0.10


def _parse_iso(ts: Optional[str]) -> Optional[datetime]:
    if not ts:
        return None
    try:
        return dt_parser.parse(ts)
    except Exception:
        return None


def _recency_score(timestamp: Optional[str], min_ts: Optional[datetime], max_ts: Optional[datetime]) -> float:
    current = _parse_iso(timestamp)
    if current is None or min_ts is None or max_ts is None or min_ts >= max_ts:
        return 0.5
    span = (max_ts - min_ts).total_seconds()
    if span <= 0:
        return 0.5
    return max(0.0, min(1.0, (current - min_ts).total_seconds() / span))


class HybridRetriever:
    """Performs filtered candidate generation and weighted reranking."""

    def __init__(
        self,
        metadata_store: SQLiteMetadataStore,
        vector_store: ChromaVectorStore,
        candidate_pool_size: int = 300,
    ):
        self.metadata_store = metadata_store
        self.vector_store = vector_store
        self.candidate_pool_size = candidate_pool_size

    def retrieve(
        self,
        query_text: str,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        metadata_filters: Optional[Dict] = None,
        top_k: int = 5,
        weights: Optional[RetrievalWeights] = None,
    ) -> List[Dict]:
        """Return ranked chunks based on vector + metadata + recency signals."""
        weights = weights or RetrievalWeights()
        filters = metadata_filters or {}

        candidates = self.metadata_store.filter_chunks(
            start_time=start_time,
            end_time=end_time,
            filters=filters,
            limit=max(self.candidate_pool_size, top_k * 20),
        )
        if not candidates:
            return []

        candidate_ids = [row["id"] for row in candidates]
        candidate_map = {row["id"]: row for row in candidates}

        vector_hits = self.vector_store.query_subset(
            query_text=query_text,
            candidate_ids=candidate_ids,
            top_k=min(len(candidate_ids), max(top_k * 6, 30)),
        )
        if not vector_hits:
            vector_hits = self.vector_store.query_global(query_text, top_k=max(top_k * 2, 10))
            vector_hits = [hit for hit in vector_hits if hit["id"] in candidate_map]

        timestamps = [_parse_iso(row.get("timestamp")) for row in candidates if row.get("timestamp")]
        min_ts = min(timestamps) if timestamps else None
        max_ts = max(timestamps) if timestamps else None

        target_emotion = filters.get("emotion")
        ranked: List[Dict] = []
        for hit in vector_hits:
            row = candidate_map.get(hit["id"])
            if not row:
                continue
            emotion_bonus = 1.0 if target_emotion and row.get("emotion") == target_emotion else 0.0
            recency = _recency_score(row.get("timestamp"), min_ts, max_ts)
            final_score = (
                weights.vector * float(hit["similarity"])
                + weights.emotion * emotion_bonus
                + weights.recency * recency
            )
            ranked.append(
                {
                    "id": row["id"],
                    "score": round(final_score, 6),
                    "similarity": round(float(hit["similarity"]), 6),
                    "source": row["source"],
                    "timestamp": row.get("timestamp"),
                    "text": row["text"],
                    "emotion": row["emotion"],
                    "time_scope": row["time_scope"],
                    "intensity": row["intensity"],
                    "voice_mode": row["voice_mode"],
                    "authenticity_score": row["authenticity_score"],
                    "specificity_score": row["specificity_score"],
                    "cliche_score": row["cliche_score"],
                }
            )

        ranked.sort(key=lambda x: x["score"], reverse=True)
        return ranked[: max(1, top_k)]
