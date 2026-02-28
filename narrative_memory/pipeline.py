"""Orchestration layer for ingesting, indexing, retrieving, and drafting."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional

from .chunking import SemanticChunker
from .config import AppConfig
from .embeddings import ChromaVectorStore
from .ingest import Ingestor
from .labeling import WeakLabeler
from .lyric_loop import LyricDraftLoop
from .retrieval import HybridRetriever, RetrievalWeights
from .storage import SQLiteMetadataStore


class NarrativeMemoryPipeline:
    """High-level pipeline composed of small local-first modules."""

    def __init__(self, config: AppConfig):
        self.config = config
        Path(config.paths.raw_dir).mkdir(parents=True, exist_ok=True)
        Path(config.paths.chroma_dir).mkdir(parents=True, exist_ok=True)
        Path(config.paths.sqlite_path).parent.mkdir(parents=True, exist_ok=True)

        self.ingestor = Ingestor()
        self.chunker = SemanticChunker(config.chunking)
        self.labeler = WeakLabeler(config.labeling, google_api_key=config.google_api_key)
        self.metadata_store = SQLiteMetadataStore(config.paths.sqlite_path)
        self.vector_store = ChromaVectorStore(config.paths.chroma_dir, config.embeddings)
        self.retriever = HybridRetriever(
            metadata_store=self.metadata_store,
            vector_store=self.vector_store,
            candidate_pool_size=config.retrieval.candidate_pool_size,
        )
        self.lyric_loop = LyricDraftLoop(
            model=config.labeling.model,
            google_api_key=config.google_api_key,
        )

    def ingest_and_index(self, input_paths: Iterable[str]) -> Dict[str, int]:
        """Run ingest -> semantic chunk -> weak label -> persist metadata/vector."""
        docs = self.ingestor.ingest_paths(input_paths)
        chunks = self.chunker.chunk_documents(docs)
        labeled = self.labeler.label_chunks(chunks)

        self.metadata_store.upsert_chunks(labeled)
        self.vector_store.upsert_chunks(labeled)

        return {
            "documents_ingested": len(docs),
            "chunks_created": len(chunks),
            "chunks_labeled": len(labeled),
            "sqlite_total_chunks": self.metadata_store.count(),
            "chroma_total_vectors": self.vector_store.count(),
        }

    def retrieve(
        self,
        query_text: str,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        metadata_filters: Optional[Dict] = None,
        top_k: Optional[int] = None,
    ) -> List[Dict]:
        """Run hybrid retrieval against indexed chunks."""
        weights = RetrievalWeights(
            vector=self.config.retrieval.vector_weight,
            emotion=self.config.retrieval.emotion_weight,
            recency=self.config.retrieval.recency_weight,
        )
        return self.retriever.retrieve(
            query_text=query_text,
            start_time=start_time,
            end_time=end_time,
            metadata_filters=metadata_filters,
            top_k=top_k or self.config.retrieval.default_top_k,
            weights=weights,
        )

    def draft_from_context(self, query_text: str, seed_text: str, hits: Optional[List[Dict]] = None) -> Dict[str, str]:
        """Summarize retrieval hits and return a refined lyric draft."""
        context_hits = hits if hits is not None else self.retrieve(query_text=query_text)
        summary = self.lyric_loop.summarize_context(context_hits)
        draft = self.lyric_loop.refine_draft(seed_text=seed_text, context_summary=summary)
        return {"summary": summary, "draft": draft}
