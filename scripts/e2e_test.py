"""Minimal end-to-end validation script for v1 pipeline."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from narrative_memory.config import AppConfig  # noqa: E402
from narrative_memory.pipeline import NarrativeMemoryPipeline  # noqa: E402


def main() -> None:
    tmp_root = PROJECT_ROOT / "data" / "tmp_e2e"
    raw_dir = tmp_root / "raw"
    sqlite_path = tmp_root / "metadata.db"
    chroma_dir = tmp_root / "chroma"

    if tmp_root.exists():
        shutil.rmtree(tmp_root)
    raw_dir.mkdir(parents=True, exist_ok=True)

    sample = raw_dir / "sample_2026-02-20.txt"
    sample.write_text(
        (
            "I remember the hallway clock freezing at 2:17.\n\n"
            "I said I'd never write about you again, then I wrote this line.\n\n"
            "Tomorrow I might call it healing, today it still sounds like regret."
        ),
        encoding="utf-8",
    )

    cfg = AppConfig.from_dict(
        {
            "paths": {
                "raw_dir": str(raw_dir),
                "sqlite_path": str(sqlite_path),
                "chroma_dir": str(chroma_dir),
            },
            "labeling": {
                "enabled": False
            },
            "retrieval": {
                "candidate_pool_size": 100,
                "default_top_k": 3,
                "vector_weight": 0.8,
                "emotion_weight": 0.1,
                "recency_weight": 0.1,
            },
        },
        base_dir=PROJECT_ROOT,
    )

    pipeline = NarrativeMemoryPipeline(cfg)
    stats = pipeline.ingest_and_index([str(sample)])
    assert stats["documents_ingested"] == 1, "Expected one ingested document."
    assert stats["chunks_created"] >= 1, "Expected at least one chunk."
    assert stats["sqlite_total_chunks"] >= 1, "SQLite should contain chunks."
    assert stats["chroma_total_vectors"] >= 1, "Chroma should contain vectors."

    hits = pipeline.retrieve(
        query_text="regret and memory in a specific place",
        metadata_filters={"time_scope": "past"},
        top_k=2,
    )
    assert len(hits) >= 1, "Expected at least one retrieval hit."

    draft = pipeline.draft_from_context(
        query_text="regret and memory",
        seed_text="The clock kept blinking but we never fixed it.",
        hits=hits,
    )
    assert draft["summary"], "Context summary should not be empty."
    assert draft["draft"], "Draft output should not be empty."

    print("E2E PASS")
    print(stats)
    print({"hits": len(hits)})


if __name__ == "__main__":
    main()
