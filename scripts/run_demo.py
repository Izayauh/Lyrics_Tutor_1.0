"""End-to-end demo: ingest -> chunk -> weak-label -> embed/store -> retrieve -> draft."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from narrative_memory.config import AppConfig  # noqa: E402
from narrative_memory.pipeline import NarrativeMemoryPipeline  # noqa: E402


def _ensure_demo_inputs(raw_dir: Path) -> list[str]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    note_path = raw_dir / "demo_notes_2026-02-25.txt"
    transcript_path = raw_dir / "demo_transcript_2026-02-26.txt"

    if not note_path.exists():
        note_path.write_text(
            (
                "I keep thinking about the old train platform near 7th street.\n\n"
                "Back then I said I'd leave by summer, but I stayed another year.\n\n"
                "I want the verse to sound honest, like a confession instead of a slogan."
            ),
            encoding="utf-8",
        )
    if not transcript_path.exists():
        transcript_path.write_text(
            (
                "Speaker 1: Today I finally felt calm enough to write.\n\n"
                "Speaker 1: Tomorrow I want to turn this memory into a chorus without cliches.\n\n"
                "Speaker 1: I still miss that station at midnight and the light on the railing."
            ),
            encoding="utf-8",
        )

    return [str(note_path), str(transcript_path)]


def main() -> None:
    config_path = PROJECT_ROOT / "config.yaml"
    config = AppConfig.from_yaml(str(config_path))
    pipeline = NarrativeMemoryPipeline(config)

    demo_inputs = _ensure_demo_inputs(Path(config.paths.raw_dir))
    stats = pipeline.ingest_and_index(demo_inputs)
    print("== Ingest/Index Stats ==")
    for key, value in stats.items():
        print(f"{key}: {value}")

    query = "nostalgic memory about a place and unfinished promises"
    hits = pipeline.retrieve(
        query_text=query,
        start_time="2026-01-01T00:00:00+00:00",
        end_time="2026-12-31T23:59:59+00:00",
        metadata_filters={"emotion": "nostalgia"},
        top_k=4,
    )

    print("\n== Retrieval Results ==")
    if not hits:
        print("No hits found.")
    else:
        for i, hit in enumerate(hits, start=1):
            preview = " ".join(hit["text"].split()[:30]).strip()
            print(
                f"{i}. score={hit['score']:.4f} sim={hit['similarity']:.4f} "
                f"emotion={hit['emotion']} time_scope={hit['time_scope']}\n   {preview}..."
            )

    draft = pipeline.draft_from_context(
        query_text=query,
        seed_text="I left my name on the station glass and called it closure.",
        hits=hits,
    )
    print("\n== Context Summary ==")
    print(draft["summary"])
    print("\n== Refined Draft ==")
    print(draft["draft"])


if __name__ == "__main__":
    main()
