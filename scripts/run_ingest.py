"""CLI entrypoint for ingesting and indexing local sources."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from narrative_memory.config import AppConfig  # noqa: E402
from narrative_memory.pipeline import NarrativeMemoryPipeline  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest and index narrative memory sources.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "config.yaml"),
        help="Path to YAML config.",
    )
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        help="File or directory to ingest (repeatable).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = AppConfig.from_yaml(args.config)
    pipeline = NarrativeMemoryPipeline(config)
    stats = pipeline.ingest_and_index(args.input)

    print("Ingestion complete.")
    for key, value in stats.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
