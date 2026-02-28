"""Configuration loading for the Narrative Memory Engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def _resolve_path(raw_path: str, base_dir: Path) -> str:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return str(path)


@dataclass
class PathsConfig:
    """Filesystem locations used by the pipeline."""

    raw_dir: str = "data/raw"
    sqlite_path: str = "data/metadata.db"
    chroma_dir: str = "data/chroma"


@dataclass
class ChunkingConfig:
    """Semantic chunking boundaries."""

    min_words: int = 80
    max_words: int = 300
    hard_max_words: int = 380


@dataclass
class LabelingConfig:
    """Weak labeling settings."""

    enabled: bool = True
    provider: str = "google"
    model: str = "gemini-2.5-flash"
    batch_size: int = 12
    fallback_heuristic: bool = True


@dataclass
class EmbeddingConfig:
    """Embedding and Chroma settings."""

    model_name: str = "all-MiniLM-L6-v2"
    collection_name: str = "narrative_chunks"


@dataclass
class RetrievalConfig:
    """Hybrid retrieval defaults and ranking weights."""

    candidate_pool_size: int = 300
    default_top_k: int = 5
    vector_weight: float = 0.75
    emotion_weight: float = 0.15
    recency_weight: float = 0.10


@dataclass
class AppConfig:
    """Top-level app configuration."""

    paths: PathsConfig = field(default_factory=PathsConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    labeling: LabelingConfig = field(default_factory=LabelingConfig)
    embeddings: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    google_api_key: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any], base_dir: Optional[Path] = None) -> "AppConfig":
        """Build config from a dictionary."""
        base = Path.cwd() if base_dir is None else base_dir

        paths_data = data.get("paths", {})
        paths = PathsConfig(
            raw_dir=_resolve_path(paths_data.get("raw_dir", "data/raw"), base),
            sqlite_path=_resolve_path(paths_data.get("sqlite_path", "data/metadata.db"), base),
            chroma_dir=_resolve_path(paths_data.get("chroma_dir", "data/chroma"), base),
        )

        chunking = ChunkingConfig(**data.get("chunking", {}))
        labeling = LabelingConfig(**data.get("labeling", {}))
        embeddings = EmbeddingConfig(**data.get("embeddings", {}))
        retrieval = RetrievalConfig(**data.get("retrieval", {}))

        return cls(
            paths=paths,
            chunking=chunking,
            labeling=labeling,
            embeddings=embeddings,
            retrieval=retrieval,
            google_api_key=data.get("google_api_key"),
        )

    @classmethod
    def from_yaml(cls, path: str) -> "AppConfig":
        """Load config from YAML."""
        config_path = Path(path).resolve()
        with config_path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        return cls.from_dict(data, base_dir=config_path.parent)
