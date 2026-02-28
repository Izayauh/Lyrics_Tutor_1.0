"""Core data structures shared across modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional


@dataclass
class RawDocument:
    """Normalized text input before chunking."""

    source: str
    text: str
    timestamp: Optional[datetime] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ChunkRecord:
    """Chunk record persisted into SQLite and Chroma."""

    id: str
    source: str
    text: str
    timestamp: Optional[datetime] = None
    emotion: str = "unknown"
    time_scope: str = "unknown"
    intensity: int = 3
    voice_mode: str = "unknown"
    authenticity_score: int = 3
    specificity_score: int = 3
    cliche_score: int = 3
    word_count: int = 0

    def timestamp_iso(self) -> Optional[str]:
        """Return the timestamp in ISO-8601 format."""
        if self.timestamp is None:
            return None
        return self.timestamp.isoformat()
