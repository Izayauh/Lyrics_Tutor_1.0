"""SQLite metadata storage for chunk records."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Dict, List, Optional

from .schemas import ChunkRecord


class SQLiteMetadataStore:
    """Persists and filters narrative chunk metadata."""

    def __init__(self, db_path: str):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY,
                source TEXT NOT NULL,
                timestamp TEXT,
                text TEXT NOT NULL,
                emotion TEXT NOT NULL,
                time_scope TEXT NOT NULL,
                intensity INTEGER NOT NULL,
                voice_mode TEXT NOT NULL,
                authenticity_score INTEGER NOT NULL,
                specificity_score INTEGER NOT NULL,
                cliche_score INTEGER NOT NULL,
                word_count INTEGER NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
            """
        )
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_timestamp ON chunks(timestamp)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_emotion ON chunks(emotion)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_time_scope ON chunks(time_scope)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source)")
        self.conn.commit()

    def upsert_chunks(self, chunks: List[ChunkRecord]) -> None:
        if not chunks:
            return
        query = """
        INSERT INTO chunks (
            id, source, timestamp, text, emotion, time_scope, intensity,
            voice_mode, authenticity_score, specificity_score, cliche_score, word_count
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            source=excluded.source,
            timestamp=excluded.timestamp,
            text=excluded.text,
            emotion=excluded.emotion,
            time_scope=excluded.time_scope,
            intensity=excluded.intensity,
            voice_mode=excluded.voice_mode,
            authenticity_score=excluded.authenticity_score,
            specificity_score=excluded.specificity_score,
            cliche_score=excluded.cliche_score,
            word_count=excluded.word_count
        """
        payload = [
            (
                chunk.id,
                chunk.source,
                chunk.timestamp_iso(),
                chunk.text,
                chunk.emotion,
                chunk.time_scope,
                int(chunk.intensity),
                chunk.voice_mode,
                int(chunk.authenticity_score),
                int(chunk.specificity_score),
                int(chunk.cliche_score),
                int(chunk.word_count),
            )
            for chunk in chunks
        ]
        self.conn.executemany(query, payload)
        self.conn.commit()

    def fetch_chunks_by_ids(self, ids: List[str]) -> Dict[str, Dict]:
        if not ids:
            return {}
        placeholders = ",".join("?" for _ in ids)
        rows = self.conn.execute(
            f"SELECT * FROM chunks WHERE id IN ({placeholders})",
            ids,
        ).fetchall()
        return {row["id"]: dict(row) for row in rows}

    def filter_chunks(
        self,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        filters: Optional[Dict] = None,
        limit: int = 500,
    ) -> List[Dict]:
        """Filter chunk metadata before vector search."""
        where = []
        params: List = []
        filters = filters or {}

        if start_time:
            where.append("timestamp IS NOT NULL")
            where.append("timestamp >= ?")
            params.append(start_time)
        if end_time:
            where.append("timestamp IS NOT NULL")
            where.append("timestamp <= ?")
            params.append(end_time)

        for key in ("source", "emotion", "time_scope", "voice_mode"):
            value = filters.get(key)
            if value:
                where.append(f"{key} = ?")
                params.append(value)

        min_intensity = filters.get("min_intensity")
        if min_intensity is not None:
            where.append("intensity >= ?")
            params.append(int(min_intensity))

        max_intensity = filters.get("max_intensity")
        if max_intensity is not None:
            where.append("intensity <= ?")
            params.append(int(max_intensity))

        min_authenticity = filters.get("min_authenticity")
        if min_authenticity is not None:
            where.append("authenticity_score >= ?")
            params.append(int(min_authenticity))

        min_specificity = filters.get("min_specificity")
        if min_specificity is not None:
            where.append("specificity_score >= ?")
            params.append(int(min_specificity))

        where_sql = f"WHERE {' AND '.join(where)}" if where else ""
        query = f"""
            SELECT * FROM chunks
            {where_sql}
            ORDER BY COALESCE(timestamp, '1970-01-01T00:00:00+00:00') DESC, created_at DESC
            LIMIT ?
        """
        params.append(max(1, limit))
        rows = self.conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]

    def count(self) -> int:
        row = self.conn.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()
        return int(row["n"]) if row else 0

    def close(self) -> None:
        self.conn.close()
