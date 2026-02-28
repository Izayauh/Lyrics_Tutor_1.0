"""Semantic chunking by paragraph/turn boundaries."""

from __future__ import annotations

import re
import uuid
from typing import List

from .config import ChunkingConfig
from .schemas import ChunkRecord, RawDocument


TURN_RE = re.compile(
    r"^(user|assistant|speaker\s*\d*|verse|chorus|bridge|hook|freestyle)\s*[:\-]",
    re.IGNORECASE,
)


def _word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def _split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


class SemanticChunker:
    """Creates semantically coherent chunks around narrative boundaries."""

    def __init__(self, config: ChunkingConfig):
        self.config = config

    def chunk_documents(self, docs: List[RawDocument]) -> List[ChunkRecord]:
        chunks: List[ChunkRecord] = []
        for doc in docs:
            chunks.extend(self._chunk_one(doc))
        return chunks

    def _chunk_one(self, doc: RawDocument) -> List[ChunkRecord]:
        segments = self._split_semantic_segments(doc.text)
        packed = self._pack_segments(segments)

        records: List[ChunkRecord] = []
        for chunk_text in packed:
            records.append(
                ChunkRecord(
                    id=str(uuid.uuid4()),
                    source=doc.source,
                    text=chunk_text,
                    timestamp=doc.timestamp,
                    word_count=_word_count(chunk_text),
                )
            )
        return records

    def _split_semantic_segments(self, text: str) -> List[str]:
        lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
        segments: List[str] = []
        current: List[str] = []

        for raw_line in lines:
            line = raw_line.strip()
            if not line:
                if current:
                    segments.append(" ".join(current).strip())
                    current = []
                continue
            if TURN_RE.match(line) and current:
                segments.append(" ".join(current).strip())
                current = [line]
                continue
            current.append(line)

        if current:
            segments.append(" ".join(current).strip())
        return [s for s in segments if s]

    def _split_long_segment(self, segment: str) -> List[str]:
        if _word_count(segment) <= self.config.hard_max_words:
            return [segment]
        out: List[str] = []
        current: List[str] = []
        cur_words = 0
        for sent in _split_sentences(segment):
            sent_words = _word_count(sent)
            if current and cur_words + sent_words > self.config.hard_max_words:
                out.append(" ".join(current).strip())
                current = [sent]
                cur_words = sent_words
            else:
                current.append(sent)
                cur_words += sent_words
        if current:
            out.append(" ".join(current).strip())
        return out

    def _pack_segments(self, segments: List[str]) -> List[str]:
        expanded: List[str] = []
        for seg in segments:
            expanded.extend(self._split_long_segment(seg))

        out: List[str] = []
        current: List[str] = []
        cur_words = 0

        def flush() -> None:
            nonlocal current, cur_words
            if current:
                out.append(" ".join(current).strip())
            current = []
            cur_words = 0

        for seg in expanded:
            seg_words = _word_count(seg)
            if not current:
                current = [seg]
                cur_words = seg_words
                continue

            proposed = cur_words + seg_words
            if proposed <= self.config.max_words:
                current.append(seg)
                cur_words = proposed
                continue

            if cur_words < self.config.min_words and proposed <= self.config.hard_max_words:
                current.append(seg)
                cur_words = proposed
                continue

            flush()
            current = [seg]
            cur_words = seg_words

        flush()

        if len(out) >= 2:
            last_wc = _word_count(out[-1])
            prev_wc = _word_count(out[-2])
            if last_wc < self.config.min_words and prev_wc + last_wc <= self.config.hard_max_words:
                out[-2] = f"{out[-2]} {out[-1]}".strip()
                out.pop()
        return out
