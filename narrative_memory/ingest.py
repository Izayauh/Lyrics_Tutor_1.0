"""Ingestion and normalization utilities for raw local sources."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional

from dateutil import parser as dt_parser

from .schemas import RawDocument


SUPPORTED_EXTENSIONS = {".txt", ".md", ".json"}


def _normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _parse_optional_timestamp(value: object) -> Optional[datetime]:
    if value is None:
        return None
    try:
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(float(value), tz=timezone.utc)
        if isinstance(value, str) and value.strip():
            return dt_parser.parse(value)
    except Exception:
        return None
    return None


def _extract_timestamp_from_filename(path: Path) -> Optional[datetime]:
    match = re.search(r"(20\d{2}[-_]\d{2}[-_]\d{2})", path.name)
    if not match:
        return None
    raw = match.group(1).replace("_", "-")
    return _parse_optional_timestamp(raw)


def _extract_strings_from_json(node: object, out: List[str]) -> None:
    if node is None:
        return
    if isinstance(node, str):
        cleaned = node.strip()
        if cleaned:
            out.append(cleaned)
        return
    if isinstance(node, list):
        for item in node:
            _extract_strings_from_json(item, out)
        return
    if isinstance(node, dict):
        # Prefer known content-bearing keys first to avoid noisy metadata.
        for key in ("parts", "text", "content", "message"):
            if key in node:
                _extract_strings_from_json(node[key], out)
        for value in node.values():
            _extract_strings_from_json(value, out)


def _dedupe_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


class Ingestor:
    """Loads local text/JSON sources into a common document schema."""

    def ingest_paths(self, inputs: Iterable[str]) -> List[RawDocument]:
        documents: List[RawDocument] = []
        for raw in inputs:
            path = Path(raw)
            if path.is_dir():
                for child in path.rglob("*"):
                    if child.is_file() and child.suffix.lower() in SUPPORTED_EXTENSIONS:
                        documents.extend(self._load_file(child))
            elif path.is_file():
                documents.extend(self._load_file(path))
        return documents

    def _load_file(self, path: Path) -> List[RawDocument]:
        suffix = path.suffix.lower()
        if suffix in {".txt", ".md"}:
            return [self._load_plaintext(path)]
        if suffix == ".json":
            return self._load_json(path)
        return []

    def _load_plaintext(self, path: Path) -> RawDocument:
        text = path.read_text(encoding="utf-8", errors="ignore")
        return RawDocument(
            source=str(path),
            text=_normalize_text(text),
            timestamp=_extract_timestamp_from_filename(path),
        )

    def _load_json(self, path: Path) -> List[RawDocument]:
        raw = path.read_text(encoding="utf-8", errors="ignore")
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return [RawDocument(source=str(path), text=_normalize_text(raw), timestamp=None)]

        if isinstance(data, list):
            return self._load_json_list(path, data)
        if isinstance(data, dict):
            return self._load_json_dict(path, data)
        return []

    def _load_json_list(self, path: Path, data: list) -> List[RawDocument]:
        docs: List[RawDocument] = []
        for idx, item in enumerate(data):
            texts: List[str] = []
            _extract_strings_from_json(item, texts)
            text = _normalize_text("\n\n".join(_dedupe_preserve_order(texts)))
            if not text:
                continue
            ts = None
            if isinstance(item, dict):
                ts = _parse_optional_timestamp(
                    item.get("create_time") or item.get("timestamp") or item.get("update_time")
                )
            docs.append(
                RawDocument(
                    source=f"{path}#{idx}",
                    text=text,
                    timestamp=ts,
                    extra={"json_index": idx},
                )
            )
        return docs

    def _load_json_dict(self, path: Path, data: dict) -> List[RawDocument]:
        texts: List[str] = []
        _extract_strings_from_json(data, texts)
        text = _normalize_text("\n\n".join(_dedupe_preserve_order(texts)))
        if not text:
            return []
        ts = _parse_optional_timestamp(
            data.get("create_time") or data.get("timestamp") or data.get("update_time")
        )
        return [RawDocument(source=str(path), text=text, timestamp=ts)]
