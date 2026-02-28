"""Weak semantic labeling with an LLM-first, heuristic-fallback design."""

from __future__ import annotations

import json
import os
import re
from typing import Dict, Iterable, List, Optional

from google import genai
from google.genai import types as genai_types

from .config import LabelingConfig
from .schemas import ChunkRecord


ALLOWED_EMOTIONS = [
    "joy",
    "sadness",
    "anger",
    "fear",
    "nostalgia",
    "hope",
    "love",
    "regret",
    "conflict",
    "calm",
    "unknown",
]

ALLOWED_TIME_SCOPES = ["past", "present", "future", "timeless", "mixed", "unknown"]
ALLOWED_VOICE_MODES = [
    "confessional",
    "observational",
    "dialogue",
    "imagistic",
    "boastful",
    "reflective",
    "unknown",
]


def _chunks(iterable: List[ChunkRecord], size: int) -> Iterable[List[ChunkRecord]]:
    for i in range(0, len(iterable), size):
        yield iterable[i : i + size]


def _extract_json_object(raw: str) -> Optional[dict]:
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{[\s\S]*\}", raw)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def _clamp_score(value: object, default: int = 3) -> int:
    try:
        score = int(value)
    except Exception:
        return default
    return max(1, min(5, score))


class WeakLabeler:
    """Assigns weak semantic labels to chunks."""

    def __init__(self, config: LabelingConfig, google_api_key: Optional[str] = None):
        self.config = config
        self.api_key = google_api_key or os.getenv("GEMINI_API_KEY")
        if self.api_key:
            self.client = genai.Client(api_key=self.api_key)
        else:
            self.client = None

    def label_chunks(self, chunks: List[ChunkRecord]) -> List[ChunkRecord]:
        if not chunks:
            return chunks
        if not self.config.enabled:
            return [self._apply_heuristic(chunk) for chunk in chunks]

        for batch in _chunks(chunks, self.config.batch_size):
            labels = self._label_batch_llm(batch) if self.client else None
            if labels is None and self.config.fallback_heuristic:
                for chunk in batch:
                    self._apply_heuristic(chunk)
                continue
            if labels is None:
                continue
            for chunk in batch:
                payload = labels.get(chunk.id)
                if payload is None:
                    if self.config.fallback_heuristic:
                        self._apply_heuristic(chunk)
                    continue
                self._apply_label_payload(chunk, payload)
        return chunks

    def _label_batch_llm(self, batch: List[ChunkRecord]) -> Optional[Dict[str, dict]]:
        if not self.client:
            return None

        batch_payload = [
            {"id": chunk.id, "text": chunk.text[:1800]}
            for chunk in batch
        ]

        prompt = f"""
You are a conservative weak-labeling engine for lyric-writing memory chunks.

Rules:
1) Use only the text evidence in each chunk.
2) Never invent facts or names.
3) If uncertain, set emotion/time_scope/voice_mode to "unknown".
4) Scores must be integers 1-5.
5) Output STRICT JSON only, no markdown and no extra commentary.

Allowed emotion values: {ALLOWED_EMOTIONS}
Allowed time_scope values: {ALLOWED_TIME_SCOPES}
Allowed voice_mode values: {ALLOWED_VOICE_MODES}

Return exactly:
{{
  "labels": [
    {{
      "id": "chunk-id",
      "emotion": "unknown",
      "time_scope": "unknown",
      "intensity": 3,
      "voice_mode": "unknown",
      "authenticity_score": 3,
      "specificity_score": 3,
      "cliche_score": 3
    }}
  ]
}}

Chunks:
{json.dumps(batch_payload, ensure_ascii=False)}
"""
        try:
            resp = self.client.models.generate_content(
                model=self.config.model,
                contents=prompt,
                config=genai_types.GenerateContentConfig(temperature=0),
            )
        except Exception:
            return None

        raw = resp.text
        if not raw:
            return None

        parsed = _extract_json_object(raw)
        if not parsed or "labels" not in parsed or not isinstance(parsed["labels"], list):
            return None

        out: Dict[str, dict] = {}
        for item in parsed["labels"]:
            if not isinstance(item, dict) or "id" not in item:
                continue
            out[str(item["id"])] = item
        return out

    def _apply_label_payload(self, chunk: ChunkRecord, payload: dict) -> None:
        emotion = str(payload.get("emotion", "unknown")).lower()
        time_scope = str(payload.get("time_scope", "unknown")).lower()
        voice_mode = str(payload.get("voice_mode", "unknown")).lower()

        chunk.emotion = emotion if emotion in ALLOWED_EMOTIONS else "unknown"
        chunk.time_scope = time_scope if time_scope in ALLOWED_TIME_SCOPES else "unknown"
        chunk.voice_mode = voice_mode if voice_mode in ALLOWED_VOICE_MODES else "unknown"
        chunk.intensity = _clamp_score(payload.get("intensity"))
        chunk.authenticity_score = _clamp_score(payload.get("authenticity_score"))
        chunk.specificity_score = _clamp_score(payload.get("specificity_score"))
        chunk.cliche_score = _clamp_score(payload.get("cliche_score"))

    def _apply_heuristic(self, chunk: ChunkRecord) -> ChunkRecord:
        text = chunk.text.lower()
        emotion = "unknown"

        if any(token in text for token in ["miss", "remember", "back then", "used to"]):
            emotion = "nostalgia"
        elif any(token in text for token in ["love", "kiss", "heart"]):
            emotion = "love"
        elif any(token in text for token in ["angry", "rage", "mad", "furious"]):
            emotion = "anger"
        elif any(token in text for token in ["sad", "cry", "alone", "hurt"]):
            emotion = "sadness"
        elif any(token in text for token in ["hope", "dream", "soon", "someday"]):
            emotion = "hope"
        elif any(token in text for token in ["calm", "quiet", "peace"]):
            emotion = "calm"

        time_scope = "unknown"
        past_markers = [
            "yesterday", "back then", "used to", "was ", "were ",
            "remember", "remembered", "said ", "told ", "wrote ",
            "knew ", "forgot", "left ", "lost ", "once ", " ago",
            "when i was", "had been", "i'd ", "i had ",
        ]
        present_markers = [
            "now", "today", "right now", "am ", "currently",
            "at this moment", "i'm ", "i am ",
        ]
        future_markers = [
            "tomorrow", "will ", "someday", "next year",
            "going to", "want to", "plan to", "hope to", "i'll ",
        ]
        if any(token in text for token in past_markers):
            time_scope = "past"
        elif any(token in text for token in present_markers):
            time_scope = "present"
        elif any(token in text for token in future_markers):
            time_scope = "future"

        voice_mode = "reflective" if any(token in text for token in ["i ", "my ", "me "]) else "observational"
        intensity = 1 + min(4, text.count("!") + (1 if any(token in text for token in ["never", "always"]) else 0))

        specificity = 2
        if re.search(r"\b\d{1,4}\b", text):
            specificity += 1
        if re.search(r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\b", text):
            specificity += 1
        if re.search(r"\b(avenue|street|station|city|room|kitchen)\b", text):
            specificity += 1

        authenticity = 3
        if re.search(r"\b(i|me|my|mine)\b", text):
            authenticity += 1
        if re.search(r"\b(feel|felt|truth|real)\b", text):
            authenticity += 1

        cliche_phrases = [
            "broken heart",
            "set me free",
            "in the dark",
            "one more chance",
            "tears in the rain",
        ]
        cliche = 1 + min(4, sum(1 for phrase in cliche_phrases if phrase in text))

        chunk.emotion = emotion
        chunk.time_scope = time_scope
        chunk.voice_mode = voice_mode if voice_mode in ALLOWED_VOICE_MODES else "unknown"
        chunk.intensity = _clamp_score(intensity)
        chunk.specificity_score = _clamp_score(specificity)
        chunk.authenticity_score = _clamp_score(authenticity)
        chunk.cliche_score = _clamp_score(cliche)
        return chunk
