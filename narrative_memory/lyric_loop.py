"""Simple lyric drafting loop using retrieved narrative context."""

from __future__ import annotations

import os
from typing import Dict, List, Optional

from google import genai
from google.genai import types as genai_types


class LyricDraftLoop:
    """Summarizes retrieved context and refines a draft line/block."""

    def __init__(self, model: str = "gemini-2.5-flash", google_api_key: Optional[str] = None):
        self.model = model
        self.api_key = google_api_key or os.getenv("GEMINI_API_KEY")
        if self.api_key:
            self.client = genai.Client(api_key=self.api_key)
        else:
            self.client = None

    def summarize_context(self, hits: List[Dict], max_items: int = 6) -> str:
        """Create a concise context summary for writing."""
        if not hits:
            return "No relevant narrative context found."
        selected = hits[:max_items]
        if not self.client:
            lines = []
            for i, hit in enumerate(selected, start=1):
                snippet = " ".join(hit["text"].split()[:28]).strip()
                lines.append(
                    f"{i}. [{hit['emotion']}/{hit['time_scope']}] {snippet}..."
                )
            return "Context cues:\n" + "\n".join(lines)

        prompt = (
            "Summarize these retrieved lyric-memory chunks into actionable writing notes.\n"
            "Keep facts grounded in the chunks. No invented details.\n"
            "Return 5-8 bullet points with emotional arc, imagery anchors, and narrative tension.\n\n"
            f"Chunks:\n{selected}"
        )
        try:
            resp = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=genai_types.GenerateContentConfig(temperature=0.2),
            )
            return resp.text.strip() or "No summary generated."
        except Exception:
            return "No summary generated."

    def refine_draft(self, seed_text: str, context_summary: str) -> str:
        """Refine a draft using the summary."""
        if not seed_text.strip():
            seed_text = "I keep chasing the same memory in different songs."

        if not self.client:
            return (
                "Refined Draft (heuristic):\n"
                f"{seed_text}\n"
                "I keep the room in frame, not just the feeling.\n"
                "I name the hour, the street, the version of me still speaking."
            )

        prompt = (
            "You are helping draft lyrics.\n"
            "Given the context summary and seed text, produce one polished 8-12 line draft.\n"
            "Prioritize specificity, emotional honesty, and fresh phrasing.\n"
            "Avoid cliches.\n\n"
            f"Context Summary:\n{context_summary}\n\n"
            f"Seed Text:\n{seed_text}\n"
        )
        try:
            resp = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=genai_types.GenerateContentConfig(temperature=0.7),
            )
            return resp.text.strip() or seed_text
        except Exception:
            return seed_text
