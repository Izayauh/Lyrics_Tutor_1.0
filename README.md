# Narrative Memory Engine v1 (Local-First)

Minimal Python scaffold for lyric-writing memory retrieval:

- Ingest local text/JSON sources
- Semantic chunking by boundaries (not fixed tokens)
- Weak labels (LLM-first, heuristic fallback)
- Embedding + Chroma storage
- SQLite metadata storage
- Hybrid retrieval (time + metadata + vector + weighted ranking)
- Lyric draft loop (summary + refinement)

## Quick Start

1. Create and activate a Python environment (Windows or WSL).
2. Install dependencies:
   - `pip install -r requirements.txt`
3. Set API key if you want LLM labeling/drafting:
   - PowerShell: `$env:GEMINI_API_KEY="your_key_here"`
   - Bash: `export GEMINI_API_KEY="your_key_here"`
4. Review config:
   - `config.yaml` (or copy from `config.example.yaml`)
5. Ingest files:
   - `python scripts/run_ingest.py --input data/raw`
6. Run full demo:
   - `python scripts/run_demo.py`
7. Run E2E validation:
   - `python scripts/e2e_test.py`

## Input Types

- `.txt`, `.md`, `.json`
- JSON supports generic extraction and common ChatGPT export-like structures.

## Core Data Fields per Chunk

- `id` (uuid)
- `source`
- `timestamp`
- `text`
- `emotion`
- `time_scope`
- `intensity` (1-5)
- `voice_mode`
- `authenticity_score` (1-5)
- `specificity_score` (1-5)
- `cliche_score` (1-5)

## Notes

- SQLite is the source of truth for metadata filters.
- Chroma stores vectors keyed by the same chunk UUID.
- Retrieval flow: SQL prefilter -> vector search over candidates -> weighted rerank.
