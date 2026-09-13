from __future__ import annotations

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
KNOWLEDGE_BASE_DIR = ROOT_DIR / "knowledge_base"
DATA_DIR = ROOT_DIR / "work"
CHROMA_DIR = DATA_DIR
LOG_DIR = DATA_DIR
TRANSCRIPTS_DIR = DATA_DIR

FIXED_COLLECTION_NAME = "ola_kb_fixed"
SENTENCE_COLLECTION_NAME = "ola_kb_sentence"
