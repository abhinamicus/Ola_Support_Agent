from __future__ import annotations

from pathlib import Path

from .constants import KNOWLEDGE_BASE_DIR


def load_documents(directory: Path = KNOWLEDGE_BASE_DIR) -> list[dict[str, str]]:
    documents = []
    for path in sorted(directory.glob("*.md")):
        documents.append({"document_id": path.stem, "text": path.read_text(encoding="utf-8").strip()})
    return documents


def sentence_chunks(text: str) -> list[str]:
    chunks = [part.strip() for part in text.replace("\n", " ").split(".") if part.strip()]
    return [f"{chunk}." for chunk in chunks]


def fixed_chunks(text: str, size: int = 180, overlap: int = 40) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + size)
        chunks.append(text[start:end].strip())
        if end == len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def build_chunks(documents: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    fixed, sentence = [], []
    for doc in documents:
        for idx, chunk in enumerate(fixed_chunks(doc["text"])):
            fixed.append(
                {
                    "chunk_id": f"{doc['document_id']}-fixed-{idx}",
                    "document_id": doc["document_id"],
                    "text": chunk,
                }
            )
        for idx, chunk in enumerate(sentence_chunks(doc["text"])):
            sentence.append(
                {
                    "chunk_id": f"{doc['document_id']}-sentence-{idx}",
                    "document_id": doc["document_id"],
                    "text": chunk,
                }
            )
    return {"fixed": fixed, "sentence": sentence}

