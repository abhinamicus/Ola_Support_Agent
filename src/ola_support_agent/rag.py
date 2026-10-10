from __future__ import annotations

import chromadb
from sentence_transformers import SentenceTransformer

from .kb import build_chunks, load_documents
from .schemas import EvaluationRow, GroundedAnswer, RetrievedChunk


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class RAGEngine:
    def __init__(self) -> None:
        self.documents = load_documents()
        self.chunk_sets = build_chunks(self.documents)

        # SentenceTransformers runs locally and converts text into dense vectors.
        self.embedding_model = SentenceTransformer(MODEL_NAME)

        # Use an in-memory Chroma client for this process.
        self.chroma = chromadb.Client()

        self.collections = {
            "fixed": self.chroma.get_or_create_collection(
                name="ola_kb_fixed",
                metadata={"hnsw:space": "cosine"},
            ),
            "sentence": self.chroma.get_or_create_collection(
                name="ola_kb_sentence",
                metadata={"hnsw:space": "cosine"},
            ),
        }

        self._index_chunks()

        self.call_counter = 0
        self.cache: dict[str, GroundedAnswer] = {}

        # None means retrieval is not ready until calibrated.
        self.threshold: float | None = None
        self.calibration_data = self.calibrate_threshold()

    def _index_chunks(self) -> None:
        for strategy, chunks in self.chunk_sets.items():
            collection = self.collections[strategy]

            texts = [chunk["text"] for chunk in chunks]

            embeddings = self.embedding_model.encode(
                texts,
                normalize_embeddings=True,
                show_progress_bar=False,
            )

            collection.upsert(
                ids=[chunk["chunk_id"] for chunk in chunks],
                embeddings=embeddings.tolist(),
                documents=texts,
                metadatas=[
                    {
                        "document_id": chunk["document_id"],
                        "chunking_strategy": strategy,
                    }
                    for chunk in chunks
                ],
            )

    def _embed_query(self, query: str) -> list[float]:
        embedding = self.embedding_model.encode(
            query,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        return embedding.tolist()

    def retrieve(
        self,
        query: str,
        strategy: str = "sentence",
        top_k: int = 3,
    ) -> list[dict]:
        if strategy not in self.collections:
            raise ValueError(f"Unknown chunking strategy: {strategy}")

        query_embedding = self._embed_query(query)

        results = self.collections[strategy].query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        retrieved: list[dict] = []

        ids = results["ids"][0]
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        for chunk_id, text, metadata, distance in zip(
            ids,
            documents,
            metadatas,
            distances,
        ):
            similarity = max(0.0, min(1.0, 1.0 - float(distance)))

            retrieved.append(
                {
                    "chunk_id": chunk_id,
                    "document_id": metadata["document_id"],
                    "text": text,
                    "score": round(similarity, 4),
                }
            )

        return retrieved

    def calibrate_threshold(self) -> dict[str, list[float] | float]:
        in_scope = [
            "What is the first-response SLA for P1 tickets?",
            "When should a billing dispute be escalated?",
            "How long are support tickets retained?",
        ]

        out_of_scope = [
            "What is Ola's maternity leave policy?",
            "Who won the IPL final?",
        ]

        in_scores = [
            self.retrieve(query, "sentence", top_k=1)[0]["score"]
            for query in in_scope
        ]

        out_scores = [
            self.retrieve(query, "sentence", top_k=1)[0]["score"]
            for query in out_of_scope
        ]

        # Do not blindly use this value.
        # We will inspect the two distributions and choose the threshold.
        self.threshold = round(
            (min(in_scores) + max(out_scores)) / 2,
            3,
        )

        return {
            "in_scope_top1_scores": [round(score, 3) for score in in_scores],
            "out_of_scope_top1_scores": [round(score, 3) for score in out_scores],
            "threshold": self.threshold,
        }

    def grounded_answer(
        self,
        query: str,
        strategy: str = "sentence",
        top_k: int = 3,
    ) -> GroundedAnswer:
        if self.threshold is None:
            raise RuntimeError(
                "RAG threshold has not been calibrated. "
                "Call calibrate_threshold() before answering queries."
            )
        normalized = " ".join(query.lower().split())
        cache_key = f"{strategy}|{top_k}|{normalized}"

        if cache_key in self.cache:
            cached = self.cache[cache_key].model_copy(deep=True)
            cached.cache_hit = True
            return cached

        self.call_counter += 1

        retrieved = self.retrieve(
            query,
            strategy=strategy,
            top_k=top_k,
        )

        top_score = retrieved[0]["score"] if retrieved else 0.0

        chunks = [
            RetrievedChunk(
                chunk_id=item["chunk_id"],
                document_id=item["document_id"],
                text=item["text"],
                score=item["score"],
            )
            for item in retrieved
        ]

        doc_ids = list(
            dict.fromkeys(
                item["document_id"]
                for item in retrieved
            )
        )

        if top_score < self.threshold:
            result = GroundedAnswer(
                answer=(
                    "I don't know based on the approved "
                    "Ola support knowledge base."
                ),
                grounded=False,
                chunking_strategy=strategy,
                top_score=top_score,
                retrieved_chunks=chunks,
                matched_documents=doc_ids,
                fallback_reason=(
                    f"top score {top_score:.3f} fell below "
                    f"threshold {self.threshold:.3f}"
                ),
            )

            self.cache[cache_key] = result
            return result

        # In MOCK_LLM mode, compose a deterministic answer from retrieved context.
        # The answer is restricted to text returned by the retriever.
        supporting_text = " ".join(
            item["text"] for item in retrieved[:2]
        )

        result = GroundedAnswer(
            answer=supporting_text[:450].strip(),
            grounded=True,
            chunking_strategy=strategy,
            top_score=top_score,
            retrieved_chunks=chunks,
            matched_documents=doc_ids,
        )

        self.cache[cache_key] = result
        return result

    def evaluate_chunking(
        self,
        queries: list[tuple[str, list[str]]],
        top_k: int = 3,
    ) -> list[EvaluationRow]:
        rows: list[EvaluationRow] = []

        for strategy in ("fixed", "sentence"):
            for query, expected_docs in queries:
                retrieved = self.retrieve(
                    query,
                    strategy=strategy,
                    top_k=top_k,
                )

                found = list(
                    dict.fromkeys(
                        item["document_id"]
                        for item in retrieved
                    )
                )

                intersection = sorted(
                    set(found) & set(expected_docs)
                )

                precision = (
                    len(intersection) / max(len(found), 1)
                )

                recall = (
                    len(intersection) / max(len(expected_docs), 1)
                )

                arithmetic = (
                    f"precision={len(intersection)}/"
                    f"{max(len(found), 1)}; "
                    f"recall={len(intersection)}/"
                    f"{max(len(expected_docs), 1)}"
                )

                rows.append(
                    EvaluationRow(
                        query=query,
                        expected_documents=expected_docs,
                        strategy=strategy,
                        precision=round(precision, 3),
                        recall=round(recall, 3),
                        arithmetic=arithmetic,
                    )
                )

        return rows