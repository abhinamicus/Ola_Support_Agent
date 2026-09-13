from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class RetrievedChunk(BaseModel):
    chunk_id: str
    document_id: str
    text: str
    score: float


class GroundedAnswer(BaseModel):
    answer: str
    grounded: bool
    chunking_strategy: Literal["fixed", "sentence"]
    top_score: float
    retrieved_chunks: list[RetrievedChunk]
    matched_documents: list[str]
    fallback_reason: str | None = None
    cache_hit: bool = False


class SupportTicketLookupResult(BaseModel):
    record_id: str
    status: str
    resolution_time_hours: float
    days_since_created: int
    escalated: bool
    escalation_score: float
    recommend_escalation: bool


class CrewResponse(BaseModel):
    trace_id: str
    user_query: str
    sanitized_query: str
    answer: str
    sources: list[str]
    lookup_result: SupportTicketLookupResult | None = None
    grounded: bool
    guardrail_events: list[str] = Field(default_factory=list)
    review_applied: bool = False
    cache_hit: bool = False


class ReviewVerdict(BaseModel):
    approved: bool
    final_answer: str
    reason: str


class AskRequest(BaseModel):
    query: str
    session_id: str = "default"


class AskResponse(BaseModel):
    response: CrewResponse


class AddDocumentRequest(BaseModel):
    document_id: str = Field(min_length=1, max_length=100)
    content: str = Field(min_length=1)

    @field_validator("document_id")
    @classmethod
    def validate_document_id(cls, value: str) -> str:
        if (
            "/" in value
            or "\\" in value
            or ".." in value
            or not value.replace("-", "").replace("_", "").isalnum()
        ):
            raise ValueError(
                "document_id may contain only letters, numbers, hyphens, and underscores"
            )
        return value


class EvaluationRow(BaseModel):
    query: str
    expected_documents: list[str]
    strategy: str
    precision: float
    recall: float
    arithmetic: str


class JudgeScore(BaseModel):
    query: str
    accuracy: int
    grounding: int
    completeness: int
    safety: int
    notes: str

