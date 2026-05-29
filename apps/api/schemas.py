from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from .config import settings
from .domain.models import RetrievedChunk


class QueryRequest(BaseModel):
    query: str = Field(min_length=1)
    k: int = Field(default=settings.default_retrieval_k, ge=1, le=20)


class ChunkResult(RetrievedChunk):
    pass


class DecisionEvent(BaseModel):
    step: str
    stage: str
    status: Literal["completed", "skipped", "failed"]
    decision: str
    details: dict[str, Any] = Field(default_factory=dict)
    timestamp: str
    tool_name: str | None = None


class RunSummary(BaseModel):
    used_tools: bool
    tool_names: list[str]
    total_steps: int


class RAGResponse(BaseModel):
    query: str
    query_language_code: str
    query_language_name: str
    response_language_code: str
    response_language_name: str
    query_embedding: list[float]
    retrieved_chunks: list[ChunkResult]
    formatted_context: str
    final_prompt: str
    llm_response: str
    image_url: str | None = None
    decision_log: list[DecisionEvent]
    run_summary: RunSummary
