from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from .config import settings
from .domain.models import RetrievedChunk, TranslatedChunkResult


class QueryRequest(BaseModel):
    query: str = Field(min_length=1)
    k: int = Field(default=settings.default_retrieval_k, ge=1, le=20)
    history: list["ConversationTurn"] = Field(default_factory=list)
    previous_turn_context: "PreviousTurnContext | None" = None


class ConversationTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1)


class PreviousTurnChunk(BaseModel):
    chunk_id: int
    source: str
    section_path: str
    content: str = Field(min_length=1)


class PreviousTurnContext(BaseModel):
    assistant_response: str = Field(min_length=1)
    cited_chunks: list[PreviousTurnChunk] = Field(default_factory=list)


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
    query_embedding: list[float]
    retrieved_chunks: list[ChunkResult]
    formatted_context: str
    final_prompt: str
    llm_response: str
    llm_response_language: str
    translated_llm_response: str | None = None
    image_url: str | None = None
    translated_chunks: list[TranslatedChunkResult] = Field(default_factory=list)
    cited_chunk_ids: list[int] = Field(default_factory=list)
    decision_log: list[DecisionEvent]
    run_summary: RunSummary
