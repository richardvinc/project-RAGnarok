from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class RetrievedChunk(BaseModel):
    id: int
    source: str
    section_path: str
    content: str


class ChunkDraft(BaseModel):
    section_path: str
    text: str


class IngestedFileResult(BaseModel):
    file_name: str
    chunks_created: int


class ToolSchemaDefinition(BaseModel):
    type: str = "function"
    function: dict[str, Any]


class GenerateStoryImageArgs(BaseModel):
    prompt: str = Field(min_length=1)


class ToolExecutionPayload(BaseModel):
    status: str
    message: str
    saved_at: str | None = None
    image_url: str | None = None


class ToolExecutionResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    tool_name: str
    arguments: dict[str, Any]
    parsed_arguments: dict[str, Any]
    tool_response: ToolExecutionPayload
    image_url: str | None = None


class SerializedToolFunction(BaseModel):
    name: str
    arguments: str


class SerializedToolCall(BaseModel):
    id: str
    type: str
    function: SerializedToolFunction


class ChatMessagePayload(BaseModel):
    role: str
    content: str | None = None
    tool_calls: list[SerializedToolCall] | None = None
    tool_call_id: str | None = None
    name: str | None = None
