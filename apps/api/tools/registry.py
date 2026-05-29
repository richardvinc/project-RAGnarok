from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from ..domain.models import (
    DetectQueryLanguageArgs,
    GenerateStoryImageArgs,
    TranslateQueryToEnglishArgs,
    TranslateRetrievedChunksArgs,
    ToolExecutionPayload,
    ToolExecutionResult,
    ToolSchemaDefinition,
)
from .image_generation import execute_generate_story_image, to_public_image_url
from .language_tools import detect_query_language, translate_query_to_english, translate_retrieved_chunks


def build_tool_schemas() -> list[dict[str, Any]]:
    return [
        ToolSchemaDefinition(
            function={
                "name": "detect_query_language",
                "description": (
                    "Detect the user's primary query language before answering. "
                    "Use this when you need to know whether the response should be in English or another language."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Raw user query to inspect for primary language.",
                        },
                    },
                    "required": ["query"],
                },
            },
        ).model_dump(),
        ToolSchemaDefinition(
            function={
                "name": "generate_story_image",
                "description": (
                    "Call this tool whenever user asks to visualize scene, "
                    "create illustration, or see character or event from The Jungle Book."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": (
                                "Vivid text prompt for image generation. "
                                "Describe raw scene elements. No style keywords like photorealistic."
                            ),
                        },
                    },
                    "required": ["prompt"],
                },
            },
        ).model_dump(),
        ToolSchemaDefinition(
            function={
                "name": "translate_query_to_english",
                "description": (
                    "Translate a non-English user query into English for retrieval over English documents."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Original non-English user query.",
                        },
                        "source_language_code": {
                            "type": "string",
                            "description": "Source language code such as id.",
                        },
                        "source_language_name": {
                            "type": "string",
                            "description": "Source language name such as Indonesian.",
                        },
                    },
                    "required": ["query", "source_language_code", "source_language_name"],
                },
            },
        ).model_dump(),
        ToolSchemaDefinition(
            function={
                "name": "translate_retrieved_chunks",
                "description": (
                    "Translate cited retrieved chunks into the user's target language "
                    "after the grounded answer is complete."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "target_language_code": {
                            "type": "string",
                            "description": "Lowercase language code such as en or id.",
                        },
                        "target_language_name": {
                            "type": "string",
                            "description": "Human-readable language name such as English or Indonesian.",
                        },
                        "chunks": {
                            "type": "array",
                            "description": "Retrieved chunks that need translation.",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "id": {"type": "integer"},
                                    "source": {"type": "string"},
                                    "section_path": {"type": "string"},
                                    "content": {"type": "string"},
                                },
                                "required": ["id", "source", "section_path", "content"],
                            },
                        },
                    },
                    "required": ["target_language_code", "target_language_name", "chunks"],
                },
            },
        ).model_dump(),
    ]


def _detect_query_language(arguments: dict[str, Any]) -> ToolExecutionResult:
    parsed_arguments = DetectQueryLanguageArgs.model_validate(arguments)
    detection = detect_query_language(parsed_arguments.model_dump())
    return ToolExecutionResult(
        tool_name="detect_query_language",
        arguments=parsed_arguments.model_dump(),
        parsed_arguments=parsed_arguments.model_dump(),
        tool_response=ToolExecutionPayload(
            status="success",
            message="Query language successfully detected.",
            data=detection,
        ),
    )


def _generate_story_image(arguments: dict[str, Any]) -> ToolExecutionResult:
    parsed_arguments = GenerateStoryImageArgs.model_validate(arguments)
    prompt = parsed_arguments.prompt.strip()
    saved_image_path = execute_generate_story_image(prompt)
    public_url = to_public_image_url(saved_image_path)
    return ToolExecutionResult(
        tool_name="generate_story_image",
        arguments={"prompt": prompt},
        parsed_arguments=parsed_arguments.model_dump(),
        tool_response=ToolExecutionPayload(
            status="success",
            saved_at=saved_image_path,
            image_url=public_url,
            message="Image successfully generated and saved to disk.",
        ),
        image_url=public_url,
    )


def _translate_query_to_english(arguments: dict[str, Any]) -> ToolExecutionResult:
    parsed_arguments = TranslateQueryToEnglishArgs.model_validate(arguments)
    translation_payload = translate_query_to_english(parsed_arguments.model_dump())
    return ToolExecutionResult(
        tool_name="translate_query_to_english",
        arguments=parsed_arguments.model_dump(),
        parsed_arguments=parsed_arguments.model_dump(),
        tool_response=ToolExecutionPayload(
            status="success",
            message="Query successfully translated to English for retrieval.",
            data=translation_payload,
        ),
    )


def _translate_retrieved_chunks(arguments: dict[str, Any]) -> ToolExecutionResult:
    parsed_arguments = TranslateRetrievedChunksArgs.model_validate(arguments)
    translation_payload = translate_retrieved_chunks(parsed_arguments.model_dump())
    return ToolExecutionResult(
        tool_name="translate_retrieved_chunks",
        arguments=parsed_arguments.model_dump(),
        parsed_arguments=parsed_arguments.model_dump(),
        tool_response=ToolExecutionPayload(
            status="success",
            message="Chunk translations successfully generated.",
            data=translation_payload,
        ),
    )


TOOL_HANDLERS: dict[str, Callable[[dict[str, Any]], ToolExecutionResult]] = {
    "detect_query_language": _detect_query_language,
    "generate_story_image": _generate_story_image,
    "translate_query_to_english": _translate_query_to_english,
    "translate_retrieved_chunks": _translate_retrieved_chunks,
}


def dispatch_tool_call(tool_name: str, raw_arguments: str) -> ToolExecutionResult:
    handler = TOOL_HANDLERS.get(tool_name)
    if handler is None:
        raise ValueError(f"Unsupported tool requested: {tool_name}")

    parsed_arguments = json.loads(raw_arguments or "{}")
    result = handler(parsed_arguments)
    return result
