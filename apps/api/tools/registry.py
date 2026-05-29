from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from openai import OpenAI

from ..config import settings
from ..domain.models import (
    GenerateStoryImageArgs,
    TranslatedChunkResult,
    TranslateExcerptArgs,
    ToolExecutionPayload,
    ToolExecutionResult,
    ToolSchemaDefinition,
)
from .image_generation import execute_generate_story_image, to_public_image_url


def build_tool_schemas() -> list[dict[str, Any]]:
    return [
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
                "name": "translate_excerpt",
                "description": (
                    "Translate cited source excerpts into target language while preserving meaning exactly."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "target_language": {
                            "type": "string",
                            "description": "Target language name such as Indonesian.",
                        },
                        "excerpts": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "chunk_id": {"type": "integer"},
                                    "text": {"type": "string"},
                                    "source": {"type": "string"},
                                    "section_path": {"type": "string"},
                                },
                                "required": ["chunk_id", "text", "source", "section_path"],
                            },
                        },
                    },
                    "required": ["target_language", "excerpts"],
                },
            },
        ).model_dump()
    ]


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


def _extract_json_object(raw_text: str) -> dict[str, Any]:
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        start = raw_text.find("{")
        end = raw_text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        return json.loads(raw_text[start : end + 1])


def _translate_excerpt(arguments: dict[str, Any]) -> ToolExecutionResult:
    parsed_arguments = TranslateExcerptArgs.model_validate(arguments)
    client = OpenAI(
        base_url=settings.lm_studio_base_url,
        api_key=settings.llm_api_key,
    )

    translation_prompt = json.dumps(
        {
            "target_language": parsed_arguments.target_language,
            "excerpts": [excerpt.model_dump() for excerpt in parsed_arguments.excerpts],
        },
        ensure_ascii=False,
    )
    response = client.chat.completions.create(
        model=settings.llm_model,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a translation tool. Translate each excerpt into the target language only. "
                    "Preserve meaning. Do not summarize. Return valid JSON with keys "
                    "`target_language` and `translations`. Each translation item must contain "
                    "`chunk_id`, `translated_content`, `source`, and `section_path`."
                ),
            },
            {
                "role": "user",
                "content": translation_prompt,
            },
        ],
    )
    raw_content = response.choices[0].message.content or "{}"
    parsed_response = _extract_json_object(raw_content)
    translated_chunks = [
        TranslatedChunkResult.model_validate(
            {
                "chunk_id": item["chunk_id"],
                "target_language": parsed_response.get(
                    "target_language",
                    parsed_arguments.target_language,
                ),
                "translated_content": item["translated_content"],
                "source": item["source"],
                "section_path": item["section_path"],
            }
        )
        for item in parsed_response.get("translations", [])
    ]
    return ToolExecutionResult(
        tool_name="translate_excerpt",
        arguments=parsed_arguments.model_dump(),
        parsed_arguments=parsed_arguments.model_dump(),
        tool_response=ToolExecutionPayload(
            status="success",
            message="Excerpts translated successfully.",
            target_language=parsed_arguments.target_language,
            translations=translated_chunks,
        ),
        translated_chunks=translated_chunks,
    )


TOOL_HANDLERS: dict[str, Callable[[dict[str, Any]], ToolExecutionResult]] = {
    "generate_story_image": _generate_story_image,
    "translate_excerpt": _translate_excerpt,
}


def dispatch_tool_call(tool_name: str, raw_arguments: str) -> ToolExecutionResult:
    handler = TOOL_HANDLERS.get(tool_name)
    if handler is None:
        raise ValueError(f"Unsupported tool requested: {tool_name}")

    parsed_arguments = json.loads(raw_arguments or "{}")
    result = handler(parsed_arguments)
    return result
