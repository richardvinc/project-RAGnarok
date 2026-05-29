from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from ..domain.models import (
    GenerateStoryImageArgs,
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


TOOL_HANDLERS: dict[str, Callable[[dict[str, Any]], ToolExecutionResult]] = {
    "generate_story_image": _generate_story_image,
}


def dispatch_tool_call(tool_name: str, raw_arguments: str) -> ToolExecutionResult:
    handler = TOOL_HANDLERS.get(tool_name)
    if handler is None:
        raise ValueError(f"Unsupported tool requested: {tool_name}")

    parsed_arguments = json.loads(raw_arguments or "{}")
    result = handler(parsed_arguments)
    return result
