from __future__ import annotations

import json
from typing import Any, cast

from openai import OpenAI

from ..config import settings
from ..domain.models import ChatMessagePayload, RetrievedChunk, SerializedToolCall, SerializedToolFunction
from ..schemas import ChunkResult, RAGResponse
from ..tools.registry import build_tool_schemas, dispatch_tool_call
from .decision_logging import DecisionLogger
from .retrieval_service import retrieve_chunks

SYSTEM_PROMPT = """You are a helpful assistant and a good storyteller. Answer the user's question using ONLY the provided context.

Rules:
- If the context does not contain the answer, say: "I don't know based on the provided context."
- Do not use outside knowledge.
- Cite sources in this exact format: `[source: {source}#chunk:{chunk_id}]`
- If the user asks to visualize, illustrate, or generate an image, you may call the available image tool.
"""

POST_TOOL_SYSTEM_PROMPT = """You have tool results available.

Rules:
- If the image generation tool succeeded, clearly tell the user the image was generated.
- Do not answer with "I don't know based on the provided context." when the user's image request was fulfilled by the tool.
- If the retrieved context supports factual details, you may include them with citations.
- If the context does not support extra factual claims, keep the reply limited to confirming the generated image and any non-factual tool result details.
"""


def get_llm_client() -> OpenAI:
    return OpenAI(
        base_url=settings.lm_studio_base_url,
        api_key=settings.llm_api_key,
    )


def format_context(chunks: list[RetrievedChunk]) -> str:
    parts: list[str] = ["--- BEGIN CONTEXT ---"]
    for chunk in chunks:
        section = chunk.section_path or ""
        section_part = f' section="{section}"' if section else ""
        parts.append(
            f'\n[chunk_id={chunk.id} source="{chunk.source}"{section_part}]\n{chunk.content}'
        )
    parts.append("\n--- END CONTEXT ---")
    return "\n".join(parts)


def build_prompt_payload(query: str, formatted_context: str) -> tuple[str, str]:
    user_content = f"Question:\n{query}\n\nContext:\n{formatted_context}"
    final_prompt = f"SYSTEM: {SYSTEM_PROMPT}\n\nUSER: {user_content}"
    return user_content, final_prompt


def _serialize_response_message(message: Any) -> dict[str, Any]:
    payload = ChatMessagePayload(role=message.role, content=getattr(message, "content", None))
    if getattr(message, "tool_calls", None):
        payload.tool_calls = [
            SerializedToolCall(
                id=tool_call.id,
                type=tool_call.type,
                function=SerializedToolFunction(
                    name=tool_function["name"],
                    arguments=tool_function["arguments"],
                ),
            )
            for tool_call in message.tool_calls
            if (tool_function := _extract_function_payload(tool_call)) is not None
        ]
    return payload.model_dump(exclude_none=True)


def _extract_function_payload(tool_call: Any) -> dict[str, str] | None:
    tool_type = getattr(tool_call, "type", None)
    tool_function = getattr(tool_call, "function", None)
    if tool_type != "function" or tool_function is None:
        return None

    name = cast(str | None, getattr(tool_function, "name", None))
    arguments = cast(str | None, getattr(tool_function, "arguments", None))
    if not name:
        return None

    return {
        "name": name,
        "arguments": arguments or "{}",
    }


def _execute_requested_tools(
    response_message: Any,
    messages: list[dict[str, Any]],
    logger: DecisionLogger,
) -> tuple[str | None, list[dict[str, Any]], list[str]]:
    image_url: str | None = None
    used_tool_names: list[str] = []

    messages.append(_serialize_response_message(response_message))

    for tool_call in response_message.tool_calls or []:
        tool_function = _extract_function_payload(tool_call)
        if tool_function is None:
            logger.log(
                step="tool_requested",
                stage="tool_use",
                status="skipped",
                decision="Skipped unsupported non-function tool call payload.",
                details={"tool_type": getattr(tool_call, "type", None)},
            )
            continue

        tool_name = tool_function["name"]
        raw_arguments = tool_function["arguments"]

        logger.log(
            step="tool_requested",
            stage="tool_use",
            status="completed",
            decision=f"LLM requested tool '{tool_name}'.",
            details={"arguments": raw_arguments},
            tool_name=tool_name,
        )

        try:
            result = dispatch_tool_call(tool_name, raw_arguments)
        except Exception as exc:
            error_payload = {
                "status": "error",
                "message": str(exc),
            }
            logger.log(
                step="tool_executed",
                stage="tool_use",
                status="failed",
                decision=f"Tool '{tool_name}' execution failed.",
                details={"error": str(exc), "arguments": raw_arguments},
                tool_name=tool_name,
            )
            messages.append(
                ChatMessagePayload(
                    role="tool",
                    tool_call_id=tool_call.id,
                    name=tool_name,
                    content=json.dumps(error_payload),
                ).model_dump(exclude_none=True)
            )
            continue

        used_tool_names.append(result.tool_name)
        image_url = result.image_url or image_url
        tool_response = result.tool_response

        logger.log(
            step="tool_executed",
            stage="tool_use",
            status="completed",
            decision=f"Tool '{tool_name}' completed successfully.",
            details={
                "arguments": result.parsed_arguments,
                "tool_response": tool_response.model_dump(),
            },
            tool_name=tool_name,
        )

        messages.append(
            ChatMessagePayload(
                role="tool",
                tool_call_id=tool_call.id,
                name=tool_name,
                content=tool_response.model_dump_json(),
            ).model_dump(exclude_none=True)
        )

    return image_url, messages, used_tool_names


def generate_grounded_response(
    query: str,
    retrieved_chunks: list[RetrievedChunk],
    logger: DecisionLogger,
) -> tuple[str, str, str, str | None]:
    client = get_llm_client()
    formatted_context = format_context(retrieved_chunks)

    user_content, final_prompt = build_prompt_payload(query, formatted_context)

    messages: list[dict[str, Any]] = [
        ChatMessagePayload(role="system", content=SYSTEM_PROMPT).model_dump(),
        ChatMessagePayload(role="user", content=user_content).model_dump(),
    ]
    first_response = client.chat.completions.create(
        model=settings.llm_model,
        temperature=0,
        messages=messages,  # type: ignore[arg-type]
        tools=build_tool_schemas(),  # type: ignore[arg-type]
    )
    first_message = first_response.choices[0].message
    tool_names = [
        tool_function["name"]
        for tool_call in (first_message.tool_calls or [])
        if (tool_function := _extract_function_payload(tool_call)) is not None
    ]
    logger.log(
        step="llm_first_response",
        stage="generation",
        status="completed",
        decision="Received the initial LLM response.",
        details={
            "requested_tools": tool_names,
            "has_tool_calls": bool(tool_names),
            "content_preview": (first_message.content or "")[:250],
        },
    )

    if first_message.tool_calls:
        image_url, messages, used_tool_names = _execute_requested_tools(first_message, messages, logger)
        final_messages = [
            *messages,
            ChatMessagePayload(role="system", content=POST_TOOL_SYSTEM_PROMPT).model_dump(),
        ]
        final_response = client.chat.completions.create(
            model=settings.llm_model,
            messages=final_messages,  # type: ignore[arg-type]
            temperature=0,
        )
        final_message = final_response.choices[0].message.content or ""
        logger.log(
            step="llm_final_response",
            stage="generation",
            status="completed",
            decision="Generated the final answer after tool execution.",
            details={
                "tool_names": used_tool_names,
                "content_preview": final_message[:250],
            },
        )
        return final_message, formatted_context, final_prompt, image_url

    logger.log(
        step="tool_requested",
        stage="tool_use",
        status="skipped",
        decision="LLM answered directly without using any tools.",
        details={},
    )
    llm_response = first_message.content or ""
    logger.log(
        step="llm_final_response",
        stage="generation",
        status="completed",
        decision="Returned the direct LLM answer.",
        details={"content_preview": llm_response[:250]},
    )
    return llm_response, formatted_context, final_prompt, None


def run_rag_pipeline(query: str, *, k: int) -> RAGResponse:
    logger = DecisionLogger()

    query_embedding, retrieved_chunks = retrieve_chunks(query, k=k)

    llm_response, formatted_context, final_prompt, image_url = generate_grounded_response(
        query,
        retrieved_chunks,
        logger,
    )

    return RAGResponse(
        query=query,
        query_embedding=query_embedding,
        retrieved_chunks=[ChunkResult.model_validate(chunk.model_dump()) for chunk in retrieved_chunks],
        formatted_context=formatted_context,
        final_prompt=final_prompt,
        llm_response=llm_response,
        image_url=image_url,
        decision_log=logger.events,
        run_summary=logger.build_summary(),
    )
