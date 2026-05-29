from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, cast

from openai import OpenAI

from ..config import settings
from ..domain.models import ChatMessagePayload, RetrievedChunk, SerializedToolCall, SerializedToolFunction
from ..schemas import ChunkResult, RAGResponse
from ..tools.registry import build_tool_schemas, dispatch_tool_call
from .decision_logging import DecisionLogger
from .retrieval_service import retrieve_chunks

CITATION_RE = re.compile(r"\[source:\s*[^#]+#chunk:([0-9,\s]+)\]")
RAW_CITATION_ITEM_RE = re.compile(r"source:\s*([^#\]]+?)#chunk(?:_id)?:\s*([0-9]+)", re.IGNORECASE)
RAW_CHUNK_ID_RE = re.compile(r"chunk_id\s*=\s*([0-9]+)", re.IGNORECASE)

POST_TOOL_SYSTEM_PROMPT = """You have tool results available.

Rules:
- Respond entirely in {response_language_name}.
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
            f'\n[chunk_id={chunk.id} source="{_display_source_name(chunk.source)}"{section_part}]\n{chunk.content}'
        )
    parts.append("\n--- END CONTEXT ---")
    return "\n".join(parts)


def build_system_prompt(*, response_language_name: str, is_english: bool) -> str:
    fallback_rule = (
        '- If the context does not contain the answer, say exactly: "I don\'t know based on the provided context."'
        if is_english
        else (
            '- If the context does not contain the answer, say the natural '
            f'equivalent of "I don\'t know based on the provided context." in {response_language_name}.'
        )
    )
    return f"""You are a helpful assistant and a good storyteller. Answer the user's question using ONLY the provided context.

Rules:
{fallback_rule}
- Do not use outside knowledge.
- Respond entirely in {response_language_name}.
- Cite sources in this exact format: `[source: {{source}}#chunk:{{chunk_id}}]`
- If the user asks to visualize, illustrate, or generate an image, you may call the available image tool.
"""


def build_prompt_payload(
    query: str,
    formatted_context: str,
    *,
    response_language_name: str,
    is_english: bool,
) -> tuple[str, str, str]:
    system_prompt = build_system_prompt(
        response_language_name=response_language_name,
        is_english=is_english,
    )
    user_content = f"Question:\n{query}\n\nContext:\n{formatted_context}"
    final_prompt = f"SYSTEM: {system_prompt}\n\nUSER: {user_content}"
    return system_prompt, user_content, final_prompt


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


def _filter_tool_schemas(*tool_names: str) -> list[dict[str, Any]]:
    tool_name_set = set(tool_names)
    return [
        schema
        for schema in build_tool_schemas()
        if cast(str, schema["function"]["name"]) in tool_name_set
    ]


def _extract_cited_chunk_ids(response_text: str) -> list[int]:
    cited_ids: list[int] = []
    for match in CITATION_RE.finditer(response_text):
        for raw_value in match.group(1).split(","):
            try:
                chunk_id = int(raw_value.strip())
            except ValueError:
                continue
            if chunk_id not in cited_ids:
                cited_ids.append(chunk_id)
    return cited_ids


def _display_source_name(source: str) -> str:
    normalized = source.replace("\\", "/").strip()
    if not normalized:
        return source
    return Path(normalized).name or source


def _normalize_response_citations(
    response_text: str,
    retrieved_chunks: list[RetrievedChunk],
) -> str:
    chunk_source_map = {
        chunk.id: _display_source_name(chunk.source)
        for chunk in retrieved_chunks
    }

    def replace_bracket(match: re.Match[str]) -> str:
        bracket_text = match.group(0)
        items = RAW_CITATION_ITEM_RE.findall(bracket_text)
        if items:
            source_name = _display_source_name(items[0][0])
            chunk_ids: list[str] = []
            for _, chunk_id in items:
                if chunk_id not in chunk_ids:
                    chunk_ids.append(chunk_id)

            return f"[source: {source_name}#chunk:{', '.join(chunk_ids)}]"

        raw_chunk_ids = RAW_CHUNK_ID_RE.findall(bracket_text)
        if not raw_chunk_ids:
            return bracket_text.replace("#chunk_id:", "#chunk:")

        grouped_ids: dict[str, list[str]] = {}
        for chunk_id in raw_chunk_ids:
            source_name = chunk_source_map.get(int(chunk_id))
            if not source_name:
                continue
            grouped_ids.setdefault(source_name, [])
            if chunk_id not in grouped_ids[source_name]:
                grouped_ids[source_name].append(chunk_id)

        if not grouped_ids:
            return bracket_text

        normalized_groups = [
            f"[source: {source_name}#chunk:{', '.join(chunk_ids)}]"
            for source_name, chunk_ids in grouped_ids.items()
        ]
        return ", ".join(normalized_groups)

    normalized = re.sub(r"\[[^\]]*source:[^\]]*\]", replace_bracket, response_text)
    normalized = re.sub(r"\[[^\]]*chunk_id[^\]]*\]", replace_bracket, normalized)
    return normalized.replace("#chunk_id:", "#chunk:")


def _sanitize_chunks_for_response(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    return [
        chunk.model_copy(update={"source": _display_source_name(chunk.source)})
        for chunk in chunks
    ]


def _execute_requested_tools(
    response_message: Any,
    messages: list[dict[str, Any]],
    logger: DecisionLogger,
) -> tuple[str | None, list[dict[str, Any]], list[str], list[dict[str, Any]]]:
    image_url: str | None = None
    used_tool_names: list[str] = []
    executed_results: list[dict[str, Any]] = []

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
        executed_results.append(
            {
                "tool_name": result.tool_name,
                "parsed_arguments": result.parsed_arguments,
                "tool_response": tool_response.model_dump(),
            }
        )

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

    return image_url, messages, used_tool_names, executed_results


def _invoke_tool_stage(
    *,
    tool_name: str,
    arguments: dict[str, Any],
    stage: str,
    requested_decision: str,
    completed_decision: str,
    logger: DecisionLogger,
) -> dict[str, Any]:
    try:
        logger.log(
            step="tool_requested",
            stage=stage,
            status="completed",
            decision=requested_decision,
            details={"arguments": arguments},
            tool_name=tool_name,
        )
        result = dispatch_tool_call(tool_name, json.dumps(arguments, ensure_ascii=False))
    except Exception as exc:
        logger.log(
            step="tool_executed",
            stage=stage,
            status="failed",
            decision=f"Tool '{tool_name}' execution failed.",
            details={"error": str(exc), "arguments": arguments},
            tool_name=tool_name,
        )
        raise

    tool_response = result.tool_response.model_dump()
    logger.log(
        step="tool_executed",
        stage=stage,
        status="completed",
        decision=completed_decision,
        details={
            "arguments": result.parsed_arguments,
            "tool_response": tool_response,
        },
        tool_name=tool_name,
    )
    return tool_response


def detect_query_language(query: str, logger: DecisionLogger) -> dict[str, Any]:
    tool_response = _invoke_tool_stage(
        tool_name="detect_query_language",
        arguments={"query": query},
        stage="preflight",
        requested_decision="Pipeline requested query language detection.",
        completed_decision="Tool 'detect_query_language' completed successfully.",
        logger=logger,
    )
    payload = cast(dict[str, Any], tool_response.get("data", {}))
    logger.log(
        step="language_detected",
        stage="preflight",
        status="completed",
        decision="Detected the user's primary query language.",
        details=payload,
        tool_name="detect_query_language",
    )
    return payload


def build_retrieval_query(
    *,
    query: str,
    is_english: bool,
    query_language_code: str,
    query_language_name: str,
    logger: DecisionLogger,
) -> str:
    if is_english:
        logger.log(
            step="query_translation",
            stage="preflight",
            status="skipped",
            decision="Skipped retrieval query translation because the user query is already English.",
            details={"retrieval_query": query},
            tool_name="translate_query_to_english",
        )
        return query

    tool_response = _invoke_tool_stage(
        tool_name="translate_query_to_english",
        arguments={
            "query": query,
            "source_language_code": query_language_code,
            "source_language_name": query_language_name,
        },
        stage="preflight",
        requested_decision="Pipeline requested query translation to English for retrieval.",
        completed_decision="Tool 'translate_query_to_english' completed successfully.",
        logger=logger,
    )
    payload = cast(dict[str, Any], tool_response.get("data", {}))
    retrieval_query = cast(str, payload.get("translated_query", query))
    logger.log(
        step="query_translation",
        stage="preflight",
        status="completed",
        decision="Translated the user query to English for retrieval over English documents.",
        details={
            "original_query": query,
            "retrieval_query": retrieval_query,
            "source_language_code": query_language_code,
            "source_language_name": query_language_name,
        },
        tool_name="translate_query_to_english",
    )
    return retrieval_query


def generate_grounded_response(
    query: str,
    retrieved_chunks: list[RetrievedChunk],
    logger: DecisionLogger,
    *,
    response_language_code: str,
    response_language_name: str,
    is_english: bool,
) -> tuple[str, str, str, str | None]:
    client = get_llm_client()
    formatted_context = format_context(retrieved_chunks)

    system_prompt, user_content, final_prompt = build_prompt_payload(
        query,
        formatted_context,
        response_language_name=response_language_name,
        is_english=is_english,
    )

    messages: list[dict[str, Any]] = [
        ChatMessagePayload(role="system", content=system_prompt).model_dump(),
        ChatMessagePayload(role="user", content=user_content).model_dump(),
    ]
    first_response = client.chat.completions.create(
        model=settings.llm_model,
        temperature=0,
        messages=messages,  # type: ignore[arg-type]
        tools=_filter_tool_schemas("generate_story_image"),  # type: ignore[arg-type]
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
            "response_language_code": response_language_code,
            "response_language_name": response_language_name,
        },
    )

    if first_message.tool_calls:
        image_url, messages, used_tool_names, _ = _execute_requested_tools(first_message, messages, logger)
        final_messages = [
            *messages,
            ChatMessagePayload(
                role="system",
                content=POST_TOOL_SYSTEM_PROMPT.format(
                    response_language_name=response_language_name,
                ),
            ).model_dump(),
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
                "response_language_code": response_language_code,
                "response_language_name": response_language_name,
            },
        )
        return _normalize_response_citations(final_message, retrieved_chunks), formatted_context, final_prompt, image_url

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
        details={
            "content_preview": llm_response[:250],
            "response_language_code": response_language_code,
            "response_language_name": response_language_name,
        },
    )
    return _normalize_response_citations(llm_response, retrieved_chunks), formatted_context, final_prompt, None


def translate_cited_chunks(
    retrieved_chunks: list[RetrievedChunk],
    *,
    cited_chunk_ids: list[int],
    target_language_code: str,
    target_language_name: str,
    logger: DecisionLogger,
) -> list[RetrievedChunk]:
    if target_language_code == "en" or not cited_chunk_ids:
        if target_language_code == "en":
            logger.log(
                step="chunk_translation",
                stage="post_processing",
                status="skipped",
                decision="Skipped chunk translation because the response language is English.",
                details={"target_language_code": target_language_code},
                tool_name="translate_retrieved_chunks",
            )
        else:
            logger.log(
                step="chunk_translation",
                stage="post_processing",
                status="skipped",
                decision="Skipped chunk translation because no cited chunks were found in the final answer.",
                details={"cited_chunk_ids": cited_chunk_ids},
                tool_name="translate_retrieved_chunks",
            )
        return retrieved_chunks

    chunks_to_translate = [
        chunk
        for chunk in retrieved_chunks
        if chunk.id in cited_chunk_ids
    ]

    tool_response = _invoke_tool_stage(
        tool_name="translate_retrieved_chunks",
        arguments={
            "target_language_code": target_language_code,
            "target_language_name": target_language_name,
            "chunks": [
                {
                    "id": chunk.id,
                    "source": chunk.source,
                    "section_path": chunk.section_path,
                    "content": chunk.content,
                }
                for chunk in chunks_to_translate
            ],
        },
        stage="post_processing",
        requested_decision="Pipeline requested cited chunk translation for the response language.",
        completed_decision="Tool 'translate_retrieved_chunks' completed successfully.",
        logger=logger,
    )
    payload = cast(dict[str, Any], tool_response.get("data", {}))
    translation_map = {
        int(item["id"]): str(item["translated_content"])
        for item in cast(list[dict[str, Any]], payload.get("translations", []))
    }

    logger.log(
        step="chunk_translation",
        stage="post_processing",
        status="completed",
        decision="Translated cited retrieved chunks into the response language.",
        details={
            "target_language_code": target_language_code,
            "target_language_name": target_language_name,
            "translated_chunk_ids": list(translation_map.keys()),
        },
        tool_name="translate_retrieved_chunks",
    )

    translated_chunks: list[RetrievedChunk] = []
    for chunk in retrieved_chunks:
        translated_chunks.append(
            chunk.model_copy(
                update={
                    "translated_content": translation_map.get(chunk.id),
                    "translated_language_code": (
                        target_language_code if chunk.id in translation_map else None
                    ),
                    "translated_language_name": (
                        target_language_name if chunk.id in translation_map else None
                    ),
                }
            )
        )

    return translated_chunks


def run_rag_pipeline(query: str, *, k: int) -> RAGResponse:
    logger = DecisionLogger()

    language_detection = detect_query_language(query, logger)
    query_language_code = cast(str, language_detection.get("language_code", "en"))
    query_language_name = cast(str, language_detection.get("language_name", "English"))
    is_english = bool(language_detection.get("is_english", query_language_code == "en"))
    response_language_code = "en" if is_english else query_language_code
    response_language_name = "English" if is_english else query_language_name
    retrieval_query = build_retrieval_query(
        query=query,
        is_english=is_english,
        query_language_code=query_language_code,
        query_language_name=query_language_name,
        logger=logger,
    )

    query_embedding, retrieved_chunks = retrieve_chunks(retrieval_query, k=k)

    llm_response, formatted_context, final_prompt, image_url = generate_grounded_response(
        query,
        retrieved_chunks,
        logger,
        response_language_code=response_language_code,
        response_language_name=response_language_name,
        is_english=is_english,
    )

    cited_chunk_ids = _extract_cited_chunk_ids(llm_response)
    translated_chunks = translate_cited_chunks(
        retrieved_chunks,
        cited_chunk_ids=cited_chunk_ids,
        target_language_code=response_language_code,
        target_language_name=response_language_name,
        logger=logger,
    )
    response_chunks = _sanitize_chunks_for_response(translated_chunks)

    return RAGResponse(
        query=query,
        query_language_code=query_language_code,
        query_language_name=query_language_name,
        response_language_code=response_language_code,
        response_language_name=response_language_name,
        query_embedding=query_embedding,
        retrieved_chunks=[ChunkResult.model_validate(chunk.model_dump()) for chunk in response_chunks],
        formatted_context=formatted_context,
        final_prompt=final_prompt,
        llm_response=llm_response,
        image_url=image_url,
        decision_log=logger.events,
        run_summary=logger.build_summary(),
    )
