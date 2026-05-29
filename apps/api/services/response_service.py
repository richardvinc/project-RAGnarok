from __future__ import annotations

import json
import re
from typing import Any, cast

from openai import OpenAI

from ..config import settings
from ..domain.models import (
    ChatMessagePayload,
    RetrievedChunk,
    SerializedToolCall,
    SerializedToolFunction,
    TranslatedChunkResult,
)
from ..schemas import ChunkResult, ConversationTurn, PreviousTurnContext, RAGResponse
from ..tools.registry import build_tool_schemas, dispatch_tool_call
from .decision_logging import DecisionLogger
from .retrieval_service import build_history_aware_search_query, retrieve_chunks

SYSTEM_PROMPT = """You are a helpful assistant and a good storyteller. Answer the user's question using ONLY the provided context.

Rules:
- If the context does not contain the answer, say: "I don't know based on the provided context."
- Do not use outside knowledge.
- Always cite sources in this exact format: `[source: {source}#chunk:{chunk_id}]`
- If user asks for image generation, call tools instead of pretending image exists.
- Follow the user's requested language, or naturally reply in the same language the user uses.
- If user needs translated source excerpts, call `translate_excerpt` with the cited chunks once they are known.
"""

CITATION_REGEX = re.compile(r"\[source:\s*[^#]+#chunk:([0-9,\s]+)\]")
INDONESIAN_HINTS = {
    "siapa",
    "siapakah",
    "apa",
    "apakah",
    "bagaimana",
    "mengapa",
    "tolong",
    "ceritakan",
    "bahasa",
    "indonesia",
    "terjemahkan",
    "gambar",
    "sumber",
}


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


def detect_translation_request(query: str) -> tuple[bool, str | None]:
    lowered = query.lower()
    explicit_indonesian = (
        "bahasa indonesia" in lowered
        or "indonesian" in lowered
        or "translate it to indonesian" in lowered
        or "translate to indonesian" in lowered
        or "terjemahkan" in lowered
    )
    token_hits = sum(1 for token in re.findall(r"[a-zA-Z]+", lowered) if token in INDONESIAN_HINTS)
    if explicit_indonesian:
        return True, "Indonesian"
    if token_hits >= 1:
        return True, "Indonesian"
    return False, None


def detect_text_language(text: str, query: str) -> str:
    lowered_text = text.lower()
    token_hits = sum(
        1 for token in re.findall(r"[a-zA-Z]+", lowered_text) if token in INDONESIAN_HINTS
    )
    if token_hits >= 1:
        return "id"

    lowered_query = query.lower()
    query_hits = sum(
        1 for token in re.findall(r"[a-zA-Z]+", lowered_query) if token in INDONESIAN_HINTS
    )
    if any(token in lowered_query for token in ("bahasa indonesia", "indonesian", "terjemahkan")):
        return "id"
    if query_hits >= 1:
        return "id"

    return "en"


def build_prompt_payload(
    query: str,
    formatted_context: str,
    *,
    translate_sources: bool,
    target_language: str | None,
    previous_turn_context: PreviousTurnContext | None,
) -> tuple[str, str]:
    translation_instruction = (
        f"If translated source data is needed, translate cited chunks to {target_language} using the translate_excerpt tool."
        if translate_sources and target_language
        else "Do not translate chunks unless the user explicitly asks for it."
    )
    user_content = (
        f"Question:\n{query}\n\n"
        "Instructions:\n"
        "- Decide the appropriate response language from the user's request.\n"
        f"- {translation_instruction}\n\n"
        f"{_build_previous_turn_context_block(previous_turn_context)}"
        f"Context:\n{formatted_context}"
    )
    final_prompt = f"SYSTEM: {SYSTEM_PROMPT}\n\nUSER: {user_content}"
    return user_content, final_prompt


def _build_previous_turn_context_block(
    previous_turn_context: PreviousTurnContext | None,
) -> str:
    if not previous_turn_context or not previous_turn_context.cited_chunks:
        return ""

    chunk_lines = []
    for chunk in previous_turn_context.cited_chunks:
        chunk_lines.append(
            f'[previous_chunk_id={chunk.chunk_id} source="{chunk.source}" section="{chunk.section_path}"]\n{chunk.content}'
        )

    return (
        "Previous turn answer context:\n"
        f"{previous_turn_context.assistant_response}\n\n"
        "Previous cited source data:\n"
        f"{chr(10).join(chunk_lines)}\n\n"
    )


def _history_to_messages(history: list[ConversationTurn]) -> list[dict[str, Any]]:
    recent_history = history[-6:]
    return [
        ChatMessagePayload(role=turn.role, content=turn.content).model_dump()
        for turn in recent_history
    ]


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


def extract_cited_chunk_ids(text: str) -> list[int]:
    cited_chunk_ids: list[int] = []
    seen: set[int] = set()
    for match in CITATION_REGEX.finditer(text):
        for raw_value in match.group(1).split(","):
            try:
                chunk_id = int(raw_value.strip())
            except ValueError:
                continue
            if chunk_id not in seen:
                seen.add(chunk_id)
                cited_chunk_ids.append(chunk_id)
    return cited_chunk_ids


def _build_translation_follow_up(
    *,
    cited_chunks: list[RetrievedChunk],
    target_language: str,
) -> dict[str, Any]:
    excerpt_payload = [
        {
            "chunk_id": chunk.id,
            "text": chunk.content,
            "source": chunk.source,
            "section_path": chunk.section_path,
        }
        for chunk in cited_chunks
    ]
    return ChatMessagePayload(
        role="user",
        content=(
            "Before finalizing, call the translate_excerpt tool for these cited chunks. "
            f"Target language: {target_language}. "
            f"Chunks: {json.dumps(excerpt_payload, ensure_ascii=False)}"
        ),
    ).model_dump()


def _build_translation_follow_up_from_previous_context(
    *,
    previous_turn_context: PreviousTurnContext,
    target_language: str,
) -> dict[str, Any]:
    excerpt_payload = [
        {
            "chunk_id": chunk.chunk_id,
            "text": chunk.content,
            "source": chunk.source,
            "section_path": chunk.section_path,
        }
        for chunk in previous_turn_context.cited_chunks
    ]
    return ChatMessagePayload(
        role="user",
        content=(
            "User is referring to previous answer. Call the translate_excerpt tool for these previous cited chunks. "
            f"Target language: {target_language}. "
            f"Chunks: {json.dumps(excerpt_payload, ensure_ascii=False)}"
        ),
    ).model_dump()


def _execute_requested_tools(
    response_message: Any,
    messages: list[dict[str, Any]],
    logger: DecisionLogger,
    *,
    round_index: int,
) -> tuple[str | None, list[dict[str, Any]], list[str], list[TranslatedChunkResult]]:
    image_url: str | None = None
    used_tool_names: list[str] = []
    translated_chunks: list[TranslatedChunkResult] = []

    messages.append(_serialize_response_message(response_message))

    for tool_call in response_message.tool_calls or []:
        tool_function = _extract_function_payload(tool_call)
        if tool_function is None:
            logger.log(
                step="tool_requested",
                stage="tool_use",
                status="skipped",
                decision="Skipped unsupported non-function tool call payload.",
                details={"tool_type": getattr(tool_call, "type", None), "round": round_index},
            )
            continue

        tool_name = tool_function["name"]
        raw_arguments = tool_function["arguments"]

        logger.log(
            step="tool_requested",
            stage="tool_use",
            status="completed",
            decision=f"LLM requested tool '{tool_name}'.",
            details={"arguments": raw_arguments, "round": round_index},
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
                details={"error": str(exc), "arguments": raw_arguments, "round": round_index},
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
        translated_chunks.extend(result.translated_chunks)

        logger.log(
            step="tool_executed",
            stage="tool_use",
            status="completed",
            decision=f"Tool '{tool_name}' completed successfully.",
            details={
                "arguments": result.parsed_arguments,
                "tool_response": result.tool_response.model_dump(),
                "round": round_index,
            },
            tool_name=tool_name,
        )

        messages.append(
            ChatMessagePayload(
                role="tool",
                tool_call_id=tool_call.id,
                name=tool_name,
                content=result.tool_response.model_dump_json(),
            ).model_dump(exclude_none=True)
        )

    return image_url, messages, used_tool_names, translated_chunks


def _merge_translated_chunks(
    base_chunks: list[RetrievedChunk],
    translated_chunks: list[TranslatedChunkResult],
) -> tuple[list[RetrievedChunk], list[TranslatedChunkResult]]:
    translations_by_chunk_id = {
        translated_chunk.chunk_id: translated_chunk for translated_chunk in translated_chunks
    }
    merged_chunks: list[RetrievedChunk] = []
    for chunk in base_chunks:
        translation = translations_by_chunk_id.get(chunk.id)
        merged_chunks.append(
            chunk.model_copy(
                update={
                    "translated_content": translation.translated_content if translation else None,
                }
            )
        )
    ordered_translations = [
        translations_by_chunk_id[chunk.id]
        for chunk in merged_chunks
        if chunk.id in translations_by_chunk_id
    ]
    return merged_chunks, ordered_translations


def run_agent_loop(
    query: str,
    retrieved_chunks: list[RetrievedChunk],
    logger: DecisionLogger,
    *,
    history: list[ConversationTurn],
    previous_turn_context: PreviousTurnContext | None,
) -> tuple[str, str, str, str | None, str, str | None, list[TranslatedChunkResult], list[int]]:
    client = get_llm_client()
    translate_sources, target_language = detect_translation_request(query)
    formatted_context = format_context(retrieved_chunks)

    logger.log(
        step="build_context",
        stage="prompting",
        status="completed",
        decision="Formatted retrieved chunks into grounded context.",
        details={
            "chunk_count": len(retrieved_chunks),
            "chunk_ids": [chunk.id for chunk in retrieved_chunks],
            "translate_sources": translate_sources,
            "target_language": target_language,
        },
    )

    user_content, final_prompt = build_prompt_payload(
        query,
        formatted_context,
        translate_sources=translate_sources,
        target_language=target_language,
        previous_turn_context=previous_turn_context,
    )
    logger.log(
        step="build_prompt",
        stage="prompting",
        status="completed",
        decision="Built the final prompt payload for the LLM.",
        details={"prompt_preview": final_prompt[:500]},
    )

    messages: list[dict[str, Any]] = [
        ChatMessagePayload(role="system", content=SYSTEM_PROMPT).model_dump(),
        *_history_to_messages(history),
        ChatMessagePayload(role="user", content=user_content).model_dump(),
    ]

    image_url: str | None = None
    translated_chunks: list[TranslatedChunkResult] = []
    translated_tool_called = False
    translation_follow_up_issued = False
    previous_context_follow_up_issued = False
    final_message = ""
    cited_chunk_ids: list[int] = []

    for round_index in range(1, settings.max_tool_rounds + 1):
        response = client.chat.completions.create(
            model=settings.llm_model,
            temperature=0,
            messages=messages,  # type: ignore[arg-type]
            tools=build_tool_schemas(),  # type: ignore[arg-type]
        )
        response_message = response.choices[0].message
        tool_names = [
            tool_function["name"]
            for tool_call in (response_message.tool_calls or [])
            if (tool_function := _extract_function_payload(tool_call)) is not None
        ]
        logger.log(
            step="llm_round_response",
            stage="generation",
            status="completed",
            decision=f"Received LLM response for round {round_index}.",
            details={
                "round": round_index,
                "requested_tools": tool_names,
                "has_tool_calls": bool(tool_names),
                "content_preview": (response_message.content or "")[:250],
            },
        )

        if response_message.tool_calls:
            current_image_url, messages, used_tool_names, current_translated_chunks = _execute_requested_tools(
                response_message,
                messages,
                logger,
                round_index=round_index,
            )
            image_url = current_image_url or image_url
            if current_translated_chunks:
                translated_tool_called = True
                translated_chunks.extend(current_translated_chunks)
            logger.log(
                step="tool_round_complete",
                stage="tool_use",
                status="completed",
                decision=f"Completed tool round {round_index}.",
                details={"round": round_index, "tool_names": used_tool_names},
            )
            continue

        final_message = response_message.content or ""
        cited_chunk_ids = extract_cited_chunk_ids(final_message)
        logger.log(
            step="citations_extracted",
            stage="generation",
            status="completed",
            decision="Extracted cited chunks from the current LLM answer.",
            details={"round": round_index, "cited_chunk_ids": cited_chunk_ids},
        )

        if (
            translate_sources
            and target_language
            and cited_chunk_ids
            and not translated_tool_called
            and not translation_follow_up_issued
        ):
            chunk_map = {chunk.id: chunk for chunk in retrieved_chunks}
            cited_chunks = [
                chunk_map[chunk_id]
                for chunk_id in cited_chunk_ids
                if chunk_id in chunk_map
            ]
            if cited_chunks:
                messages.append(
                    ChatMessagePayload(role="assistant", content=final_message).model_dump()
                )
                messages.append(
                    _build_translation_follow_up(
                        cited_chunks=cited_chunks,
                        target_language=target_language,
                    )
                )
                translation_follow_up_issued = True
                logger.log(
                    step="translation_follow_up",
                    stage="tool_use",
                    status="completed",
                    decision="Asked LLM to translate cited chunks using the translate_excerpt tool.",
                    details={
                        "round": round_index,
                        "target_language": target_language,
                        "cited_chunk_ids": cited_chunk_ids,
                    },
                )
                continue

        if (
            translate_sources
            and target_language
            and not cited_chunk_ids
            and previous_turn_context
            and previous_turn_context.cited_chunks
            and not translated_tool_called
            and not previous_context_follow_up_issued
        ):
            messages.append(
                ChatMessagePayload(role="assistant", content=final_message).model_dump()
            )
            messages.append(
                _build_translation_follow_up_from_previous_context(
                    previous_turn_context=previous_turn_context,
                    target_language=target_language,
                )
            )
            previous_context_follow_up_issued = True
            logger.log(
                step="translation_follow_up",
                stage="tool_use",
                status="completed",
                decision="Asked LLM to translate previous turn cited chunks using the translate_excerpt tool.",
                details={
                    "round": round_index,
                    "target_language": target_language,
                    "previous_chunk_ids": [
                        chunk.chunk_id for chunk in previous_turn_context.cited_chunks
                    ],
                },
            )
            continue

        logger.log(
            step="llm_final_response",
            stage="generation",
            status="completed",
            decision="Returned final answer from agent loop.",
            details={
                "round": round_index,
                "cited_chunk_ids": cited_chunk_ids,
                "content_preview": final_message[:250],
            },
        )
        llm_response_language = detect_text_language(final_message, query)
        return (
            final_message,
            formatted_context,
            final_prompt,
            image_url,
            llm_response_language,
            None,
            translated_chunks,
            cited_chunk_ids,
        )

    logger.log(
        step="llm_final_response",
        stage="generation",
        status="failed",
        decision="Stopped agent loop after reaching max tool rounds.",
        details={"max_tool_rounds": settings.max_tool_rounds},
    )
    llm_response_language = detect_text_language(final_message, query)
    return (
        final_message or "I don't know based on the provided context.",
        formatted_context,
        final_prompt,
        image_url,
        llm_response_language,
        None,
        translated_chunks,
        cited_chunk_ids,
    )

def run_rag_pipeline(
    query: str,
    *,
    k: int,
    history: list[ConversationTurn] | None = None,
    previous_turn_context: PreviousTurnContext | None = None,
) -> RAGResponse:
    logger = DecisionLogger()
    safe_history = history or []
    retrieval_query = build_history_aware_search_query(
        query,
        [f"{turn.role}: {turn.content}" for turn in safe_history],
    )

    logger.log(
        step="embed_query",
        stage="retrieval",
        status="completed",
        decision="Requested an embedding for the user query.",
        details={"query": query, "history_turns": len(safe_history)},
    )
    query_embedding, retrieved_chunks = retrieve_chunks(retrieval_query, k=k)
    logger.log(
        step="retrieve_chunks",
        stage="retrieval",
        status="completed",
        decision="Retrieved the most relevant chunks from pgvector.",
        details={
            "k": k,
            "retrieval_query_preview": retrieval_query[:300],
            "returned_chunk_count": len(retrieved_chunks),
            "chunk_ids": [chunk.id for chunk in retrieved_chunks],
        },
    )

    (
        llm_response,
        formatted_context,
        final_prompt,
        image_url,
        llm_response_language,
        translated_llm_response,
        translated_chunks,
        cited_chunk_ids,
    ) = run_agent_loop(
        query,
        retrieved_chunks,
        logger,
        history=safe_history,
        previous_turn_context=previous_turn_context,
    )

    merged_chunks, ordered_translated_chunks = _merge_translated_chunks(
        retrieved_chunks,
        translated_chunks,
    )

    return RAGResponse(
        query=query,
        query_embedding=query_embedding,
        retrieved_chunks=[
            ChunkResult.model_validate(chunk.model_dump()) for chunk in merged_chunks
        ],
        formatted_context=formatted_context,
        final_prompt=final_prompt,
        llm_response=llm_response,
        llm_response_language=llm_response_language,
        translated_llm_response=translated_llm_response,
        image_url=image_url,
        translated_chunks=ordered_translated_chunks,
        cited_chunk_ids=cited_chunk_ids,
        decision_log=logger.events,
        run_summary=logger.build_summary(),
    )
