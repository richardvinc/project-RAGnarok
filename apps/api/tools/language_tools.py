from __future__ import annotations

import json
from typing import Any

from openai import OpenAI

from ..config import settings
from ..domain.models import (
    DetectQueryLanguageArgs,
    TranslateQueryToEnglishArgs,
    TranslateRetrievedChunksArgs,
)


def get_language_tool_client() -> OpenAI:
    return OpenAI(
        base_url=settings.lm_studio_base_url,
        api_key=settings.llm_api_key,
    )


def _parse_json_content(content: str) -> dict[str, Any]:
    raw_content = content.strip()
    if not raw_content:
        return {}

    try:
        return json.loads(raw_content)
    except json.JSONDecodeError:
        start = raw_content.find("{")
        end = raw_content.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        return json.loads(raw_content[start : end + 1])


def _create_json_completion(client: OpenAI, messages: list[dict[str, str]]) -> dict[str, Any]:
    try:
        response = client.chat.completions.create(
            model=settings.llm_model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=messages,  # type: ignore[arg-type]
        )
    except Exception:
        response = client.chat.completions.create(
            model=settings.llm_model,
            temperature=0,
            messages=messages,  # type: ignore[arg-type]
        )

    return _parse_json_content(response.choices[0].message.content or "")


def detect_query_language(arguments: dict[str, Any]) -> dict[str, Any]:
    parsed_arguments = DetectQueryLanguageArgs.model_validate(arguments)
    client = get_language_tool_client()

    payload = _create_json_completion(
        client,
        [
            {
                "role": "system",
                "content": (
                    "Detect the primary language of the user's query. "
                    "Return strict JSON with keys: language_code, language_name, is_english. "
                    "Use lowercase ISO 639-1 language_code when possible. "
                    "Set is_english true only when the query is primarily English."
                ),
            },
            {
                "role": "user",
                "content": parsed_arguments.query,
            },
        ],
    )

    language_code = str(payload.get("language_code", "en")).strip().lower() or "en"
    language_name = str(payload.get("language_name", "English")).strip() or "English"
    is_english = bool(payload.get("is_english", language_code == "en"))

    if is_english:
        language_code = "en"
        language_name = "English"

    return {
        "language_code": language_code,
        "language_name": language_name,
        "is_english": is_english,
    }


def translate_query_to_english(arguments: dict[str, Any]) -> dict[str, Any]:
    parsed_arguments = TranslateQueryToEnglishArgs.model_validate(arguments)
    client = get_language_tool_client()

    payload = _create_json_completion(
        client,
        [
            {
                "role": "system",
                "content": (
                    "Translate the user's search query into natural English for retrieval over English documents. "
                    "Return strict JSON with one key named translated_query. "
                    "Preserve meaning, named entities, and implied question intent. "
                    "Return only the translated query, not an answer."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "source_language_code": parsed_arguments.source_language_code,
                        "source_language_name": parsed_arguments.source_language_name,
                        "query": parsed_arguments.query,
                    },
                    ensure_ascii=False,
                ),
            },
        ],
    )

    translated_query = str(payload.get("translated_query", "")).strip()
    if not translated_query:
        translated_query = parsed_arguments.query

    return {
        "translated_query": translated_query,
    }


def translate_retrieved_chunks(arguments: dict[str, Any]) -> dict[str, Any]:
    parsed_arguments = TranslateRetrievedChunksArgs.model_validate(arguments)
    client = get_language_tool_client()

    chunk_payload = [
        {
            "id": chunk.id,
            "source": chunk.source,
            "section_path": chunk.section_path,
            "content": chunk.content,
        }
        for chunk in parsed_arguments.chunks
    ]

    payload = _create_json_completion(
        client,
        [
            {
                "role": "system",
                "content": (
                    "Translate each chunk into the requested target language. "
                    "Return strict JSON with one key named translations. "
                    "translations must be an array of objects with keys id and translated_content. "
                    "Preserve meaning. Do not omit any provided chunk."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "target_language_code": parsed_arguments.target_language_code,
                        "target_language_name": parsed_arguments.target_language_name,
                        "chunks": chunk_payload,
                    },
                    ensure_ascii=False,
                ),
            },
        ],
    )
    translations = payload.get("translations", [])

    normalized: list[dict[str, Any]] = []
    for item in translations:
        chunk_id = int(item["id"])
        translated_content = str(item["translated_content"]).strip()
        normalized.append(
            {
                "id": chunk_id,
                "translated_content": translated_content,
            }
        )

    return {
        "target_language_code": parsed_arguments.target_language_code,
        "target_language_name": parsed_arguments.target_language_name,
        "translations": normalized,
    }
