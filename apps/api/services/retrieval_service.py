from __future__ import annotations

import json
import re
import uuid
from pathlib import Path

import psycopg
import tiktoken
from openai import OpenAI

from ..config import settings
from ..domain.models import ChunkDraft, IngestedFileResult, RetrievedChunk

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)\s*$")

_client = OpenAI(
    base_url=settings.lm_studio_base_url,
    api_key=settings.llm_api_key,
)


def get_embedding_client() -> OpenAI:
    return _client


def vector_to_pg(vec: list[float]) -> str:
    return "[" + ",".join(f"{x:.6f}" for x in vec) + "]"


def embed_query(query: str) -> list[float]:
    response = _client.embeddings.create(
        model=settings.embedding_model,
        input=f"search_query: {query}",
    )
    return response.data[0].embedding


def embed_document(text: str) -> list[float]:
    response = _client.embeddings.create(
        model=settings.embedding_model,
        input=f"search_document: {text}",
    )
    return response.data[0].embedding


def retrieve_chunks(query: str, *, k: int) -> tuple[list[float], list[RetrievedChunk]]:
    if not settings.database_url:
        raise RuntimeError("DATABASE_URL is not defined")

    query_embedding = embed_query(query)

    with psycopg.connect(settings.database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                  chunks.id,
                  documents.source,
                  chunks.section_path,
                  chunks.content
                FROM chunks
                JOIN documents ON documents.id = chunks.doc_id
                ORDER BY chunks.embedding <=> %s::vector
                LIMIT %s
                """,
                (vector_to_pg(query_embedding), k),
            )
            rows = cur.fetchall()

    chunks = [
        RetrievedChunk(
            id=row[0],
            source=row[1],
            section_path=row[2],
            content=row[3],
        )
        for row in rows
    ]
    return query_embedding, chunks


def build_history_aware_search_query(query: str, history: list[str]) -> str:
    recent_history = [item.strip() for item in history if item.strip()][-6:]
    if not recent_history:
        return query

    history_block = "\n".join(recent_history)
    return f"Conversation history:\n{history_block}\n\nCurrent user request:\n{query}"


def chunk_by_tokens(text: str) -> list[str]:
    encoding = tiktoken.get_encoding(settings.token_encoding_name)
    tokens = encoding.encode(text)

    chunks: list[str] = []
    start = 0
    while start < len(tokens):
        end = min(start + settings.chunk_size_tokens, len(tokens))
        chunks.append(encoding.decode(tokens[start:end]))
        if end >= len(tokens):
            break
        start = max(0, end - settings.chunk_overlap_tokens)
    return chunks


def split_markdown_sections(markdown_text: str) -> list[tuple[str, str]]:
    sections: list[tuple[str, str]] = []
    heading_stack: list[str] = []
    current_lines: list[str] = []
    current_path = "Document"

    def flush() -> None:
        nonlocal current_lines
        text = "\n".join(current_lines).strip()
        if text:
            sections.append((current_path, text))
        current_lines = []

    for line in markdown_text.splitlines():
        match = HEADING_RE.match(line)
        if match:
            flush()
            level = len(match.group(1))
            title = match.group(2).strip()
            heading_stack[:] = heading_stack[: max(0, level - 1)]
            heading_stack.append(title)
            current_path = " > ".join(heading_stack)
            current_lines.append(line)
        else:
            current_lines.append(line)

    flush()
    return sections


def chunk_document(text: str, *, is_markdown: bool) -> list[ChunkDraft]:
    sections = split_markdown_sections(text) if is_markdown else [("Document", text)]
    chunks: list[ChunkDraft] = []

    for section_path, section_text in sections:
        for chunk in chunk_by_tokens(section_text):
            chunks.append(ChunkDraft(section_path=section_path, text=chunk))

    return chunks


def ingest_directory(directory: Path) -> list[IngestedFileResult]:
    if not settings.database_url:
        raise RuntimeError("DATABASE_URL is not defined")

    files = sorted(
        path for path in directory.glob("*") if path.suffix.lower() in {".md", ".txt"}
    )
    if not files:
        raise RuntimeError(f"No .md or .txt files found in {directory.resolve()}")

    ingested_files: list[IngestedFileResult] = []

    with psycopg.connect(settings.database_url) as conn:
        with conn.cursor() as cur:
            for path in files:
                source = str(path)
                text = path.read_text(encoding="utf-8", errors="replace")
                is_markdown = path.suffix.lower() == ".md"

                doc_id = uuid.uuid4()
                cur.execute(
                    """
                    INSERT INTO documents (id, source, title, metadata)
                    VALUES (%s, %s, %s, %s::jsonb)
                    """,
                    (
                        doc_id,
                        source,
                        path.stem,
                        json.dumps({"filetype": path.suffix.lower()}),
                    ),
                )

                chunk_objects = chunk_document(text, is_markdown=is_markdown)
                embeddings = [embed_document(chunk.text) for chunk in chunk_objects]

                for index, (chunk, embedding) in enumerate(zip(chunk_objects, embeddings)):
                    metadata = {
                        "source": source,
                        "chunker": f"token_{settings.chunk_size_tokens}_{settings.chunk_overlap_tokens}",
                        "embedding_model": settings.embedding_model,
                    }
                    cur.execute(
                        """
                        INSERT INTO chunks (doc_id, chunk_index, section_path, content, metadata, embedding)
                        VALUES (%s, %s, %s, %s, %s::jsonb, %s::vector)
                        """,
                        (
                            doc_id,
                            index,
                            chunk.section_path,
                            chunk.text,
                            json.dumps(metadata),
                            vector_to_pg(embedding),
                        ),
                    )

                ingested_files.append(
                    IngestedFileResult(
                        file_name=path.name,
                        chunks_created=len(chunk_objects),
                    )
                )

        conn.commit()

    return ingested_files
