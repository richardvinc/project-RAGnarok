from __future__ import annotations

import json
import os
import re
import uuid
from pathlib import Path

import psycopg
import tiktoken
import lmstudio as lms

EMBED_MODEL = "text-embedding-google_embeddinggemma-300m-qat"
ENCODING_NAME = "cl100k_base"

PROJECT_ROOT = Path(__file__).parent.parent 
FOLDER_PATH = PROJECT_ROOT / "documents"

embedding_model = lms.embedding_model(EMBED_MODEL)

HEADING_RE = re.compile(r"^(#{1,6})\\s+(.*)\\s*$")


def vector_to_pg(vec: list[float]) -> str:
    # pgvector text format: [1,2,3]
    return "[" + ",".join(f"{x:.6f}" for x in vec) + "]"


def chunk_by_tokens(
    text: str,
    *,
    max_tokens: int = 300,
    overlap_tokens: int = 50,
    encoding_name: str = ENCODING_NAME,
) -> list[str]:
    enc = tiktoken.get_encoding(encoding_name)
    tokens = enc.encode(text)

    chunks: list[str] = []
    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunks.append(enc.decode(tokens[start:end]))
        if end >= len(tokens):
            break
        start = max(0, end - overlap_tokens)
    return chunks


def split_markdown_sections(md: str) -> list[tuple[str, str]]:
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

    for line in md.splitlines():
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


def chunk_document(text: str, *, is_markdown: bool) -> list[dict]:
    chunks: list[dict] = []
    if is_markdown:
        sections = split_markdown_sections(text)
    else:
        sections = [("Document", text)]

    for section_path, section_text in sections:
        for chunk in chunk_by_tokens(section_text):
            chunks.append({"section_path": section_path, "text": chunk})
    return chunks


def embed_texts(texts: list[str]) -> list[list[float]]:
    embeddings = []
    for text in texts:
        result = embedding_model.embed(f"search_document: {text}")
        embeddings.append(result)
    return embeddings


def ingest_directory(directory: Path) -> None:
    db_url = os.environ["DATABASE_URL"]
    files = sorted([p for p in directory.glob("*") if p.suffix.lower() in {".md", ".txt"}])

    if not files:
        raise RuntimeError(f"No .md/.txt files found in {directory.resolve()}")

    with psycopg.connect(db_url) as conn:
        with conn.cursor() as cur:
            for path in files:
                source = str(path)
                text = path.read_text(encoding="utf-8", errors="replace")
                is_markdown = path.suffix.lower() == ".md"

                doc_id = uuid.uuid4()
                cur.execute(
                    "INSERT INTO documents (id, source, title, metadata) VALUES (%s, %s, %s, %s::jsonb)",
                    (doc_id, source, path.stem, json.dumps({"filetype": path.suffix.lower()})),
                )

                chunk_objs = chunk_document(text, is_markdown=is_markdown)
                chunk_texts = [c["text"] for c in chunk_objs]

                embeddings = embed_texts(chunk_texts)

                for idx, (c, emb) in enumerate(zip(chunk_objs, embeddings)):
                    metadata = {"source": source, "chunker": "token_300_50", "embedding_model": EMBED_MODEL}
                    cur.execute(
                        """
                        INSERT INTO chunks (doc_id, chunk_index, section_path, content, metadata, embedding)
                        VALUES (%s, %s, %s, %s, %s::jsonb, %s::vector)
                        """,
                        (
                            doc_id,
                            idx,
                            c["section_path"],
                            c["text"],
                            json.dumps(metadata),
                            vector_to_pg(emb),
                        ),
                    )

                print(f"Ingested {path.name}: {len(chunk_objs)} chunks")

        conn.commit()


if __name__ == "__main__":
    ingest_directory(FOLDER_PATH)