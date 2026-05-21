from __future__ import annotations

import os
import psycopg
from openai import OpenAI

from dotenv import load_dotenv

load_dotenv()

client = OpenAI(base_url="http://localhost:1234/v1", api_key="test")
EMBED_MODEL = "text-embedding-3-small"


def vector_to_pg(vec: list[float]) -> str:
    return "[" + ",".join(f"{x:.6f}" for x in vec) + "]"


def embed_query(q: str) -> list[float]:
    resp = client.embeddings.create(model=EMBED_MODEL, input=q)
    return resp.data[0].embedding


def retrieve(q: str, *, k: int = 8) -> list[dict]:
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise Exception("DATABASE_URL is not defined")
    
    q_vec = embed_query(q)

    with psycopg.connect(db_url) as conn:
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
                (vector_to_pg(q_vec), k),
            )
            rows = cur.fetchall()

    return [
        {"id": r[0], "source": r[1], "section_path": r[2], "content": r[3]}
        for r in rows
    ]