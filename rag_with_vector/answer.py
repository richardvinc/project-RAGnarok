from __future__ import annotations

from openai import OpenAI
from retrieval import retrieve


client = OpenAI(base_url="http://localhost:1234/v1", api_key="test")

SYSTEM_PROMPT = """You are a helpful assistant. Answer the user's question using ONLY the provided context.

Rules:
- If the context does not contain the answer, say: "I don't know based on the provided context."
- Do not use outside knowledge.
- Cite sources in this format: [source: {source}#chunk:{chunk_id}]
"""


def format_context(chunks: list[dict]) -> str:
    parts: list[str] = ["--- BEGIN CONTEXT ---"]
    for c in chunks:
        section = c.get("section_path") or ""
        section_part = f' section="{section}"' if section else ""
        parts.append(
            f'\n[chunk_id={c["id"]} source="{c["source"]}"{section_part}]\n{c["content"]}'
        )
    parts.append("\n--- END CONTEXT ---")
    return "\n".join(parts)


def answer(question: str, *, chunks: list[dict]) -> str:
    user_content = f"Question:\n{question}\n\nContext:\n{format_context(chunks)}"
    resp = client.chat.completions.create(
        model="google/gemma-4-e4b",
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
    )
    return resp.choices[0].message.content or ""

def main():
    user_query = "what is vicia and what does it for?"
    chunks = retrieve(user_query)
    
    print(answer(user_query, chunks=chunks))

if __name__ == "__main__":
    main()