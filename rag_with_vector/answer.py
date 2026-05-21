from __future__ import annotations
import json
from pathlib import Path
from typing import TypedDict

from openai import OpenAI
from retrieval import retrieve
from dotenv import load_dotenv
from generate_image import generate_story_image

load_dotenv()

MODEL_NAME="google/gemma-4-e4b"
client = OpenAI(base_url="http://localhost:1234/v1", api_key="test")
PUBLIC_BACKEND_URL = "http://127.0.0.1:8000"


class AnswerResult(TypedDict):
    response: str
    image_url: str | None

SYSTEM_PROMPT = """You are a helpful assistant and a good story teller. Answer the user's question using ONLY the provided context.

Rules:
- If the context does not contain the answer, say: "I don't know based on the provided context."
- Do not use outside knowledge.
- It's IMPORTANT for the cited sources to be in this format: `[source: {source}#chunk:{chunk_id}]`
"""

# This is the schema configuration you pass to your OpenAI/LM Studio client router
tools = [{
    "type": "function",
    "function": {
        "name": "generate_story_image",
        "description": "Call this tool whenever the user asks to visualize a scene, create an illustration, or see a character/event from 'The Jungle Book'.",
        "parameters": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "A descriptive, vivid text prompt optimized for image generation. Expand on what's happening in the text (e.g., 'Mowgli talking to Baloo under the jungle canopy'). Do not include stylistic formatting words like 'photorealistic', just describe the raw scene elements."
                }
            },
            "required": ["prompt"]
        }
    }
}]


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


def _to_public_image_url(saved_image_path: str) -> str:
    normalized_path = Path(saved_image_path).as_posix().lstrip("./")
    return f"{PUBLIC_BACKEND_URL}/{normalized_path}"


def answer(question: str, *, chunks: list[dict]) -> AnswerResult:
    user_content = f"Question:\n{question}\n\nContext:\n{format_context(chunks)}"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=0,
        messages=messages, # type: ignore
        tools=tools # type: ignore
    )
    response_message = resp.choices[0].message
    generated_image_url: str | None = None

    if response_message and response_message.tool_calls:
        print("Triggered tool call")
        
        # Append Gemma's tool request to history as required by the chat model protocol
        messages.append(response_message) # type: ignore
        
        for tool_call in response_message.tool_calls:
            if tool_call.function.name == "generate_story_image": # type: ignore
                # Safe parse arguments string to dict
                tool_args = json.loads(tool_call.function.arguments) # type: ignore
                print(tool_args)
                target_prompt = tool_args.get("prompt")
                
                print(f"Tool Prompt: '{target_prompt}'")
                print("Handling to generate_story_image function")
                
                # Execute your local diffusers script function
                saved_image_path = generate_story_image(target_prompt)
                generated_image_url = _to_public_image_url(saved_image_path)
                
                # Package tool response for the LLM
                tool_response_content = json.dumps({
                    "status": "success",
                    "saved_at": saved_image_path,
                    "message": "Image successfully generated and saved to disk."
                })
                
                # Append tool results back into the conversation context array
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": "generate_story_image",
                    "content": tool_response_content
                })
        
        print("Generate final message to user")
        # Get final textual response from Gemma confirming completion
        final_response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages, # type: ignore
            temperature=0
        )
        return {
            "response": final_response.choices[0].message.content or "",
            "image_url": generated_image_url,
        }
    
    # If no tool calls were requested, simply return the text answer
    return {
        "response": response_message.content if response_message else "", # type: ignore
        "image_url": None,
    }


def main():
    try:
        user_query = input("\nAsk a question or request a scene (like 'Show me Shere Khan' or 'Give me short story about Mowgli'): ").strip()
        if not user_query or user_query.lower() == 'exit':
            return
            
        print("Getting context from pgvector...")
        chunks = retrieve(user_query)
        

        print("Processing..")
        final_output = answer(user_query, chunks=chunks)
        print(f"Response:\n{final_output}")
        
    except Exception as e:
        print(f"Error encountered: {e}")

if __name__ == "__main__":
    main()