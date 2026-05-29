# Project RAGnarok

Project RAGnarok is a portfolio-friendly RAG demo built around The Jungle Book. It now has a cleaner production-style layout, a FastAPI backend, a Next.js frontend inspector, explicit tool routing, and structured decision logs that expose what the LLM decided at each step.

## Architecture

- `apps/api` contains the FastAPI service, retrieval pipeline, tool registry, and LLM response orchestration.
- `apps/web` contains the Next.js inspector UI and proxy routes.
- `scripts/ingest_documents.py` ingests `.md` and `.txt` files from `documents/` into PostgreSQL with pgvector.
- `run_dev.py` validates the environment and starts both services from the repo root.

## Features

- Grounded RAG over documents stored in PostgreSQL + pgvector
- Structured decision logging for embedding, retrieval, prompt construction, tool requests, tool execution, and final response generation
- Tool registry pattern for LLM function calling
- Inspector UI for prompt, context, retrieved chunks, citations, images, and decision traces

## Prerequisites

- Python 3.10+
- PostgreSQL with the `vector` extension enabled
- LM Studio running an embedding model and chat model through its OpenAI-compatible endpoint
- Node.js with npm
- `DATABASE_URL` set in `.env` or the shell environment

## Install

### Python

```powershell
uv sync
```

### Frontend

```powershell
Set-Location apps\web
npm install
Set-Location ..\..
```

## Ingest documents

```powershell
python scripts\ingest_documents.py
```

This reads from `documents/`, chunks the files, embeds them, and stores the chunks in PostgreSQL.

## Run the full stack

```powershell
python run_dev.py
```

The runner:

- validates Python modules and repository entrypoints
- checks that `DATABASE_URL` is present
- warns if LM Studio or PostgreSQL are unreachable
- starts the API on `http://127.0.0.1:8000`
- starts the frontend on `http://localhost:3000`

## API

`POST /rag`

Request body:

```json
{
  "query": "Who is Mowgli's enemy in the story?",
  "k": 8
}
```

Response highlights:

- `retrieved_chunks`
- `formatted_context`
- `final_prompt`
- `llm_response`
- `image_url`
- `decision_log`
- `run_summary`

## Decision Log Inspector

Each query now exposes a step-by-step execution trace in the frontend. The log shows:

- query embedding request
- retrieved chunk ids and counts
- context and prompt construction
- whether the LLM answered directly or requested a tool
- tool arguments and tool outputs
- final response generation

This makes the demo much easier to present as an “agent execution building block” instead of a black-box chatbot.
