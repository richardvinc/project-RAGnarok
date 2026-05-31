# Project RAGnarok

A simple, no-framework, Retrieval-Augmented Generation (RAG) app to demonstrate how LLM+RAG+tool calls work. The dataset used is taken from `The Jungle Book` dataset by [Project Gutenberg](https://www.gutenberg.org/ebooks/236).

Everything in this project is develop, run, and tested using local development environment. We use `google/gemma-4-e4b` for LLM model and `text-embedding-google_embeddinggemma-300m-qat` for text-embedding model.

## My develpment environment

This application is tested on this setup:

- Processor: AMD Ryzen 5 7500F
- GPU: NVIDIA GeForce RTX 3060
- RAM: 32 GB

This app barely run with that setup 🤣 and only able to serve some basic query. The image generation also take a while. Mind you, this is only for demonstration.

Why do I tell this? Since pytorch on `pyproject.toml` is accustomed to this GPU. You might have to change/search around for your setup.

## Prerequisites

- **uv** for package manager (more info [here](https://github.com/astral-sh/uv))
- **Python**: Python 3.12.13 (used in this project. other version might also work, but not tested)
- **Node.js / npm**: for the frontend (Next.js)
- LM Studio/Ollama running locally (more info [here](https://lmstudio.ai/download))
- PgVector: vector database on Postgres (more info [here](https://hub.docker.com/r/pgvector/pgvector))

## Quickstart

1. Create and activate a virtual environment:

```bash
uv venv
```

2. Install dependencies:

```bash
uv sync
```

3. Copy the example environment file and edit values:

```bash
copy .env.example .env  # Windows
cp .env.example .env    # macOS / Linux
```

4. Edit `.env` to set `DATABASE_URL` based on local db

## Run

- Prepare the DB

```
import the `schema.sql` to your vector database to create required table.
```

- Ingest documents (do this one time to populate DB):

```bash
python scripts/ingest_documents.py
```

- Start the application using launcher

```bash
python run_dev.py
```

## Caveats

This readme is supposed to only served to be my documentation on how it setup on my development environment. Your experience may vary.
