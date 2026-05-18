# RAG with Vector Database Inspector

Full-stack RAG system using PostgreSQL + pgvector for semantic search with vector embeddings. Inspect every step of the RAG pipeline.

## Prerequisites

- PostgreSQL running with pgvector extension
- LM Studio with `google/gemma-4-e4b` model loaded
- Database schema created with `documents` and `chunks` tables
- Environment variable: `DATABASE_URL=postgresql://user:password@localhost/dbname`
- Python packages: `fastapi uvicorn openai psycopg tiktoken lmstudio`

## Setup

### 1. Ingest Documents (One-time)

```bash
python document_ingest.py
```

This:

- Reads `.md` and `.txt` files from `../documents` folder
- Splits into chunks (300 tokens with 50 token overlap)
- Generates embeddings using LM Studio
- Stores chunks + embeddings in PostgreSQL with pgvector

You can modify `FOLDER_PATH` in `document_ingest.py` to ingest different folders.

### 2. Start Backend

```bash
uvicorn backend:app --reload
```

Backend will be at: `http://127.0.0.1:8000`

### 3. Open Frontend

Open `index.html` in browser or serve:

```bash
python -m http.server 8080
```

Then go to: `http://localhost:8080`

## Architecture

```
User Query
    ↓
[Embedding] → query embedding (768-dim) via LM Studio
    ↓
[Vector Search] → pgvector finds top-k similar chunks
    ↓
[Format Context] → chunks formatted for LLM
    ↓
[LLM Generation] → answer based on context only
    ↓
Show all steps in Inspector UI
```

## Files

- `backend.py` - FastAPI server, imports functions from retrieval.py and main.py
- `index.html` - Interactive inspector UI (Tailwind + Vanilla JS)
- `retrieval.py` - Vector search functions (exported)
- `main.py` - LLM response formatting (exported)
- `document_ingest.py` - Document ingestion script
- Database schema (created by your setup script)

## UI Inspector Sections

1. **Query** - Input your question
2. **Query Embedding** - 768-dimensional vector (first 20 dims shown)
3. **Retrieved Chunks** - Top 8 chunks from vector search with source info
4. **Formatted Context** - How chunks are presented to LLM
5. **Final Prompt** - Complete system + user prompt sent to model
6. **LLM Response** - Model's answer (only uses provided context)

All sections are expandable to keep UI clean.

## Learn

This system demonstrates:

- **Vector embeddings**: Converting text to semantic vectors
- **Similarity search**: pgvector efficient nearest-neighbor search
- **RAG pattern**: Using retrieved context to ground LLM responses
- **Retrieval-augmented generation**: Better answers by providing relevant documents
- **Prompt engineering**: How context affects LLM output

Try different queries to see which documents are retrieved and how the LLM responds!

## Modify

**Change retrieval count:**
Edit `index.html` line with `k: 8` or modify backend request

**Change embedding model:**
Edit `EMBED_MODEL` in `retrieval.py`

**Change LLM model:**
Edit `model="google/gemma-4-e4b"` in `backend.py`

**Different document folder:**
Edit `FOLDER_PATH` in `document_ingest.py`, then re-ingest

## Troubleshooting

- **"No connection to server"** - Make sure backend is running
- **"Database error"** - Check `DATABASE_URL` and PostgreSQL is running
- **"No embeddings"** - Verify LM Studio is running and model is loaded
- **"No chunks retrieved"** - Ingest documents first with `document_ingest.py`
