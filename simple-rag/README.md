# Simple RAG Inspector

Interactive UI for exploring how RAG works using the simple-rag.py corpus.

## Quick Start

### Prerequisites

- Python 3.9+
- LM Studio running with:
  - `google/embedding-gemma-300m-qat` (embeddings)
  - `google/gemma-4-e4b` (LLM)
- FastAPI: `pip install fastapi uvicorn lmstudio openai`

### Run

**Terminal 1 - Backend:**

```bash
cd simple-rag
uvicorn backend:app --reload
```

**Terminal 2 - Frontend:**

```bash
# Option A: Open index.html directly in browser
# file:///c:/Users/vincr/Desktop/projects/project-RAGnarok/simple-rag/index.html

# Option B: Use simple HTTP server
python -m http.server 8080
# Then open http://localhost:8080
```

## What's in the UI

1. **Query Input** - Enter your question
2. **Query Embedding** - 768-dimensional vector representation
3. **Retrieved Chunks** - Top 5 most relevant documents with similarity scores
4. **Final Prompt** - The complete prompt sent to the LLM
5. **LLM Response** - The model's answer

All sections are expandable to avoid clutter!

## Files

- `backend.py` - FastAPI server with RAG logic
- `index.html` - Interactive frontend
- `simple-rag.py` - Original RAG implementation (reference)

## Modify

**Change corpus:** Edit `corpus_of_documents` in `backend.py`

**Change embedding model:** Update `EMBED_MODEL` in `backend.py`

**Change LLM model:** Update model name in `@app.post("/rag")` endpoint
