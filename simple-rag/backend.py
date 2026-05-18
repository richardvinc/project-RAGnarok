"""
FastAPI backend for RAG system - exposes all intermediate steps for inspection
Run: uvicorn backend:app --reload
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
from openai import OpenAI
import uvicorn

# Import from simple-rag module
from functions import corpus_of_documents, embedding_model, embed_query, search

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize LLM client
client = OpenAI(base_url="http://localhost:1234/v1")

# Pre-embed all documents
embedded_data_source = []
for chunk in corpus_of_documents:
    results = embedding_model.embed(f"search_document: {chunk}")
    embedding_arr = np.array(results, dtype=np.float32)
    embedded_data_source.append((embedding_arr, chunk))

class QueryRequest(BaseModel):
    query: str
    k: int = 5

class ChunkResult(BaseModel):
    text: str
    similarity: float
    embedding: list[float]

class RAGResponse(BaseModel):
    query: str
    query_embedding: list[float]
    retrieved_chunks: list[ChunkResult]
    final_prompt: str
    llm_response: str

@app.post("/rag", response_model=RAGResponse)
async def rag_endpoint(request: QueryRequest):
    """RAG endpoint that returns all intermediate steps"""
    query = request.query
    k = request.k
    
    # Search for relevant chunks using imported search function
    retrieved_results = search(query, embedded_data_source, k=k)
    
    # Get query embedding
    query_embedding = embed_query(query)
    # Convert to list
    query_embedding_list = query_embedding.tolist() if hasattr(query_embedding, 'tolist') else list(query_embedding)
    
    # Prepare chunks for response (add embeddings)
    chunks_info = []
    for similarity, chunk in retrieved_results:
        # Get embedding for this chunk
        chunk_embedding = embedding_model.embed(f"search_document: {chunk}")
        chunks_info.append(ChunkResult(
            text=chunk,
            similarity=float(similarity),
            embedding=[float(x) for x in chunk_embedding] # type: ignore
        ))
    
    # Build the prompt
    chunks_text = "\n\n".join([c.text for c in chunks_info])
    base_prompt = """You are an AI assistant for RAG. Your task is to understand the user question, and provide an answer using the provided contexts.

Your answers are correct, high-quality, and written by a domain expert. If the provided context does not contain the answer, simply state, "The provided context does not have the answer."

User question: {user_query}

Contexts:
{chunks_information}
"""
    
    final_prompt = base_prompt.format(user_query=query, chunks_information=chunks_text)
    
    # Get LLM response
    response = client.chat.completions.create(
        model="google/gemma-4-e4b",
        temperature=0,
        messages=[
            {"role": "user", "content": final_prompt},
        ],
    )
    
    llm_response = response.choices[0].message.content
    
    return RAGResponse(
        query=query,
        query_embedding=[float(x) for x in query_embedding],
        retrieved_chunks=chunks_info,
        final_prompt=final_prompt,
        llm_response=llm_response  # type: ignore
    )

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
