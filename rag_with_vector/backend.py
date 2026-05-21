"""
FastAPI backend for RAG with Vector database system
Run: uvicorn backend:app --reload
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI

# Import from retrieval and document_ingest modules
from retrieval import embed_query, retrieve
from answer import format_context, SYSTEM_PROMPT

from dotenv import load_dotenv

load_dotenv()

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
client = OpenAI(base_url="http://localhost:1234/v1", api_key="test")

class QueryRequest(BaseModel):
    query: str
    k: int = 8

class ChunkResult(BaseModel):
    id: int
    source: str
    section_path: str
    content: str

class RAGResponse(BaseModel):
    query: str
    query_embedding: list[float]
    retrieved_chunks: list[ChunkResult]
    formatted_context: str
    final_prompt: str
    llm_response: str

@app.post("/rag", response_model=RAGResponse)
async def rag_endpoint(request: QueryRequest):
    """RAG endpoint that returns all intermediate steps"""
    query = request.query
    k = request.k
    
    # Step 1: Get query embedding
    query_embedding = embed_query(query)
    
    # Step 2: Retrieve relevant chunks
    retrieved_chunks_raw = retrieve(query, k=k)
    
    # Step 3: Convert to response format
    chunks_info = [
        ChunkResult(
            id=c["id"],
            source=c["source"],
            section_path=c["section_path"],
            content=c["content"]
        )
        for c in retrieved_chunks_raw
    ]
    
    # Step 4: Format context for prompt
    formatted_context = format_context(retrieved_chunks_raw)
    
    # Step 5: Build the full prompt
    user_content = f"Question:\n{query}\n\nContext:\n{formatted_context}"
    
    final_prompt = f"SYSTEM: {SYSTEM_PROMPT}\n\nUSER: {user_content}"
    
    # Step 6: Get LLM response
    response = client.chat.completions.create(
        model="google/gemma-4-e4b",
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
    )
    
    llm_response = response.choices[0].message.content or ""
    
    return RAGResponse(
        query=query,
        query_embedding=query_embedding,
        retrieved_chunks=chunks_info,
        formatted_context=formatted_context,
        final_prompt=final_prompt,
        llm_response=llm_response
    )

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
