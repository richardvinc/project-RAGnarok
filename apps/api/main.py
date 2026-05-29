from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .config import settings
from .schemas import QueryRequest, RAGResponse
from .services.response_service import run_rag_pipeline

app = FastAPI(title="RAGnarok API", version="0.1.0")

settings.image_output_dir.mkdir(parents=True, exist_ok=True)
app.mount(
    "/images_generated",
    StaticFiles(directory=str(settings.image_output_dir)),
    name="images_generated",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/rag", response_model=RAGResponse)
async def rag_endpoint(request: QueryRequest) -> RAGResponse:
    return run_rag_pipeline(request.query, k=request.k)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
