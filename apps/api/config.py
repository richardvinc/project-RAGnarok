from __future__ import annotations

from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parents[2]
ENV_FILE = ROOT_DIR / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    root_dir: Path = ROOT_DIR
    documents_dir: Path = root_dir / "documents"
    image_output_dir: Path = root_dir / "images_generated"
    database_url: str | None = Field(
        default=None,
        validation_alias=AliasChoices("DATABASE_URL", "database_url"),
    )
    lm_studio_base_url: str = Field(
        default="http://localhost:1234/v1",
        validation_alias=AliasChoices("LM_STUDIO_BASE_URL", "lm_studio_base_url"),
    )
    llm_api_key: str = Field(
        default="test",
        validation_alias=AliasChoices("LLM_API_KEY", "OPENAI_API_KEY", "llm_api_key"),
    )
    llm_model: str = Field(
        default="google/gemma-4-e4b",
        validation_alias=AliasChoices("LLM_MODEL", "llm_model"),
    )
    embedding_model: str = Field(
        default="text-embedding-google_embeddinggemma-300m-qat",
        validation_alias=AliasChoices("EMBEDDING_MODEL", "embedding_model"),
    )
    backend_host: str = Field(
        default="127.0.0.1",
        validation_alias=AliasChoices("BACKEND_HOST", "backend_host"),
    )
    backend_port: int = Field(
        default=8000,
        validation_alias=AliasChoices("BACKEND_PORT", "backend_port"),
    )
    backend_public_url: str | None = Field(
        default=None,
        validation_alias=AliasChoices("BACKEND_PUBLIC_URL", "backend_public_url"),
    )
    default_retrieval_k: int = Field(
        default=8,
        ge=1,
        le=20,
        validation_alias=AliasChoices("DEFAULT_RETRIEVAL_K", "default_retrieval_k"),
    )
    chunk_size_tokens: int = Field(
        default=300,
        ge=1,
        validation_alias=AliasChoices("CHUNK_SIZE_TOKENS", "chunk_size_tokens"),
    )
    chunk_overlap_tokens: int = Field(
        default=50,
        ge=0,
        validation_alias=AliasChoices("CHUNK_OVERLAP_TOKENS", "chunk_overlap_tokens"),
    )
    token_encoding_name: str = Field(
        default="cl100k_base",
        validation_alias=AliasChoices("TOKEN_ENCODING_NAME", "token_encoding_name"),
    )
    max_tool_rounds: int = Field(
        default=5,
        ge=1,
        le=10,
        validation_alias=AliasChoices("MAX_TOOL_ROUNDS", "max_tool_rounds"),
    )
    frontend_dir: Path = root_dir / "apps" / "web"

    @property
    def resolved_backend_public_url(self) -> str:
        return self.backend_public_url or f"http://{self.backend_host}:{self.backend_port}"


settings = Settings()
