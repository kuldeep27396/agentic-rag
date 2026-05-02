from __future__ import annotations
from functools import lru_cache
from typing import Literal

from pydantic import Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    environment: Literal["development", "test", "production"] = "development"
    api_base_url: str = "http://localhost:8000"
    web_base_url: str = "http://localhost:3000"

    openrouter_api_key: str = ""
    openrouter_chat_model: str = "google/gemma-4-31b-it:free"
    openrouter_embedding_model: str = "openai/text-embedding-3-small"

    blob_read_write_token: str = ""
    qstash_token: str = ""
    qstash_current_signing_key: str = ""
    qstash_next_signing_key: str = ""
    upstash_redis_rest_url: str = ""
    upstash_redis_rest_token: str = ""
    session_secret: str = Field(default="", min_length=0)

    max_upload_bytes: int = 25 * 1024 * 1024
    retention_days: int = 30
    parent_chunk_tokens: int = 1500
    child_chunk_tokens: int = 420
    child_overlap_tokens: int = 60
    retrieval_child_top_k: int = 12
    retrieval_parent_top_k: int = 4
    allow_local_embeddings_fallback: bool = True

    @computed_field
    @property
    def missing_production_env(self) -> list[str]:
        required = [
            "openrouter_api_key",
            "blob_read_write_token",
            "qstash_token",
            "qstash_current_signing_key",
            "qstash_next_signing_key",
            "upstash_redis_rest_url",
            "upstash_redis_rest_token",
            "session_secret",
        ]
        return [name.upper() for name in required if not getattr(self, name)]


@lru_cache
def get_settings() -> Settings:
    return Settings()
