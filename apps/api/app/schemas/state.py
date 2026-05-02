from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, HttpUrl

from app.schemas.models import DocumentStatus


class ChunkRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    document_id: str
    parent_id: str
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)
    text: str = Field(min_length=1)
    text_hash: str
    token_count: int = Field(gt=0)
    vector_id: str | None = None


class DocumentRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    filename: str = Field(min_length=1, max_length=255)
    blob_url: HttpUrl
    size_bytes: int = Field(gt=0)
    status: DocumentStatus
    session_token_hash: str
    expires_at: datetime
    ingestion_job_id: str | None = None
    page_count: int | None = Field(default=None, ge=0)
    chunk_count: int | None = Field(default=None, ge=0)
    error: str | None = None


class ChatSessionRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    document_id: str
    session_token_hash: str
    expires_at: datetime


class StoredVector(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    document_id: str
    parent_id: str
    embedding: list[float]


class SearchHit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    parent_id: str
    score: float

