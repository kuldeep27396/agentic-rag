from datetime import datetime
from typing import List, Optional

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
    vector_id: Optional[str] = None


class DocumentRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    filename: str = Field(min_length=1, max_length=255)
    blob_url: HttpUrl
    size_bytes: int = Field(gt=0)
    status: DocumentStatus
    session_token_hash: str
    expires_at: datetime
    created_at: datetime = Field(default_factory=lambda: datetime.now(tz=None))
    ingestion_job_id: Optional[str] = None
    page_count: Optional[int] = Field(default=None, ge=0)
    chunk_count: Optional[int] = Field(default=None, ge=0)
    error: Optional[str] = None


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
    embedding: List[float]


class SearchHit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    parent_id: str
    score: float
