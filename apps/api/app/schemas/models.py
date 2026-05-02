from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, HttpUrl


class DocumentStatus(str, Enum):
    uploaded = "uploaded"
    ingesting = "ingesting"
    ready = "ready"
    failed = "failed"
    deleted = "deleted"


class Citation(BaseModel):
    chunk_id: str
    filename: str
    page_start: int | None = None
    page_end: int | None = None
    source_type: Literal["pdf", "web"] = "pdf"
    score: float | None = None
    url: str | None = None


class DocumentCreateRequest(BaseModel):
    filename: str = Field(min_length=1, max_length=255)
    blob_url: HttpUrl
    size_bytes: int = Field(gt=0)
    content_type: str = "application/pdf"
    session_token: str | None = None


class DocumentResponse(BaseModel):
    id: str
    filename: str
    size_bytes: int
    status: DocumentStatus
    page_count: int | None = None
    chunk_count: int | None = None
    error: str | None = None
    expires_at: datetime
    ingestion_job_id: str | None = None


class ChatSessionCreateRequest(BaseModel):
    document_id: str
    session_token: str


class ChatSessionResponse(BaseModel):
    id: str
    document_id: str
    expires_at: datetime


class ChatMessageRequest(BaseModel):
    content: str = Field(min_length=1, max_length=8000)
    session_token: str
    hybrid: bool = True


class ChatMessage(BaseModel):
    id: str
    session_id: str
    role: Literal["user", "assistant"]
    content: str
    citations: list[Citation] = Field(default_factory=list)
    created_at: datetime
    metadata: dict[str, Any] = Field(default_factory=dict)
