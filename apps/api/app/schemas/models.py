from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Literal, Optional

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
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    source_type: Literal["pdf", "web"] = "pdf"
    score: Optional[float] = None
    url: Optional[str] = None


class DocumentCreateRequest(BaseModel):
    filename: str = Field(min_length=1, max_length=255)
    blob_url: HttpUrl
    size_bytes: int = Field(gt=0)
    content_type: str = "application/pdf"
    session_token: Optional[str] = None


class DocumentResponse(BaseModel):
    id: str
    filename: str
    size_bytes: int
    status: DocumentStatus
    page_count: Optional[int] = None
    chunk_count: Optional[int] = None
    error: Optional[str] = None
    expires_at: datetime
    ingestion_job_id: Optional[str] = None


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
    citations: List[Citation] = Field(default_factory=list)
    created_at: datetime
    metadata: Dict[str, Any] = Field(default_factory=dict)
