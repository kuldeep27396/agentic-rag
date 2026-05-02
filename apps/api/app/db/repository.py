from uuid import uuid4

from app.core.config import get_settings
from app.core.security import expires_in_days, hash_session_token, utcnow, verify_session_token
from app.schemas.models import ChatMessage, Citation, DocumentResponse, DocumentStatus
from app.schemas.state import ChatSessionRecord, ChunkRecord, DocumentRecord
from app.services.redis_state import redis_state


def _ttl_seconds() -> int:
    return get_settings().retention_days * 24 * 60 * 60


def _document_key(document_id: str) -> str:
    return f"document:{document_id}"


def _job_key(job_id: str) -> str:
    return f"ingestion_job:{job_id}"


def _chunks_key(document_id: str) -> str:
    return f"chunks:{document_id}"


def _chat_session_key(session_id: str) -> str:
    return f"chat_session:{session_id}"


def _messages_key(session_id: str) -> str:
    return f"messages:{session_id}"


class SessionRepository:
    async def create_document(self, filename: str, blob_url: str, size_bytes: int, session_token: str) -> DocumentRecord:
        document_id = str(uuid4())
        job_id = str(uuid4())
        record = DocumentRecord(
            id=document_id,
            filename=filename,
            blob_url=blob_url,
            size_bytes=size_bytes,
            status=DocumentStatus.uploaded,
            session_token_hash=hash_session_token(session_token),
            expires_at=expires_in_days(get_settings().retention_days),
            ingestion_job_id=job_id,
        )
        ttl = _ttl_seconds()
        await redis_state.set_json(_document_key(document_id), record.model_dump(mode="json"), ttl)
        await redis_state.set_json(_job_key(job_id), {"document_id": document_id}, ttl)
        return record

    async def get_document(self, document_id: str) -> DocumentRecord | None:
        data = await redis_state.get_json(_document_key(document_id))
        return DocumentRecord.model_validate(data) if data else None

    async def require_document_access(self, document_id: str, session_token: str) -> DocumentRecord:
        document = await self.get_document(document_id)
        if not document or document.status == DocumentStatus.deleted:
            raise KeyError("Document not found")
        if not verify_session_token(session_token, document.session_token_hash):
            raise PermissionError("Invalid session token")
        if document.expires_at <= utcnow():
            raise PermissionError("Document has expired")
        return document

    async def get_document_for_job(self, job_id: str) -> DocumentRecord | None:
        job = await redis_state.get_json(_job_key(job_id))
        return await self.get_document(job["document_id"]) if job else None

    async def update_document_status(
        self,
        document_id: str,
        status: DocumentStatus,
        *,
        page_count: int | None = None,
        chunk_count: int | None = None,
        error: str | None = None,
    ) -> DocumentRecord:
        document = await self.get_document(document_id)
        if not document:
            raise KeyError("Document not found")
        updated = document.model_copy(
            update={
                "status": status,
                "page_count": page_count if page_count is not None else document.page_count,
                "chunk_count": chunk_count if chunk_count is not None else document.chunk_count,
                "error": error,
            }
        )
        await redis_state.set_json(_document_key(document_id), updated.model_dump(mode="json"), _ttl_seconds())
        return updated

    async def add_chunks(self, chunks: list[ChunkRecord]) -> None:
        if not chunks:
            return
        await redis_state.set_json(
            _chunks_key(chunks[0].document_id),
            [chunk.model_dump(mode="json") for chunk in chunks],
            _ttl_seconds(),
        )

    async def get_parent_chunks(self, document_id: str, parent_ids: list[str]) -> list[ChunkRecord]:
        data = await redis_state.get_json(_chunks_key(document_id)) or []
        wanted = set(parent_ids)
        return [ChunkRecord.model_validate(chunk) for chunk in data if chunk["parent_id"] in wanted]

    async def create_chat_session(self, document_id: str, session_token: str) -> ChatSessionRecord:
        document = await self.require_document_access(document_id, session_token)
        session = ChatSessionRecord(
            id=str(uuid4()),
            document_id=document.id,
            session_token_hash=hash_session_token(session_token),
            expires_at=document.expires_at,
        )
        await redis_state.set_json(_chat_session_key(session.id), session.model_dump(mode="json"), _ttl_seconds())
        await redis_state.set_json(_messages_key(session.id), [], _ttl_seconds())
        return session

    async def require_chat_session(self, session_id: str, session_token: str) -> ChatSessionRecord:
        data = await redis_state.get_json(_chat_session_key(session_id))
        if not data:
            raise KeyError("Chat session not found")
        session = ChatSessionRecord.model_validate(data)
        if not verify_session_token(session_token, session.session_token_hash):
            raise PermissionError("Invalid session token")
        if session.expires_at <= utcnow():
            raise PermissionError("Chat session has expired")
        return session

    async def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        citations: list[Citation] | None = None,
        metadata: dict | None = None,
    ) -> ChatMessage:
        message = ChatMessage(
            id=str(uuid4()),
            session_id=session_id,
            role=role,  # type: ignore[arg-type]
            content=content,
            citations=citations or [],
            created_at=utcnow(),
            metadata=metadata or {},
        )
        messages = await self.list_messages(session_id)
        messages.append(message)
        await redis_state.set_json(
            _messages_key(session_id),
            [item.model_dump(mode="json") for item in messages],
            _ttl_seconds(),
        )
        return message

    async def list_messages(self, session_id: str) -> list[ChatMessage]:
        data = await redis_state.get_json(_messages_key(session_id)) or []
        return [ChatMessage.model_validate(item) for item in data]

    async def delete_document_state(self, document_id: str) -> None:
        await redis_state.delete(_document_key(document_id), _chunks_key(document_id), f"vectors:{document_id}")


repository = SessionRepository()


def to_document_response(document: DocumentRecord) -> DocumentResponse:
    return DocumentResponse(
        id=document.id,
        filename=document.filename,
        size_bytes=document.size_bytes,
        status=document.status,
        page_count=document.page_count,
        chunk_count=document.chunk_count,
        error=document.error,
        expires_at=document.expires_at,
        ingestion_job_id=document.ingestion_job_id,
    )
