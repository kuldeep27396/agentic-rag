import json
from typing import List

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from app.agent.graph import PdfRagAgent
from app.core.config import get_settings
from app.core.security import new_session_token
from app.db.repository import repository, to_document_response
from app.schemas.models import (
    ChatMessage,
    ChatMessageRequest,
    ChatSessionCreateRequest,
    ChatSessionResponse,
    DocumentCreateRequest,
    DocumentResponse,
    DocumentStatus,
)
from app.services.llm import llm_client
from app.services.pdf import PdfValidationError, chunk_pages, extract_pdf_pages, validate_pdf_upload
from app.services.qstash import verify_qstash_request
from app.services.rate_limit import rate_limiter
from app.services.redis_state import redis_state
from app.services.storage import storage
from app.services.vector_store import vector_store

router = APIRouter()


@router.get("/health")
def health() -> dict:
    settings = get_settings()
    return {
        "ok": True,
        "environment": settings.environment,
        "missing_production_env": settings.missing_production_env if settings.environment == "production" else [],
    }


@router.post("/v1/documents", response_model=DocumentResponse)
async def create_document(payload: DocumentCreateRequest, request: Request) -> DocumentResponse:
    await rate_limiter.check(request, limit=10)
    try:
        validate_pdf_upload(payload.filename, payload.content_type, payload.size_bytes)
    except PdfValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    token = payload.session_token or new_session_token()
    document = await repository.create_document(payload.filename, str(payload.blob_url), payload.size_bytes, token)
    return to_document_response(document)


@router.get("/v1/documents/{document_id}", response_model=DocumentResponse)
async def get_document(document_id: str) -> DocumentResponse:
    document = await repository.get_document(document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    return to_document_response(document)


@router.post("/v1/ingestion/jobs/{job_id}/run", dependencies=[Depends(verify_qstash_request)])
async def run_ingestion(job_id: str, request: Request) -> DocumentResponse:
    document = await repository.get_document_for_job(job_id)
    if not document:
        raise HTTPException(status_code=404, detail="Ingestion job not found")
    await repository.update_document_status(document.id, DocumentStatus.ingesting)
    try:
        pdf_bytes = await storage.download(str(document.blob_url))
        pages = extract_pdf_pages(pdf_bytes)
        chunks = chunk_pages(document.id, pages)
        await repository.add_chunks(chunks)
        await vector_store.upsert_chunks(chunks)
        await repository.update_document_status(
            document.id,
            DocumentStatus.ready,
            page_count=len(pages),
            chunk_count=len(chunks),
        )
    except Exception as exc:
        await repository.update_document_status(document.id, DocumentStatus.failed, error=str(exc))
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return to_document_response(document)


@router.post("/v1/chat/sessions", response_model=ChatSessionResponse)
async def create_chat_session(payload: ChatSessionCreateRequest) -> ChatSessionResponse:
    try:
        session = await repository.create_chat_session(payload.document_id, payload.session_token)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    return ChatSessionResponse(id=session.id, document_id=session.document_id, expires_at=session.expires_at)


@router.get("/v1/chat/sessions/{session_id}/messages", response_model=List[ChatMessage])
async def list_messages(session_id: str, session_token: str) -> List[ChatMessage]:
    try:
        await repository.require_chat_session(session_id, session_token)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    return await repository.list_messages(session_id)


@router.post("/v1/chat/sessions/{session_id}/messages/stream")
async def stream_message(session_id: str, payload: ChatMessageRequest, request: Request) -> StreamingResponse:
    await rate_limiter.check(request, limit=20)
    try:
        session = await repository.require_chat_session(session_id, payload.session_token)
        document = await repository.require_document_access(session.document_id, payload.session_token)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc

    await repository.add_message(session_id, "user", payload.content)
    agent = PdfRagAgent(repository, vector_store, llm_client)
    answer, citations, suggestions = await agent.run(document.id, document.filename, payload.content, hybrid=payload.hybrid)
    assistant_message = await repository.add_message(session_id, "assistant", answer, citations)

    async def events():
        yield f"data: {json.dumps({'type': 'delta', 'content': answer})}\n\n"
        yield f"data: {assistant_message.model_dump_json()}\n\n"
        if suggestions:
            yield f"data: {json.dumps({'type': 'suggestions', 'items': suggestions})}\n\n"
        yield "event: done\ndata: {}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")


@router.delete("/v1/documents/{document_id}")
async def delete_document(document_id: str, session_token: str) -> DocumentResponse:
    try:
        document = await repository.require_document_access(document_id, session_token)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    await repository.update_document_status(document.id, DocumentStatus.deleted)
    await repository.delete_document_state(document.id)
    await vector_store.delete_collection(document.id)
    return to_document_response(document)
