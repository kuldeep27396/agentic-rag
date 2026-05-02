# End-to-End Architecture

## Goal

This app lets an anonymous user upload one PDF, wait for temporary ingestion, and chat with that PDF in the same browser session. It does not keep a shared document library and does not require a durable application database.

## Active Stack

- `apps/web`: Next.js App Router frontend deployed as a Vercel web project.
- `apps/api`: FastAPI backend deployed as a Vercel Python project.
- Vercel Blob: stores uploaded PDF files.
- Upstash Redis: stores temporary document metadata, chunks, vectors, chat sessions, messages, and rate-limit counters.
- Upstash QStash: runs ingestion asynchronously with retries.
- OpenRouter: generates answers and provides optional Firecrawl-backed web search.

## Removed Services

These are intentionally not required in v1:

- Neon/Postgres
- Zilliz/Milvus
- Upstash Vector
- direct `FIRECRAWL_API_KEY`
- LangGraph
- Langfuse
- Streamlit

## Request Flow

1. The browser creates a temporary session token and stores it in `sessionStorage`.
2. The user uploads a PDF through the Next.js upload route.
3. Vercel Blob receives the file directly from the browser.
4. The frontend calls `POST /v1/documents` with Blob metadata and the session token.
5. FastAPI validates file metadata, hashes the session token, creates a document record, and stores it in Redis with TTL.
6. The frontend calls the Next.js ingestion route.
7. If `QSTASH_TOKEN` exists, Next.js publishes the job to QStash. In local development, it calls FastAPI directly.
8. QStash calls `POST /v1/ingestion/jobs/{job_id}/run`.
9. FastAPI downloads the PDF from Blob, extracts text, chunks pages, creates lightweight hash embeddings, and stores chunks/vectors in Redis with TTL.
10. The frontend polls `GET /v1/documents/{document_id}` until status is `ready`.
11. The frontend creates a chat session with `POST /v1/chat/sessions`.
12. Chat requests stream through `POST /v1/chat/sessions/{session_id}/messages/stream`.
13. FastAPI checks the session token, retrieves relevant chunks by cosine similarity, calls OpenRouter, stores the assistant response in Redis, and streams the result.

## Data Model

All temporary server-side state is validated with Pydantic.

Redis keys:

- `document:{document_id}`: document metadata, status, Blob URL, expiry, session token hash.
- `ingestion_job:{job_id}`: document mapping for QStash.
- `chunks:{document_id}`: extracted PDF chunks with page ranges and text hashes.
- `vectors:{document_id}`: lightweight embeddings derived from chunks.
- `chat_session:{session_id}`: session access record.
- `messages:{session_id}`: chat history for the current temporary session.
- `rate:{ip}:{window}`: anonymous rate-limit counters.

## Retrieval

The app uses a deliberately simple vector search for v1:

- Extract text with `pypdf`.
- Split into parent and child chunks.
- Build local hashed word vectors.
- Store vectors in Redis.
- Search vectors in Python with cosine similarity.
- Collapse child hits to parent contexts.
- Answer with OpenRouter using only selected PDF context unless web search is needed.

This avoids a vector database while keeping behavior good enough for temporary single-PDF sessions.

## Web Search

Firecrawl is enabled through OpenRouter settings, not through a direct Firecrawl key.

The backend adds the OpenRouter web plugin only when hybrid mode is enabled and the question appears to need current or external information. OpenRouter then uses the configured Firecrawl search engine.

## Privacy And Retention

- The browser session token controls access.
- Server-side Redis data expires after `RETENTION_DAYS`, default `30`.
- Uploaded PDFs should be cleaned up by a scheduled cleanup job or manual delete flow in a later hardening pass.
- Losing browser session storage means losing access to the document.
- The app does not create a global searchable library.

## Deployment Shape

Use two Vercel projects from the same repository:

- Web project root: `apps/web`
- API project root: `apps/api`

The web project needs the API URL and upload/QStash env vars. The API project needs OpenRouter, Blob, QStash, Redis, and session env vars.

