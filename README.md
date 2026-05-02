# Agentic PDF RAG

Session-only PDF RAG application for anonymous browser-based document chat. The active app is a Vercel monorepo with a Next.js UI and a Python FastAPI backend.

## Architecture

- `apps/web`: Next.js App Router UI for PDF upload, ingestion status, chat streaming, and citations.
- `apps/api`: FastAPI backend for document metadata, ingestion jobs, PDF parsing, chunking, temporary retrieval, and chat orchestration.

The simplified v1 stack is:

- Vercel Blob: temporary PDF file storage.
- Upstash Redis: temporary metadata, chunks, vectors, chat state, and rate limits.
- Upstash QStash: async PDF ingestion.
- OpenRouter: chat generation and optional OpenRouter-managed web search.

No Neon/Postgres, Zilliz/Milvus, Upstash Vector, or direct Firecrawl API key is required.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full flow and [DEPLOYMENT_KEYS.md](DEPLOYMENT_KEYS.md) for the exact secrets to add.

## Local Development

Backend, using Python 3.9-3.12:

```bash
cd apps/api
python3.12 -m venv ../../.venv
../../.venv/bin/pip install -r requirements.txt
../../.venv/bin/uvicorn app.main:app --reload --port 8000
```

Frontend:

```bash
npm install
npm run dev:web
```

Open `http://localhost:3000`. If `QSTASH_TOKEN` is not set, the web app calls the ingestion job directly for local development. If Upstash Redis is not configured locally, the API uses an in-memory fallback for tests and single-process development only.

## Environment

Copy `.env.example` to `.env` and configure production secrets in the relevant Vercel projects.

Required production variables:

```bash
ENVIRONMENT=production
WEB_BASE_URL=
API_BASE_URL=
NEXT_PUBLIC_API_BASE_URL=
SESSION_SECRET=

OPENROUTER_API_KEY=
OPENROUTER_CHAT_MODEL=google/gemma-4-31b-it:free
OPENROUTER_EMBEDDING_MODEL=openai/text-embedding-3-small

BLOB_READ_WRITE_TOKEN=

QSTASH_TOKEN=
QSTASH_CURRENT_SIGNING_KEY=
QSTASH_NEXT_SIGNING_KEY=

UPSTASH_REDIS_REST_URL=
UPSTASH_REDIS_REST_TOKEN=
```

## Privacy Model

- Uploads are limited to 25MB PDFs.
- Uploaded PDFs and derived state are scoped to a temporary browser session token.
- Redis state has a TTL based on `RETENTION_DAYS`, defaulting to 30 days.
- Browser session storage holds the token; losing browser session state loses access.
- OpenRouter Firecrawl web search is used only through OpenRouter when hybrid mode decides web context is needed.

The current Vercel Blob client upload SDK exposes public Blob objects for direct browser uploads. The application still scopes document metadata, chat, retrieval, and deletion to the temporary session token, and upload paths use random suffixes. If Vercel private direct uploads are available in your account/SDK version, switch the upload access mode and keep the same API flow.

## Verification

```bash
../../.venv/bin/python -m pytest
../../.venv/bin/python -m compileall apps/api/app
npm run build:web
```

The old Streamlit demo and sample document corpus have been removed from the active repository.
