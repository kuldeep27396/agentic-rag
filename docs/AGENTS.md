# Repository Guidelines

## Project Overview

Session-only PDF RAG app for anonymous browser-based document chat. Vercel monorepo with two apps:

- `apps/web` — Next.js 15 App Router frontend (React 19, TypeScript)
- `apps/api` — FastAPI backend (Python 3.9–3.12)

The old Streamlit demo is gone from the tree and is not part of the active deployment target.

## Commands

```bash
# Backend — venv lives at repo root, API code in apps/api
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt          # forwards to apps/api/requirements.txt
.venv/bin/uvicorn app.main:app --reload --port 8000 # run from apps/api/

# Frontend
npm install
npm run dev:web

# Verification (run from repo root)
.venv/bin/python -m pytest                          # uses pytest.ini: pythonpath=apps/api
.venv/bin/python -m compileall apps/api/app
npm run build:web
npm run lint:web
```

CI (`deploy.yml`) runs `cd apps/api && python -m pytest` and `npm run build:web` on Python 3.9 / Node 20.

## Architecture

- **Entrypoint**: `apps/api/app/main.py` → `create_app()` mounts `app.api.routes:router`.
- **Agent**: `apps/api/app/agent/graph.py` — `PdfRagAgent` retrieves chunks, optionally adds web search, calls OpenRouter.
- **No vector DB**: embeddings are simple hashed word vectors stored in Redis; retrieval is Python cosine similarity. No Neon, Milvus, Zilliz, LangGraph, or direct Firecrawl key.
- **Stack**: Vercel Blob (PDF storage), Upstash Redis (all temp state/chunks/vectors/chat), Upstash QStash (async ingestion), OpenRouter (LLM + optional Firecrawl web search through OpenRouter plugins).
- **Local dev fallbacks**: missing `QSTASH_TOKEN` → ingestion runs synchronously. Missing Redis → in-memory fallback (single-process only).

## Testing

- `pytest.ini` at root sets `pythonpath = apps/api` and `testpaths = apps/api/tests`.
- Tests import `app.*` directly (e.g. `from app.services.pdf import ...`).
- No external services needed — current tests validate pure functions only.
- New test files go in `apps/api/tests/`, named `test_*.py`.

## Environment

Copy `.env.example` to `.env`. Settings loaded via `pydantic-settings` with `extra="ignore"`. Required for production: `OPENROUTER_API_KEY`, `BLOB_READ_WRITE_TOKEN`, `QSTASH_TOKEN`, `QSTASH_CURRENT_SIGNING_KEY`, `QSTASH_NEXT_SIGNING_KEY`, `UPSTASH_REDIS_REST_URL`, `UPSTASH_REDIS_REST_TOKEN`, `SESSION_SECRET`. Local dev works without most of these.

## Style

Python: 4-space indent, `snake_case`, `pathlib.Path`. TypeScript/React: follows Next.js App Router conventions. No comments unless asked.
