# Agentic PDF RAG

**Chat with your PDFs using grounded, page-level citations.**

Session-only PDF RAG app for anonymous browser-based document chat. Upload a PDF, ask questions, get answers backed by exact page references — no account needed.

Built as a Vercel monorepo with a modern Next.js UI and a Python FastAPI backend.

## Features

- **Drag-and-drop PDF upload** with automatic ingestion and chunking
- **Streaming chat** with real-time answer generation
- **Page-level citations** showing exactly where answers come from
- **Hybrid web search** via OpenRouter + Firecrawl for current information
- **Dark/light mode** with system preference detection
- **Privacy-first** — files scoped to a temporary browser session, auto-expire after 30 days
- **No vector database** — lightweight hash-based embeddings stored in Redis

## Architecture

```
apps/web   — Next.js 15 App Router (React 19, Tailwind CSS v4, shadcn/ui)
apps/api   — FastAPI backend (Python 3.9–3.12)
```

| Service | Purpose |
|---|---|
| Vercel Blob | Temporary PDF file storage |
| Upstash Redis | Metadata, chunks, vectors, chat state, rate limits |
| Upstash QStash | Async PDF ingestion with retries |
| OpenRouter | LLM chat + optional Firecrawl web search |

No Neon/Postgres, Zilliz/Milvus, Upstash Vector, LangGraph, or direct Firecrawl API key required.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full request flow and [DEPLOYMENT_KEYS.md](DEPLOYMENT_KEYS.md) for deployment setup.

## Local Development

**Backend** (Python 3.9–3.12):

```bash
cd apps/api
python3.12 -m venv ../../.venv
../../.venv/bin/pip install -r requirements.txt
../../.venv/bin/uvicorn app.main:app --reload --port 8000
```

**Frontend**:

```bash
npm install
npm run dev:web
```

Open `http://localhost:3000`. Missing `QSTASH_TOKEN` triggers synchronous ingestion for local dev. Missing Redis falls back to in-memory storage (single-process only).

## Environment

Copy `.env.example` to `.env`. Local dev works without most variables. See [DEPLOYMENT_KEYS.md](DEPLOYMENT_KEYS.md) for production setup.

## Verification

```bash
.venv/bin/python -m pytest
.venv/bin/python -m compileall apps/api/app
npm run build:web
```

## Privacy Model

- Uploads limited to 25MB PDFs
- All data scoped to a temporary browser session token
- Redis state auto-expires after 30 days (`RETENTION_DAYS`)
- Losing browser session storage loses access
- No global searchable document library

## License

MIT
