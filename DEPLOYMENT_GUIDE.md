# Deployment Guide

This repository deploys as two Vercel projects from one monorepo.

## Projects

- Web project: `apps/web`, framework `nextjs`
- API project: `apps/api`, framework `fastapi` through Vercel Python runtime

Set the Vercel project root directory separately for each project. Both projects can share the same Git repository and branch.

## Services

The simplified session-only stack uses:

- Vercel Blob for uploaded PDFs.
- Upstash Redis for temporary document metadata, chunks, vectors, chat state, and rate limits.
- Upstash QStash for asynchronous ingestion retries.
- OpenRouter for chat, optional web search through OpenRouter Firecrawl settings, and future hosted embeddings.

No Neon/Postgres, Zilliz/Milvus, Upstash Vector, or direct Firecrawl key is required.

## Environment Variables

Configure these variables in GitHub Secrets and in the matching Vercel projects.

Web project:

```bash
NEXT_PUBLIC_API_BASE_URL=
BLOB_READ_WRITE_TOKEN=
QSTASH_TOKEN=
```

API project:

```bash
ENVIRONMENT=production
WEB_BASE_URL=
API_BASE_URL=
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

## OpenRouter Web Search

For Firecrawl through OpenRouter, enable Firecrawl in OpenRouter Web Search or Plugin settings and accept the terms there. The app does not need `FIRECRAWL_API_KEY`.

## Notes

Vercel serverless request bodies are too small for direct PDF upload through the API, so the browser uploads directly to Vercel Blob through the Next.js client upload route. The API only receives Blob metadata and later downloads the PDF during QStash ingestion.

Redis keys use a TTL based on `RETENTION_DAYS`; by default data expires after 30 days. Browser session storage holds the session token, so losing browser session state loses access.
