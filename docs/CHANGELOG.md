# Changelog

## 2026-05-02

- Replaced the original Streamlit educational demo with a Vercel monorepo.
- Added `apps/web` Next.js UI for PDF upload, ingestion status, chat streaming, and citations.
- Added `apps/api` FastAPI backend for temporary session-only PDF RAG.
- Simplified storage to Vercel Blob plus Upstash Redis and QStash.
- Removed Neon/Postgres, Zilliz/Milvus, Upstash Vector, direct Firecrawl keys, LangGraph, and Langfuse from the active app.
- Added OpenRouter Gemma 4 31B free as the default chat model.
- Added OpenRouter Firecrawl web search support through OpenRouter plugin settings.
- Added Pydantic models for API payloads and Redis-backed state records.
- Removed legacy Streamlit code, legacy sample documents, and obsolete Milvus/database scripts.

