# Deployment Keys

Add these values in two places:

1. GitHub repository secrets, so CI can validate builds without leaking values.
2. Vercel project environment variables, so deployed apps can run.

GitHub Secrets alone are not enough for runtime.

## Web Project

Vercel project root: `apps/web`

Required:

```bash
NEXT_PUBLIC_API_BASE_URL=
BLOB_READ_WRITE_TOKEN=
QSTASH_TOKEN=
```

Use:

- `NEXT_PUBLIC_API_BASE_URL`: deployed FastAPI API URL, for example `https://your-api.vercel.app`.
- `BLOB_READ_WRITE_TOKEN`: from Vercel Blob.
- `QSTASH_TOKEN`: from Upstash QStash.

## API Project

Vercel project root: `apps/api`

Required:

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

Use:

- `WEB_BASE_URL`: deployed Next.js URL, for example `https://your-web.vercel.app`.
- `API_BASE_URL`: deployed FastAPI URL, for example `https://your-api.vercel.app`.
- `SESSION_SECRET`: random 32-byte or longer secret.
- `OPENROUTER_API_KEY`: OpenRouter API key.
- `BLOB_READ_WRITE_TOKEN`: Vercel Blob token.
- `QSTASH_*`: Upstash QStash token and signing keys.
- `UPSTASH_REDIS_*`: Upstash Redis REST URL and token.

Generate `SESSION_SECRET` locally:

```bash
openssl rand -hex 32
```

## OpenRouter Settings

Recommended:

- Default model: `google/gemma-4-31b-it:free`
- Provider sort: tool-call quality first
- Web Search engine: Firecrawl
- Firecrawl terms: accepted

PDF Inputs can stay enabled in OpenRouter. The app does not use OpenRouter PDF Inputs in v1, so Mistral OCR settings should not affect normal app usage.

## Not Needed

Do not add these for the simplified v1:

```bash
FIRECRAWL_API_KEY=
DATABASE_URL=
ZILLIZ_URI=
ZILLIZ_TOKEN=
UPSTASH_VECTOR_REST_URL=
UPSTASH_VECTOR_REST_TOKEN=
LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=
```

## Deployment Order

1. Create Upstash Redis and copy REST URL/token.
2. Create Upstash QStash and copy token/current signing key/next signing key.
3. Create Vercel Blob store and copy `BLOB_READ_WRITE_TOKEN`.
4. Create OpenRouter key and enable Firecrawl web search in OpenRouter settings.
5. Create the Vercel API project with root `apps/api`.
6. Add API env vars and deploy.
7. Copy the API deployment URL.
8. Create the Vercel web project with root `apps/web`.
9. Add web env vars, including `NEXT_PUBLIC_API_BASE_URL`.
10. Add `WEB_BASE_URL` to the API project once the web URL exists and redeploy API if needed.

