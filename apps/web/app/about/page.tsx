import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Agentic PDF RAG — Design & Architecture",
  description: "High-level design, low-level design, tech stack, and scaling details for the Agentic PDF RAG system.",
};

export default function AboutPage() {
  return (
    <div className="min-h-dvh bg-background text-foreground">
      <div className="mx-auto max-w-4xl px-6 py-16">
        <div className="mb-16 text-center">
          <h1 className="text-4xl font-bold tracking-tight">Agentic PDF RAG</h1>
          <p className="mt-3 text-lg text-muted-foreground">
            Design & Architecture Documentation
          </p>
          <div className="mt-4 inline-flex items-center gap-2 rounded-full bg-primary/10 px-4 py-1.5 text-sm text-primary">
            <span className="h-2 w-2 rounded-full bg-primary animate-pulse" />
            Production-Ready System
          </div>
        </div>

        <nav className="mb-16 flex flex-wrap justify-center gap-3">
          {["Overview", "High-Level Design", "Low-Level Design", "Tech Stack", "Data Model", "Retrieval Pipeline", "Security & Privacy", "Scaling & Impact", "Use Cases", "Deployment"].map((s) => (
            <a key={s} href={`#${s.toLowerCase().replace(/ & /g, "-").replace(/ /g, "-")}`} className="rounded-full border px-3.5 py-1.5 text-sm text-muted-foreground transition-colors hover:bg-accent hover:text-accent-foreground">
              {s}
            </a>
          ))}
        </nav>

        <div className="space-y-20">
          {/* Overview */}
          <section id="overview">
            <h2 className="mb-6 text-2xl font-semibold tracking-tight">Overview</h2>
            <div className="space-y-4 text-sm leading-relaxed text-muted-foreground">
              <p>
                <strong className="text-foreground">Agentic PDF RAG</strong> is a temporary-session document chat system that lets users upload a PDF, automatically ingest and chunk it, then ask questions with grounded citations — all within a single browser session. No accounts, no shared library, no persistent database.
              </p>
              <p>
                The system is designed as a lean, serverless-first architecture using two Vercel projects (frontend + API), with ephemeral state stored in Upstash Redis. It combines traditional retrieval-augmented generation with optional web search, delivering accurate answers with page-level source attribution.
              </p>
            </div>
          </section>

          {/* HLD */}
          <section id="high-level-design">
            <h2 className="mb-6 text-2xl font-semibold tracking-tight">High-Level Design</h2>
            <div className="space-y-6">
              <div className="rounded-2xl border bg-card p-6">
                <h3 className="mb-4 text-base font-semibold">System Architecture</h3>
                <pre className="overflow-x-auto rounded-xl bg-muted p-4 text-xs leading-relaxed font-mono">
{`┌──────────────────────────────────────────────────────┐
│                    Browser (Client)                     │
│  ┌──────────────────────────────────────────────────┐  │
│  │  Next.js App (React 19 + TypeScript + Tailwind)  │  │
│  │  • Upload → Vercel Blob (direct)                 │  │
│  │  • Session token → sessionStorage                │  │
│  │  • SSE streaming → chat responses                │  │
│  └────────────────┬─────────────────┬───────────────┘  │
└───────────────────┼─────────────────┼───────────────────┘
                    │ REST API        │ SSE
┌───────────────────▼─────────────────▼───────────────────┐
│                FastAPI Backend (Python)                  │
│  ┌──────────┐  ┌──────────┐  ┌───────────────────────┐ │
│  │ REST API │  │ Agent    │  │ LLM Client            │ │
│  │ Routes   │  │ Pipeline │  │ (OpenRouter + fallback)│ │
│  └────┬─────┘  └────┬─────┘  └───────────┬───────────┘ │
│       │              │                     │             │
│  ┌────▼──────────────▼─────────────────────▼─────────┐  │
│  │           Service Layer                            │  │
│  │  ┌─────────┐ ┌──────────┐ ┌────────┐ ┌──────────┐│  │
│  │  │ Redis   │ │ Vector   │ │ PDF    │ │ Rate     ││  │
│  │  │ State   │ │ Store    │ │ Parser │ │ Limiter  ││  │
│  │  └────┬────┘ └────┬─────┘ └────────┘ └──────────┘│  │
│  └───────┼───────────┼───────────────────────────────┘  │
└──────────┼───────────┼──────────────────────────────────┘
           │           │
    ┌──────▼──────┐  ┌─▼──────────┐
    │ Upstash     │  │ Zilliz /   │
    │ Redis       │  │ Milvus     │
    │ (or in-mem) │  │ (or Redis) │
    └─────────────┘  └────────────┘`}
                </pre>
              </div>

              <div className="rounded-2xl border bg-card p-6">
                <h3 className="mb-4 text-base font-semibold">Request Flow (13 Steps)</h3>
                <ol className="space-y-2 text-sm leading-relaxed text-muted-foreground">
                  {[
                    ["1.", "Browser creates a temporary session token, stored in sessionStorage."],
                    ["2.", "User uploads a PDF through the Next.js upload route."],
                    ["3.", "Vercel Blob receives the file directly from the browser."],
                    ["4.", "Frontend calls POST /v1/documents with Blob metadata and session token."],
                    ["5.", "FastAPI validates file metadata, hashes session token, creates document record in Redis with TTL."],
                    ["6.", "Frontend calls the Next.js ingestion route."],
                    ["7.", "If QSTASH_TOKEN exists, Next.js publishes to QStash. Otherwise calls FastAPI directly."],
                    ["8.", "QStash calls POST /v1/ingestion/jobs/{id}/run (with retries)."],
                    ["9.", "FastAPI downloads PDF from Blob, extracts text, chunks pages, creates embeddings, stores in vector store."],
                    ["10.", "Frontend polls GET /v1/documents/{id} until status is ready."],
                    ["11.", "Frontend creates a chat session with POST /v1/chat/sessions."],
                    ["12.", "Chat requests stream through POST /v1/chat/sessions/{id}/messages/stream."],
                    ["13.", "FastAPI retrieves relevant chunks via vector search, calls OpenRouter, streams the answer with citations."],
                  ].map(([num, text]) => (
                    <li key={num} className="flex gap-3">
                      <span className="shrink-0 font-mono font-semibold text-primary">{num}</span>
                      <span>{text}</span>
                    </li>
                  ))}
                </ol>
              </div>
            </div>
          </section>

          {/* LLD */}
          <section id="low-level-design">
            <h2 className="mb-6 text-2xl font-semibold tracking-tight">Low-Level Design</h2>
            <div className="space-y-6">
              <div className="rounded-2xl border bg-card p-6">
                <h3 className="mb-4 text-base font-semibold">Backend Modules</h3>
                <div className="overflow-x-auto">
                  <table className="w-full text-xs">
                    <thead>
                      <tr className="border-b text-left">
                        <th className="pb-2 pr-4 font-semibold">Module</th>
                        <th className="pb-2 pr-4 font-semibold">File</th>
                        <th className="pb-2 font-semibold">Responsibility</th>
                      </tr>
                    </thead>
                    <tbody className="font-mono text-muted-foreground">
                      {[
                        ["API Routes", "api/routes.py", "REST endpoints: CRUD for documents, chat sessions, streaming messages, ingestion jobs"],
                        ["Agent", "agent/graph.py", "RAG pipeline: retrieve chunks → build context → call LLM → return answer + citations + suggestions"],
                        ["LLM Client", "services/llm.py", "OpenRouter integration with model fallback chain, response parsing, suggestion extraction"],
                        ["Vector Store", "services/vector_store.py", "Shared Milvus collection with document_id filtering, fallback to Redis-based cosine search"],
                        ["Redis State", "services/redis_state.py", "Upstash Redis REST client with in-memory fallback for local development"],
                        ["Repository", "db/repository.py", "Data access layer: documents, chat sessions, messages, chunks — all with TTL"],
                        ["PDF Parser", "services/pdf.py", "PDF validation, text extraction (pypdf), parent/child chunking with configurable token sizes"],
                        ["Rate Limiter", "services/rate_limit.py", "IP-based sliding window rate limiting via Redis INCR + EXPIRE"],
                        ["Security", "core/security.py", "HMAC-based session token hashing and verification, expiry management"],
                        ["Config", "core/config.py", "Pydantic Settings with env file support, production readiness checks"],
                        ["Blob Storage", "services/storage.py", "PDF download from Vercel Blob URLs"],
                        ["QStash", "services/qstash.py", "Request signature verification for QStash webhooks"],
                      ].map(([name, file, desc]) => (
                        <tr key={name} className="border-b border-border/50">
                          <td className="py-2.5 pr-4 font-semibold text-foreground font-sans">{name}</td>
                          <td className="py-2.5 pr-4 text-primary">{file}</td>
                          <td className="py-2.5 font-sans">{desc}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>

              <div className="rounded-2xl border bg-card p-6">
                <h3 className="mb-4 text-base font-semibold">Frontend Architecture</h3>
                <div className="overflow-x-auto">
                  <table className="w-full text-xs">
                    <thead>
                      <tr className="border-b text-left">
                        <th className="pb-2 pr-4 font-semibold">Component</th>
                        <th className="pb-2 font-semibold">Responsibility</th>
                      </tr>
                    </thead>
                    <tbody className="font-mono text-muted-foreground">
                      {[
                        ["App.tsx", "Root component: session state, upload flow, chat UI, citations panel, suggestions"],
                        ["MarkdownRenderer.tsx", "Renders LLM output as rich markdown (headers, lists, code, tables, blockquotes)"],
                        ["lib/api.ts", "API client: document CRUD, chat session creation, SSE streaming with suggestion parsing"],
                        ["lib/env.ts", "Environment config: API base URL resolution, upload limits"],
                        ["app/api/upload/route.ts", "Next.js API route: Vercel Blob upload handler with PDF validation"],
                        ["app/api/ingest/route.ts", "Next.js API route: QStash job enqueue (or direct dev mode)"],
                      ].map(([name, desc]) => (
                        <tr key={name} className="border-b border-border/50">
                          <td className="py-2.5 pr-4 font-semibold text-foreground font-sans">{name}</td>
                          <td className="py-2.5 font-sans">{desc}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          </section>

          {/* Tech Stack */}
          <section id="tech-stack">
            <h2 className="mb-6 text-2xl font-semibold tracking-tight">Tech Stack</h2>
            <div className="grid gap-4 sm:grid-cols-2">
              {[
                { title: "Frontend", color: "bg-blue-500/10 text-blue-600 dark:text-blue-400", items: ["Next.js 15 (App Router)", "React 19", "TypeScript", "Tailwind CSS v4", "Geist Fonts", "react-markdown + remark-gfm", "Lucide Icons"] },
                { title: "Backend", color: "bg-emerald-500/10 text-emerald-600 dark:text-emerald-400", items: ["FastAPI (Python 3.12)", "Pydantic v2 + Settings", "httpx (async HTTP)", "pypdf (PDF extraction)", "uvicorn (ASGI server)"] },
                { title: "Infrastructure", color: "bg-purple-500/10 text-purple-600 dark:text-purple-400", items: ["Vercel (serverless deploy)", "Vercel Blob (PDF storage)", "Upstash Redis (ephemeral state)", "Upstash QStash (job queue)", "Zilliz / Milvus (vector DB)"] },
                { title: "AI / LLM", color: "bg-amber-500/10 text-amber-600 dark:text-amber-400", items: ["OpenRouter (model gateway)", "Multiple free model fallbacks", "Gemma 4, Llama 3.3, Nemotron", "OpenAI text-embedding-3-small", "Optional Firecrawl web search"] },
              ].map(({ title, color, items }) => (
                <div key={title} className="rounded-2xl border bg-card p-5">
                  <div className="mb-3 flex items-center gap-2">
                    <span className={`rounded-md px-2 py-0.5 text-xs font-semibold ${color}`}>{title}</span>
                  </div>
                  <ul className="space-y-1.5 text-sm text-muted-foreground">
                    {items.map((item) => (
                      <li key={item} className="flex items-center gap-2">
                        <span className="h-1 w-1 rounded-full bg-muted-foreground/40" />
                        {item}
                      </li>
                    ))}
                  </ul>
                </div>
              ))}
            </div>
          </section>

          {/* Data Model */}
          <section id="data-model">
            <h2 className="mb-6 text-2xl font-semibold tracking-tight">Data Model</h2>
            <div className="rounded-2xl border bg-card p-6">
              <p className="mb-4 text-sm text-muted-foreground">All state is ephemeral, stored in Redis with configurable TTL (default 30 days). No persistent database.</p>
              <div className="overflow-x-auto">
                <table className="w-full text-xs">
                  <thead>
                    <tr className="border-b text-left">
                      <th className="pb-2 pr-4 font-semibold">Redis Key Pattern</th>
                      <th className="pb-2 pr-4 font-semibold">Type</th>
                      <th className="pb-2 font-semibold">Contents</th>
                    </tr>
                  </thead>
                  <tbody className="font-mono text-muted-foreground">
                    {[
                      ["document:{id}", "JSON", "Document metadata, status, Blob URL, expiry, session token hash"],
                      ["ingestion_job:{id}", "JSON", "Job → document ID mapping for QStash webhook lookup"],
                      ["chunks:{doc_id}", "JSON array", "Extracted PDF chunks with page ranges, parent IDs, text hashes"],
                      ["vectors:{doc_id}", "JSON array", "Chunk embeddings (fallback mode when Milvus unavailable)"],
                      ["chat_session:{id}", "JSON", "Session access record with document reference and token hash"],
                      ["messages:{session_id}", "JSON array", "Chat history (user + assistant messages with citations)"],
                      ["rate:{ip}:{window}", "counter", "IP-based anonymous rate limit counters"],
                    ].map(([key, type_, contents]) => (
                      <tr key={key} className="border-b border-border/50">
                        <td className="py-2.5 pr-4 text-primary">{key}</td>
                        <td className="py-2.5 pr-4">{type_}</td>
                        <td className="py-2.5 font-sans">{contents}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </section>

          {/* Retrieval Pipeline */}
          <section id="retrieval-pipeline">
            <h2 className="mb-6 text-2xl font-semibold tracking-tight">Retrieval Pipeline</h2>
            <div className="space-y-6">
              <div className="rounded-2xl border bg-card p-6">
                <h3 className="mb-4 text-base font-semibold">Chunking Strategy</h3>
                <div className="space-y-3 text-sm text-muted-foreground">
                  <p>Documents are chunked using a <strong className="text-foreground">two-tier parent-child approach</strong>:</p>
                  <ul className="ml-4 space-y-1.5 list-disc">
                    <li><strong className="text-foreground">Parent chunks</strong> (~1,500 tokens): Larger context windows that preserve semantic coherence. Returned as context to the LLM.</li>
                    <li><strong className="text-foreground">Child chunks</strong> (~420 tokens, 60 token overlap): Smaller, overlapping windows used for precise vector matching. Higher overlap ensures no context falls between chunks.</li>
                    <li>Child hits are <strong className="text-foreground">collapsed to unique parent IDs</strong>, and top-k parents are retrieved for the LLM context.</li>
                  </ul>
                </div>
              </div>

              <div className="rounded-2xl border bg-card p-6">
                <h3 className="mb-4 text-base font-semibold">Embedding & Search</h3>
                <div className="space-y-3 text-sm text-muted-foreground">
                  <div className="grid gap-4 sm:grid-cols-2">
                    <div className="rounded-xl bg-muted/50 p-4">
                      <p className="mb-2 font-semibold text-foreground">Production Mode</p>
                      <ul className="space-y-1 list-disc ml-4">
                        <li>OpenAI text-embedding-3-small (1536 dims)</li>
                        <li>Zilliz/Milvus shared collection</li>
                        <li>Document-scoped filtered search</li>
                        <li>COSINE metric with AUTOINDEX</li>
                      </ul>
                    </div>
                    <div className="rounded-xl bg-muted/50 p-4">
                      <p className="mb-2 font-semibold text-foreground">Development Fallback</p>
                      <ul className="space-y-1 list-disc ml-4">
                        <li>Hash-based word vectors (256 dims)</li>
                        <li>Stored in Redis as JSON</li>
                        <li>Python-side cosine similarity</li>
                        <li>No external vector DB needed</li>
                      </ul>
                    </div>
                  </div>
                </div>
              </div>

              <div className="rounded-2xl border bg-card p-6">
                <h3 className="mb-4 text-base font-semibold">LLM Generation</h3>
                <div className="space-y-3 text-sm text-muted-foreground">
                  <ul className="space-y-2">
                    <li className="flex gap-2"><span className="text-primary font-semibold">1.</span> Retrieved parent contexts are formatted with citation labels and page ranges.</li>
                    <li className="flex gap-2"><span className="text-primary font-semibold">2.</span> The LLM is instructed to answer using only PDF context, with markdown formatting.</li>
                    <li className="flex gap-2"><span className="text-primary font-semibold">3.</span> For questions needing current info, OpenRouter&apos;s Firecrawl web search plugin is enabled.</li>
                    <li className="flex gap-2"><span className="text-primary font-semibold">4.</span> The LLM generates 3 follow-up suggestion questions appended as a parsed suggestions block.</li>
                    <li className="flex gap-2"><span className="text-primary font-semibold">5.</span> On rate limit or error, the system automatically cascades through 7+ free model fallbacks.</li>
                  </ul>
                </div>
              </div>
            </div>
          </section>

          {/* Security */}
          <section id="security-and-privacy">
            <h2 className="mb-6 text-2xl font-semibold tracking-tight">Security & Privacy</h2>
            <div className="grid gap-4 sm:grid-cols-2">
              {[
                { title: "Session Isolation", desc: "Each browser session gets a unique HMAC-signed token. Documents and chats are scoped to that token — no cross-session access." },
                { title: "Token Hashing", desc: "Session tokens are never stored in plaintext. HMAC-SHA256 hashes are used for verification, with a configurable secret key." },
                { title: "Automatic Expiry", desc: "All Redis data has TTL-based expiry (default 30 days). Documents, chunks, vectors, sessions, and messages auto-purge." },
                { title: "No Persistent Storage", desc: "No accounts, no shared library, no durable database. Closing the browser loses access. PDFs can be deleted on demand." },
                { title: "Rate Limiting", desc: "IP-based rate limits on document creation (10/min) and chat requests (20/min) prevent abuse." },
                { title: "QStash Verification", desc: "Ingestion webhooks from QStash are verified using signing keys to prevent unauthorized job triggers." },
              ].map(({ title, desc }) => (
                <div key={title} className="rounded-2xl border bg-card p-5">
                  <h3 className="mb-2 text-sm font-semibold">{title}</h3>
                  <p className="text-sm text-muted-foreground">{desc}</p>
                </div>
              ))}
            </div>
          </section>

          {/* Scaling & Impact */}
          <section id="scaling-and-impact">
            <h2 className="mb-6 text-2xl font-semibold tracking-tight">Scaling & Impact</h2>
            <div className="space-y-6">
              <div className="rounded-2xl border bg-card p-6">
                <h3 className="mb-4 text-base font-semibold">Current Architecture (Free Tier)</h3>
                <div className="overflow-x-auto">
                  <table className="w-full text-xs">
                    <thead>
                      <tr className="border-b text-left">
                        <th className="pb-2 pr-4 font-semibold">Metric</th>
                        <th className="pb-2 pr-4 font-semibold">Free Tier</th>
                        <th className="pb-2 font-semibold">Limiting Factor</th>
                      </tr>
                    </thead>
                    <tbody className="text-muted-foreground">
                      {[
                        ["Concurrent Users", "Unlimited (serverless)", "Vercel cold starts"],
                        ["Documents / Session", "1 PDF per session", "Design choice (ephemeral)"],
                        ["PDF Size Limit", "25 MB", "Vercel Blob + pypdf"],
                        ["Redis Storage", "256 MB", "Upstash free tier"],
                        ["Milvus Collections", "1 shared collection", "Zilliz free tier (was 5)"],
                        ["Embedding Model", "text-embedding-3-small", "OpenRouter rate limits"],
                        ["Chat Model", "Free models with fallback", "~20 req/min per model"],
                        ["Ingestion Queue", "QStash (500k/mo)", "Free tier limit"],
                      ].map(([metric, value, note]) => (
                        <tr key={metric} className="border-b border-border/50">
                          <td className="py-2.5 pr-4 font-semibold text-foreground">{metric}</td>
                          <td className="py-2.5 pr-4">{value}</td>
                          <td className="py-2.5 italic">{note}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>

              <div className="rounded-2xl border bg-card p-6">
                <h3 className="mb-4 text-base font-semibold">With Paid Options</h3>
                <div className="overflow-x-auto">
                  <table className="w-full text-xs">
                    <thead>
                      <tr className="border-b text-left">
                        <th className="pb-2 pr-4 font-semibold">Upgrade</th>
                        <th className="pb-2 pr-4 font-semibold">What Changes</th>
                        <th className="pb-2 font-semibold">Impact</th>
                      </tr>
                    </thead>
                    <tbody className="text-muted-foreground">
                      {[
                        ["OpenRouter Paid Model", "Switch to Gemini Flash, GPT-4o-mini, Claude Haiku", "~$0.10-0.50/1M tokens, no rate limits"],
                        ["Upstash Redis Pro", "More storage, faster ops, persistence", "Handle millions of concurrent sessions"],
                        ["Zilliz Standard", "More collections, higher throughput", "Sub-10ms vector search at scale"],
                        ["Vercel Pro", "More serverless function invocations", "Higher concurrent request handling"],
                        ["Neon Postgres", "Add persistent document library", "Users can revisit past documents"],
                        ["Custom Domain + CDN", "Branded deployment with edge caching", "Global low-latency access"],
                      ].map(([upgrade, change, impact]) => (
                        <tr key={upgrade} className="border-b border-border/50">
                          <td className="py-2.5 pr-4 font-semibold text-foreground">{upgrade}</td>
                          <td className="py-2.5 pr-4">{change}</td>
                          <td className="py-2.5">{impact}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>

              <div className="rounded-2xl border bg-card p-6">
                <h3 className="mb-4 text-base font-semibold">Estimated Scale with Full Paid Stack</h3>
                <div className="grid gap-3 sm:grid-cols-3">
                  {[
                    { label: "Monthly Active Users", value: "100K+", detail: "With Vercel Pro + Redis Pro" },
                    { label: "Documents Processed", value: "500K+/mo", detail: "With QStash Pro + Blob" },
                    { label: "Monthly Cost", value: "~$50-200", detail: "Depending on LLM usage volume" },
                  ].map(({ label, value, detail }) => (
                    <div key={label} className="rounded-xl bg-muted/50 p-4 text-center">
                      <p className="text-2xl font-bold text-primary">{value}</p>
                      <p className="mt-1 text-sm font-semibold">{label}</p>
                      <p className="text-xs text-muted-foreground">{detail}</p>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </section>

          {/* Use Cases */}
          <section id="use-cases">
            <h2 className="mb-6 text-2xl font-semibold tracking-tight">Use Cases</h2>
            <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
              {[
                { emoji: "Research", title: "Academic Paper Analysis", desc: "Upload research papers and ask about methodology, findings, and conclusions with page-level citations." },
                { emoji: "Legal", title: "Contract Review", desc: "Upload legal documents and quickly find specific clauses, obligations, and terms across lengthy contracts." },
                { emoji: "Finance", title: "Financial Report Q&A", desc: "Chat with earnings reports, annual statements, and market analyses to extract key metrics instantly." },
                { emoji: "Medical", title: "Medical Document Search", desc: "Navigate complex medical literature, clinical trials, and patient reports with grounded answers." },
                { emoji: "Education", title: "Study Material Assistant", desc: "Upload textbooks or lecture notes and generate summaries, explanations, and practice questions." },
                { emoji: "Engineering", title: "Technical Spec Review", desc: "Query architecture docs, RFCs, and technical specifications to find design decisions and trade-offs." },
              ].map(({ emoji, title, desc }) => (
                <div key={title} className="rounded-2xl border bg-card p-5">
                  <div className="mb-2 inline-flex h-8 w-8 items-center justify-center rounded-lg bg-primary/10 text-xs font-bold text-primary">{emoji.slice(0, 2)}</div>
                  <h3 className="mb-1.5 text-sm font-semibold">{title}</h3>
                  <p className="text-xs text-muted-foreground leading-relaxed">{desc}</p>
                </div>
              ))}
            </div>
          </section>

          {/* Deployment */}
          <section id="deployment">
            <h2 className="mb-6 text-2xl font-semibold tracking-tight">Deployment</h2>
            <div className="rounded-2xl border bg-card p-6">
              <p className="mb-4 text-sm text-muted-foreground">The system deploys as two separate Vercel projects from the same monorepo:</p>
              <div className="grid gap-4 sm:grid-cols-2">
                <div className="rounded-xl bg-muted/50 p-4">
                  <p className="mb-2 font-semibold text-sm">Web Project</p>
                  <ul className="space-y-1 text-xs text-muted-foreground list-disc ml-4">
                    <li>Root: apps/web</li>
                    <li>Framework: Next.js 15</li>
                    <li>Env: NEXT_PUBLIC_API_BASE_URL, BLOB_READ_WRITE_TOKEN, QSTASH_TOKEN</li>
                  </ul>
                </div>
                <div className="rounded-xl bg-muted/50 p-4">
                  <p className="mb-2 font-semibold text-sm">API Project</p>
                  <ul className="space-y-1 text-xs text-muted-foreground list-disc ml-4">
                    <li>Root: apps/api</li>
                    <li>Framework: FastAPI + uvicorn</li>
                    <li>Env: OPENROUTER_API_KEY, UPSTASH_REDIS_REST_URL, UPSTASH_REDIS_REST_TOKEN, ZILLIZ_URI, ZILLIZ_TOKEN, BLOB_READ_WRITE_TOKEN, SESSION_SECRET, QSTASH_TOKEN, QSTASH_SIGNING_KEYS</li>
                  </ul>
                </div>
              </div>
              <div className="mt-4 rounded-xl border bg-emerald-500/5 p-4">
                <p className="text-xs text-emerald-600 dark:text-emerald-400">
                  <strong>Local development</strong> works without any external services — Redis falls back to in-memory storage, QStash calls run directly, and embeddings use hash-based vectors.
                </p>
              </div>
            </div>
          </section>
        </div>

        <footer className="mt-20 border-t pt-8 text-center text-xs text-muted-foreground">
          <p>Built with Next.js, FastAPI, Upstash Redis, Zilliz, and OpenRouter</p>
        </footer>
      </div>
    </div>
  );
}
