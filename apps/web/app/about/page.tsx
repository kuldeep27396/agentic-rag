import type { Metadata } from "next";
import ServiceLogo from "@/components/ServiceLogo";

export const metadata: Metadata = {
  title: "Agentic PDF RAG — Design & Architecture",
  description: "High-level design, low-level design, tech stack, and scaling details for the Agentic PDF RAG system.",
};

function SectionBadge({ children, color = "primary" }: { children: React.ReactNode; color?: string }) {
  const colors: Record<string, string> = {
    primary: "bg-blue-500/10 text-blue-600 dark:text-blue-400 border-blue-500/20",
    purple: "bg-purple-500/10 text-purple-600 dark:text-purple-400 border-purple-500/20",
    emerald: "bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border-emerald-500/20",
    amber: "bg-amber-500/10 text-amber-600 dark:text-amber-400 border-amber-500/20",
    rose: "bg-rose-500/10 text-rose-600 dark:text-rose-400 border-rose-500/20",
    cyan: "bg-cyan-500/10 text-cyan-600 dark:text-cyan-400 border-cyan-500/20",
  };
  return (
    <span className={`inline-flex items-center rounded-full border px-3 py-1 text-xs font-semibold ${colors[color] ?? colors.primary}`}>
      {children}
    </span>
  );
}

function FlowStep({ num, title, desc }: { num: number; title: string; desc: string }) {
  return (
    <div className="flex gap-4 items-start">
      <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full bg-gradient-to-br from-blue-500 to-purple-600 text-white text-sm font-bold shadow-sm">
        {num}
      </div>
      <div className="min-w-0 pt-0.5">
        <p className="text-sm font-semibold">{title}</p>
        <p className="text-xs text-muted-foreground mt-0.5">{desc}</p>
      </div>
    </div>
  );
}

export default function AboutPage() {
  return (
    <div className="min-h-dvh bg-background text-foreground">
      <div className="mx-auto max-w-5xl px-6 py-16">
        {/* Hero */}
        <div className="mb-16 text-center">
          <div className="mb-6 inline-flex items-center gap-3 rounded-2xl bg-gradient-to-r from-blue-500/10 via-purple-500/10 to-emerald-500/10 px-6 py-3">
            <span className="text-3xl">📄</span>
            <span className="text-3xl">🤖</span>
            <span className="text-3xl">💬</span>
          </div>
          <h1 className="text-5xl font-bold tracking-tight bg-gradient-to-r from-blue-600 via-purple-600 to-emerald-600 bg-clip-text text-transparent">
            Agentic PDF RAG
          </h1>
          <p className="mt-4 text-lg text-muted-foreground max-w-2xl mx-auto">
            Design & Architecture Documentation — A temporary-session document chat system with grounded citations, powered by RAG.
          </p>
          <div className="mt-6 flex flex-wrap justify-center gap-2">
            {[
              { label: "Serverless", color: "bg-black/5 text-black dark:bg-white/5 dark:text-white" },
              { label: "Session-Scoped", color: "bg-blue-500/10 text-blue-600 dark:text-blue-400" },
              { label: "No Database", color: "bg-purple-500/10 text-purple-600 dark:text-purple-400" },
              { label: "Open Source", color: "bg-emerald-500/10 text-emerald-600 dark:text-emerald-400" },
            ].map((t) => (
              <span key={t.label} className={`rounded-full px-3 py-1 text-xs font-medium ${t.color}`}>{t.label}</span>
            ))}
          </div>
        </div>

        {/* Navigation */}
        <nav className="mb-16 flex flex-wrap justify-center gap-2">
          {["Overview", "HLD", "LLD", "Tech Stack", "Data Model", "Retrieval", "Security", "Scaling", "Use Cases", "Deployment"].map((s) => (
            <a key={s} href={`#${s.toLowerCase().replace(/ /g, "-")}`} className="rounded-full border border-border/50 bg-card px-3.5 py-1.5 text-xs font-medium text-muted-foreground transition-all hover:border-primary/50 hover:text-foreground hover:shadow-sm">
              {s}
            </a>
          ))}
        </nav>

        <div className="space-y-24">
          {/* ===== OVERVIEW ===== */}
          <section id="overview">
            <div className="mb-8">
              <SectionBadge color="primary">Overview</SectionBadge>
              <h2 className="mt-3 text-3xl font-bold tracking-tight">What is this?</h2>
            </div>
            <div className="grid gap-6 md:grid-cols-3">
              <div className="rounded-2xl border bg-gradient-to-br from-blue-500/5 to-blue-500/0 p-6">
                <div className="mb-3 text-2xl">📤</div>
                <h3 className="mb-2 text-sm font-semibold">Upload</h3>
                <p className="text-xs text-muted-foreground leading-relaxed">Drop any PDF — up to 25MB. It gets stored securely and ingested automatically in the background.</p>
              </div>
              <div className="rounded-2xl border bg-gradient-to-br from-purple-500/5 to-purple-500/0 p-6">
                <div className="mb-3 text-2xl">🔍</div>
                <h3 className="mb-2 text-sm font-semibold">Ingest & Chunk</h3>
                <p className="text-xs text-muted-foreground leading-relaxed">Text is extracted, split into parent-child chunks, embedded, and indexed — ready for semantic search.</p>
              </div>
              <div className="rounded-2xl border bg-gradient-to-br from-emerald-500/5 to-emerald-500/0 p-6">
                <div className="mb-3 text-2xl">💬</div>
                <h3 className="mb-2 text-sm font-semibold">Chat with Citations</h3>
                <p className="text-xs text-muted-foreground leading-relaxed">Ask questions and get answers grounded in your document, with page-level citations and follow-up suggestions.</p>
              </div>
            </div>
          </section>

          {/* ===== HLD ===== */}
          <section id="hld">
            <div className="mb-8">
              <SectionBadge color="purple">High-Level Design</SectionBadge>
              <h2 className="mt-3 text-3xl font-bold tracking-tight">System Architecture</h2>
            </div>

            {/* Architecture Diagram */}
            <div className="rounded-2xl border-2 border-blue-500/20 bg-gradient-to-br from-blue-500/5 via-transparent to-purple-500/5 p-6 md:p-8">
              <div className="space-y-6">
                {/* Browser Layer */}
                <div className="rounded-xl bg-gradient-to-r from-sky-500/10 to-sky-500/5 border border-sky-500/20 p-5">
                  <div className="flex items-center gap-3 mb-4">
                    <span className="inline-flex items-center gap-1.5 rounded-lg bg-sky-500/20 px-3 py-1 text-xs font-bold text-sky-600 dark:text-sky-400">
                      🌐 Browser Client
                    </span>
                    <ServiceLogo name="nextjs" />
                    <ServiceLogo name="react" />
                    <ServiceLogo name="TS" />
                    <ServiceLogo name="tailwind" />
                  </div>
                  <div className="grid gap-2 sm:grid-cols-3 text-xs">
                    <div className="rounded-lg bg-white/50 dark:bg-black/20 p-2.5 text-center">
                      <p className="font-semibold">PDF Upload</p>
                      <p className="text-muted-foreground">Direct to Vercel Blob</p>
                    </div>
                    <div className="rounded-lg bg-white/50 dark:bg-black/20 p-2.5 text-center">
                      <p className="font-semibold">Session State</p>
                      <p className="text-muted-foreground">sessionStorage</p>
                    </div>
                    <div className="rounded-lg bg-white/50 dark:bg-black/20 p-2.5 text-center">
                      <p className="font-semibold">SSE Streaming</p>
                      <p className="text-muted-foreground">Real-time chat</p>
                    </div>
                  </div>
                </div>

                {/* Arrow */}
                <div className="flex justify-center text-muted-foreground">
                  <div className="flex items-center gap-2">
                    <span className="text-xs font-mono">REST + SSE</span>
                    <svg width="40" height="24" viewBox="0 0 40 24" fill="none" className="text-blue-500"><path d="M2 12h30M28 4l8 8-8 8" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/></svg>
                  </div>
                </div>

                {/* API Layer */}
                <div className="rounded-xl bg-gradient-to-r from-emerald-500/10 to-emerald-500/5 border border-emerald-500/20 p-5">
                  <div className="flex items-center gap-3 mb-4">
                    <span className="inline-flex items-center gap-1.5 rounded-lg bg-emerald-500/20 px-3 py-1 text-xs font-bold text-emerald-600 dark:text-emerald-400">
                      ⚡ API Server
                    </span>
                    <ServiceLogo name="fastapi" />
                    <ServiceLogo name="python" />
                    <ServiceLogo name="pydantic" />
                    <ServiceLogo name="httpx" />
                  </div>
                  <div className="grid gap-2 sm:grid-cols-4 text-xs">
                    <div className="rounded-lg bg-white/50 dark:bg-black/20 p-2.5 text-center border-l-4 border-emerald-500">
                      <p className="font-semibold">REST Routes</p>
                      <p className="text-muted-foreground">CRUD + Stream</p>
                    </div>
                    <div className="rounded-lg bg-white/50 dark:bg-black/20 p-2.5 text-center border-l-4 border-purple-500">
                      <p className="font-semibold">RAG Agent</p>
                      <p className="text-muted-foreground">Retrieve → Generate</p>
                    </div>
                    <div className="rounded-lg bg-white/50 dark:bg-black/20 p-2.5 text-center border-l-4 border-amber-500">
                      <p className="font-semibold">LLM Client</p>
                      <p className="text-muted-foreground">Fallback chain</p>
                    </div>
                    <div className="rounded-lg bg-white/50 dark:bg-black/20 p-2.5 text-center border-l-4 border-rose-500">
                      <p className="font-semibold">Rate Limiter</p>
                      <p className="text-muted-foreground">IP-based</p>
                    </div>
                  </div>
                </div>

                {/* Arrow */}
                <div className="flex justify-center text-muted-foreground">
                  <div className="flex items-center gap-2">
                    <svg width="24" height="40" viewBox="0 0 24 40" fill="none" className="text-emerald-500"><path d="M12 2v30M4 24l8 8 8-8" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/></svg>
                  </div>
                </div>

                {/* Infrastructure Layer */}
                <div className="grid gap-4 sm:grid-cols-3">
                  <div className="rounded-xl border-2 border-red-500/20 bg-gradient-to-br from-red-500/5 to-red-500/0 p-4 text-center">
                    <div className="mb-2"><ServiceLogo name="upstash" /></div>
                    <p className="text-xs font-semibold">State Store</p>
                    <p className="text-[10px] text-muted-foreground">Documents, sessions, messages, rate limits</p>
                  </div>
                  <div className="rounded-xl border-2 border-blue-500/20 bg-gradient-to-br from-blue-500/5 to-blue-500/0 p-4 text-center">
                    <div className="mb-2"><ServiceLogo name="milvus" /></div>
                    <p className="text-xs font-semibold">Vector Database</p>
                    <p className="text-[10px] text-muted-foreground">Shared collection with document filtering</p>
                  </div>
                  <div className="rounded-xl border-2 border-orange-500/20 bg-gradient-to-br from-orange-500/5 to-orange-500/0 p-4 text-center">
                    <div className="mb-2"><ServiceLogo name="openrouter" /></div>
                    <p className="text-xs font-semibold">LLM Gateway</p>
                    <p className="text-[10px] text-muted-foreground">Multi-model fallback + embeddings</p>
                  </div>
                </div>

                {/* Supporting services */}
                <div className="grid gap-3 sm:grid-cols-4">
                  <div className="rounded-lg border bg-card p-3 text-center">
                    <ServiceLogo name="blob" />
                    <p className="text-[10px] mt-1.5 text-muted-foreground">PDF Storage</p>
                  </div>
                  <div className="rounded-lg border bg-card p-3 text-center">
                    <ServiceLogo name="qstash" />
                    <p className="text-[10px] mt-1.5 text-muted-foreground">Job Queue</p>
                  </div>
                  <div className="rounded-lg border bg-card p-3 text-center">
                    <ServiceLogo name="pypdf" />
                    <p className="text-[10px] mt-1.5 text-muted-foreground">PDF Parser</p>
                  </div>
                  <div className="rounded-lg border bg-card p-3 text-center">
                    <ServiceLogo name="vercel" />
                    <p className="text-[10px] mt-1.5 text-muted-foreground">Deployment</p>
                  </div>
                </div>
              </div>
            </div>

            {/* Request Flow */}
            <div className="mt-8 rounded-2xl border bg-card p-6">
              <h3 className="mb-6 text-base font-semibold">Request Flow</h3>
              <div className="grid gap-5 md:grid-cols-2">
                <div className="space-y-4">
                  <p className="text-xs font-bold uppercase tracking-wider text-blue-500">Upload & Ingest</p>
                  <FlowStep num={1} title="Create Session Token" desc="Browser generates a unique token, stored in sessionStorage" />
                  <FlowStep num={2} title="Upload PDF" desc="File sent directly to Vercel Blob from the browser" />
                  <FlowStep num={3} title="Register Document" desc="POST /v1/documents — FastAPI hashes token, stores metadata in Redis" />
                  <FlowStep num={4} title="Enqueue Ingestion" desc="Next.js publishes to QStash (or calls directly in dev)" />
                  <FlowStep num={5} title="Process PDF" desc="QStash triggers ingestion: extract → chunk → embed → store vectors" />
                </div>
                <div className="space-y-4">
                  <p className="text-xs font-bold uppercase tracking-wider text-emerald-500">Chat & Retrieve</p>
                  <FlowStep num={6} title="Poll Status" desc="Frontend polls GET /v1/documents/{id} until status = ready" />
                  <FlowStep num={7} title="Create Chat Session" desc="POST /v1/chat/sessions — links session to document" />
                  <FlowStep num={8} title="Send Message" desc="POST /v1/chat/sessions/{id}/messages/stream" />
                  <FlowStep num={9} title="Vector Search" desc="Embed query → cosine similarity → retrieve top-k parent chunks" />
                  <FlowStep num={10} title="Generate Answer" desc="Send context to LLM with fallback chain → stream response + suggestions" />
                </div>
              </div>
            </div>
          </section>

          {/* ===== LLD ===== */}
          <section id="lld">
            <div className="mb-8">
              <SectionBadge color="emerald">Low-Level Design</SectionBadge>
              <h2 className="mt-3 text-3xl font-bold tracking-tight">Module Breakdown</h2>
            </div>
            <div className="space-y-6">
              <div className="rounded-2xl border bg-card overflow-hidden">
                <div className="bg-gradient-to-r from-emerald-500/10 to-emerald-500/0 px-6 py-3 border-b">
                  <h3 className="text-sm font-semibold flex items-center gap-2">
                    <span className="h-2 w-2 rounded-full bg-emerald-500" />
                    Backend Modules (Python)
                  </h3>
                </div>
                <div className="p-4">
                  <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
                    {[
                      { name: "Routes", file: "api/routes.py", desc: "REST endpoints for documents, chat, ingestion", color: "border-l-blue-500" },
                      { name: "RAG Agent", file: "agent/graph.py", desc: "Retrieve → build context → call LLM", color: "border-l-purple-500" },
                      { name: "LLM Client", file: "services/llm.py", desc: "OpenRouter + 7 model fallbacks", color: "border-l-amber-500" },
                      { name: "Vector Store", file: "services/vector_store.py", desc: "Milvus shared collection + Redis fallback", color: "border-l-cyan-500" },
                      { name: "Redis State", file: "services/redis_state.py", desc: "Upstash REST with in-memory dev fallback", color: "border-l-red-500" },
                      { name: "Repository", file: "db/repository.py", desc: "Data access: docs, sessions, messages, chunks", color: "border-l-emerald-500" },
                      { name: "PDF Parser", file: "services/pdf.py", desc: "pypdf extraction + parent/child chunking", color: "border-l-orange-500" },
                      { name: "Rate Limiter", file: "services/rate_limit.py", desc: "IP sliding window via Redis INCR", color: "border-l-rose-500" },
                      { name: "Security", file: "core/security.py", desc: "HMAC-SHA256 token hashing + verification", color: "border-l-slate-500" },
                    ].map((m) => (
                      <div key={m.name} className={`rounded-lg border bg-muted/30 p-3 border-l-4 ${m.color}`}>
                        <p className="text-xs font-semibold">{m.name}</p>
                        <p className="text-[10px] text-muted-foreground font-mono mt-0.5">{m.file}</p>
                        <p className="text-[10px] text-muted-foreground mt-1">{m.desc}</p>
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              <div className="rounded-2xl border bg-card overflow-hidden">
                <div className="bg-gradient-to-r from-sky-500/10 to-sky-500/0 px-6 py-3 border-b">
                  <h3 className="text-sm font-semibold flex items-center gap-2">
                    <span className="h-2 w-2 rounded-full bg-sky-500" />
                    Frontend Components (React)
                  </h3>
                </div>
                <div className="p-4">
                  <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
                    {[
                      { name: "App", file: "App.tsx", desc: "Root: upload, chat UI, citations, suggestions" },
                      { name: "Markdown", file: "MarkdownRenderer.tsx", desc: "Rich markdown rendering with GFM" },
                      { name: "API Client", file: "lib/api.ts", desc: "SSE streaming + suggestion parsing" },
                      { name: "Upload Route", file: "api/upload/route.ts", desc: "Vercel Blob handler with PDF validation" },
                      { name: "Ingest Route", file: "api/ingest/route.ts", desc: "QStash enqueue (or direct dev mode)" },
                      { name: "Env Config", file: "lib/env.ts", desc: "API base URL + upload limits" },
                    ].map((m) => (
                      <div key={m.name} className="rounded-lg border-l-4 border-l-sky-500 border bg-muted/30 p-3">
                        <p className="text-xs font-semibold">{m.name}</p>
                        <p className="text-[10px] text-muted-foreground font-mono mt-0.5">{m.file}</p>
                        <p className="text-[10px] text-muted-foreground mt-1">{m.desc}</p>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          </section>

          {/* ===== TECH STACK ===== */}
          <section id="tech-stack">
            <div className="mb-8">
              <SectionBadge color="amber">Tech Stack</SectionBadge>
              <h2 className="mt-3 text-3xl font-bold tracking-tight">Technologies & Services</h2>
            </div>
            <div className="grid gap-4 sm:grid-cols-2">
              {[
                { title: "Frontend", gradient: "from-sky-500/10 to-blue-500/0", border: "border-sky-500/20", badges: [["nextjs"], ["react"], ["TS"], ["tailwind"]] },
                { title: "Backend", gradient: "from-emerald-500/10 to-emerald-500/0", border: "border-emerald-500/20", badges: [["fastapi"], ["python"], ["pydantic"], ["httpx"]] },
                { title: "Infrastructure", gradient: "from-purple-500/10 to-purple-500/0", border: "border-purple-500/20", badges: [["vercel"], ["upstash"], ["qstash"], ["blob"]] },
                { title: "AI & Search", gradient: "from-amber-500/10 to-amber-500/0", border: "border-amber-500/20", badges: [["openrouter"], ["milvus"], ["pypdf"], ["gemini"]] },
              ].map(({ title, gradient, border, badges }) => (
                <div key={title} className={`rounded-2xl border ${border} bg-gradient-to-br ${gradient} p-6`}>
                  <h3 className="mb-4 text-sm font-bold">{title}</h3>
                  <div className="flex flex-wrap gap-2">
                    {badges.map(([name]) => (
                      <ServiceLogo key={name} name={name} />
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </section>

          {/* ===== DATA MODEL ===== */}
          <section id="data-model">
            <div className="mb-8">
              <SectionBadge color="rose">Data Model</SectionBadge>
              <h2 className="mt-3 text-3xl font-bold tracking-tight">Ephemeral State in Redis</h2>
            </div>
            <div className="rounded-2xl border bg-card p-6">
              <div className="flex items-center gap-3 mb-6">
                <ServiceLogo name="upstash" />
                <span className="text-xs text-muted-foreground">All data expires after 30 days. No persistent database needed.</span>
              </div>
              <div className="space-y-2">
                {[
                  { key: "document:{id}", type: "JSON", desc: "Metadata, status, blob URL, token hash, expiry", color: "bg-blue-500" },
                  { key: "chunks:{doc_id}", type: "JSON[]", desc: "Text chunks with page ranges, parent IDs, text hashes", color: "bg-purple-500" },
                  { key: "vectors:{doc_id}", type: "JSON[]", desc: "Embeddings (fallback mode when Milvus unavailable)", color: "bg-cyan-500" },
                  { key: "chat_session:{id}", type: "JSON", desc: "Session record with document reference + token hash", color: "bg-emerald-500" },
                  { key: "messages:{session_id}", type: "JSON[]", desc: "Full chat history with citations", color: "bg-amber-500" },
                  { key: "ingestion_job:{id}", type: "JSON", desc: "Job → document mapping for QStash webhook", color: "bg-orange-500" },
                  { key: "rate:{ip}:{window}", type: "counter", desc: "IP-based anonymous rate limit counters", color: "bg-rose-500" },
                ].map(({ key, type, desc, color }) => (
                  <div key={key} className="flex items-center gap-3 rounded-lg bg-muted/30 px-4 py-2.5">
                    <div className={`h-2 w-2 shrink-0 rounded-full ${color}`} />
                    <code className="text-xs font-semibold text-primary shrink-0 w-48">{key}</code>
                    <span className="text-[10px] font-bold uppercase tracking-wider text-muted-foreground shrink-0 w-14">{type}</span>
                    <span className="text-xs text-muted-foreground">{desc}</span>
                  </div>
                ))}
              </div>
            </div>
          </section>

          {/* ===== RETRIEVAL ===== */}
          <section id="retrieval">
            <div className="mb-8">
              <SectionBadge color="cyan">Retrieval Pipeline</SectionBadge>
              <h2 className="mt-3 text-3xl font-bold tracking-tight">How RAG Works</h2>
            </div>
            <div className="grid gap-4 md:grid-cols-4">
              {[
                { step: "01", title: "Chunk", desc: "Two-tier parent (1500 tok) / child (420 tok, 60 overlap) splitting", color: "from-blue-500 to-blue-600", icon: "✂️" },
                { step: "02", title: "Embed", desc: "text-embedding-3-small (1536d) via OpenRouter or hash-based fallback", color: "from-purple-500 to-purple-600", icon: "🔢" },
                { step: "03", title: "Search", desc: "Cosine similarity on child chunks → collapse to top-k parent contexts", color: "from-emerald-500 to-emerald-600", icon: "🔎" },
                { step: "04", title: "Generate", desc: "LLM answers with PDF context + optional web search via Firecrawl", color: "from-amber-500 to-amber-600", icon: "✨" },
              ].map(({ step, title, desc, color, icon }) => (
                <div key={step} className="relative rounded-2xl border bg-card p-5 overflow-hidden">
                  <div className={`absolute top-0 left-0 right-0 h-1 bg-gradient-to-r ${color}`} />
                  <span className="text-2xl">{icon}</span>
                  <div className="mt-3 flex items-center gap-2">
                    <span className={`rounded-md bg-gradient-to-r ${color} px-2 py-0.5 text-[10px] font-bold text-white`}>STEP {step}</span>
                  </div>
                  <h3 className="mt-2 text-sm font-semibold">{title}</h3>
                  <p className="mt-1 text-xs text-muted-foreground leading-relaxed">{desc}</p>
                </div>
              ))}
            </div>
          </section>

          {/* ===== SECURITY ===== */}
          <section id="security">
            <div className="mb-8">
              <SectionBadge color="rose">Security & Privacy</SectionBadge>
              <h2 className="mt-3 text-3xl font-bold tracking-tight">Built for Privacy</h2>
            </div>
            <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
              {[
                { icon: "🔐", title: "HMAC Token Security", desc: "Session tokens are hashed with HMAC-SHA256. Never stored in plaintext." },
                { icon: "🧹", title: "Auto-Expiry", desc: "All data auto-purges after 30 days via Redis TTL. No cleanup jobs needed." },
                { icon: "👤", title: "No Accounts", desc: "Zero user data collected. No cookies. No tracking. Session-scoped only." },
                { icon: "🚫", title: "No Shared Library", desc: "Documents are private to your browser session. No cross-session access." },
                { icon: "⏱️", title: "Rate Limiting", desc: "IP-based limits: 10 doc creates/min, 20 chat requests/min." },
                { icon: "✅", title: "QStash Verification", desc: "Webhook signatures verified with signing keys to prevent spoofing." },
              ].map(({ icon, title, desc }) => (
                <div key={title} className="rounded-2xl border bg-card p-5">
                  <span className="text-xl">{icon}</span>
                  <h3 className="mt-2 text-sm font-semibold">{title}</h3>
                  <p className="mt-1 text-xs text-muted-foreground leading-relaxed">{desc}</p>
                </div>
              ))}
            </div>
          </section>

          {/* ===== SCALING ===== */}
          <section id="scaling">
            <div className="mb-8">
              <SectionBadge color="amber">Scaling & Impact</SectionBadge>
              <h2 className="mt-3 text-3xl font-bold tracking-tight">Free Today, Scales Tomorrow</h2>
            </div>
            <div className="space-y-6">
              <div className="grid gap-4 sm:grid-cols-3">
                {[
                  { label: "MAU", value: "100K+", sub: "Vercel Pro + Redis Pro", gradient: "from-blue-500 to-purple-600" },
                  { label: "Docs / Month", value: "500K+", sub: "QStash Pro + Blob", gradient: "from-purple-500 to-emerald-600" },
                  { label: "Monthly Cost", value: "~$50-200", sub: "Depending on LLM volume", gradient: "from-emerald-500 to-amber-600" },
                ].map(({ label, value, sub, gradient }) => (
                  <div key={label} className="rounded-2xl border bg-card p-6 text-center">
                    <div className={`mx-auto mb-3 inline-block rounded-lg bg-gradient-to-r ${gradient} px-3 py-1`}>
                      <span className="text-2xl font-black text-white">{value}</span>
                    </div>
                    <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wider">{label}</p>
                    <p className="text-[10px] text-muted-foreground mt-0.5">{sub}</p>
                  </div>
                ))}
              </div>

              <div className="rounded-2xl border bg-card overflow-hidden">
                <div className="grid md:grid-cols-2">
                  <div className="p-6 border-b md:border-b-0 md:border-r">
                    <div className="flex items-center gap-2 mb-4">
                      <span className="rounded-full bg-emerald-500/10 px-2.5 py-0.5 text-[10px] font-bold text-emerald-600">FREE TIER</span>
                      <span className="text-xs text-muted-foreground">Currently running on</span>
                    </div>
                    <ul className="space-y-2 text-xs text-muted-foreground">
                      {[
                        "Unlimited concurrent users (serverless)",
                        "25 MB PDF size limit",
                        "256 MB Redis storage",
                        "1 shared Milvus collection",
                        "7 free LLM models with fallback",
                        "500K QStash messages/month",
                      ].map((item) => (
                        <li key={item} className="flex items-start gap-2">
                          <span className="text-emerald-500 mt-0.5">✓</span> {item}
                        </li>
                      ))}
                    </ul>
                  </div>
                  <div className="p-6">
                    <div className="flex items-center gap-2 mb-4">
                      <span className="rounded-full bg-purple-500/10 px-2.5 py-0.5 text-[10px] font-bold text-purple-600">PAID UPGRADE</span>
                      <span className="text-xs text-muted-foreground">Path to production</span>
                    </div>
                    <ul className="space-y-2 text-xs text-muted-foreground">
                      {[
                        "Gemini Flash / Claude Haiku (no rate limits)",
                        "Redis Pro: millions of concurrent sessions",
                        "Zilliz Standard: sub-10ms vector search",
                        "Vercel Pro: higher function invocations",
                        "Neon Postgres: persistent document library",
                        "Custom domain + CDN: global low-latency",
                      ].map((item) => (
                        <li key={item} className="flex items-start gap-2">
                          <span className="text-purple-500 mt-0.5">→</span> {item}
                        </li>
                      ))}
                    </ul>
                  </div>
                </div>
              </div>
            </div>
          </section>

          {/* ===== USE CASES ===== */}
          <section id="use-cases">
            <div className="mb-8">
              <SectionBadge color="primary">Use Cases</SectionBadge>
              <h2 className="mt-3 text-3xl font-bold tracking-tight">Where This Shines</h2>
            </div>
            <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
              {[
                { icon: "🎓", title: "Academic Papers", desc: "Ask about methodology, findings, and conclusions with page citations.", color: "from-blue-500/10 to-blue-500/0 border-blue-500/20" },
                { icon: "⚖️", title: "Legal Contracts", desc: "Find specific clauses, obligations, and terms across lengthy documents.", color: "from-purple-500/10 to-purple-500/0 border-purple-500/20" },
                { icon: "📊", title: "Financial Reports", desc: "Extract key metrics from earnings reports and annual statements.", color: "from-emerald-500/10 to-emerald-500/0 border-emerald-500/20" },
                { icon: "🏥", title: "Medical Literature", desc: "Navigate clinical trials, research papers, and patient reports.", color: "from-rose-500/10 to-rose-500/0 border-rose-500/20" },
                { icon: "📚", title: "Study Materials", desc: "Generate summaries, explanations, and practice questions from textbooks.", color: "from-amber-500/10 to-amber-500/0 border-amber-500/20" },
                { icon: "🏗️", title: "Technical Specs", desc: "Query architecture docs, RFCs, and specs for design decisions.", color: "from-cyan-500/10 to-cyan-500/0 border-cyan-500/20" },
              ].map(({ icon, title, desc, color }) => (
                <div key={title} className={`rounded-2xl border bg-gradient-to-br ${color} p-5`}>
                  <span className="text-2xl">{icon}</span>
                  <h3 className="mt-2 text-sm font-semibold">{title}</h3>
                  <p className="mt-1 text-xs text-muted-foreground leading-relaxed">{desc}</p>
                </div>
              ))}
            </div>
          </section>

          {/* ===== DEPLOYMENT ===== */}
          <section id="deployment">
            <div className="mb-8">
              <SectionBadge color="cyan">Deployment</SectionBadge>
              <h2 className="mt-3 text-3xl font-bold tracking-tight">Two Vercel Projects</h2>
            </div>
            <div className="grid gap-4 md:grid-cols-2">
              <div className="rounded-2xl border border-sky-500/20 bg-gradient-to-br from-sky-500/5 to-transparent p-6">
                <div className="flex items-center gap-3 mb-4">
                  <ServiceLogo name="nextjs" />
                  <h3 className="text-sm font-semibold">Web Project</h3>
                </div>
                <ul className="space-y-1.5 text-xs text-muted-foreground">
                  <li className="flex gap-2"><span className="text-sky-500 font-bold">→</span> Root: <code>apps/web</code></li>
                  <li className="flex gap-2"><span className="text-sky-500 font-bold">→</span> Framework: Next.js 15 App Router</li>
                  <li className="flex gap-2"><span className="text-sky-500 font-bold">→</span> NEXT_PUBLIC_API_BASE_URL</li>
                  <li className="flex gap-2"><span className="text-sky-500 font-bold">→</span> BLOB_READ_WRITE_TOKEN</li>
                  <li className="flex gap-2"><span className="text-sky-500 font-bold">→</span> QSTASH_TOKEN</li>
                </ul>
              </div>
              <div className="rounded-2xl border border-emerald-500/20 bg-gradient-to-br from-emerald-500/5 to-transparent p-6">
                <div className="flex items-center gap-3 mb-4">
                  <ServiceLogo name="fastapi" />
                  <h3 className="text-sm font-semibold">API Project</h3>
                </div>
                <ul className="space-y-1.5 text-xs text-muted-foreground">
                  <li className="flex gap-2"><span className="text-emerald-500 font-bold">→</span> Root: <code>apps/api</code></li>
                  <li className="flex gap-2"><span className="text-emerald-500 font-bold">→</span> Framework: FastAPI + uvicorn</li>
                  <li className="flex gap-2"><span className="text-emerald-500 font-bold">→</span> OPENROUTER_API_KEY</li>
                  <li className="flex gap-2"><span className="text-emerald-500 font-bold">→</span> UPSTASH_REDIS_REST_URL + TOKEN</li>
                  <li className="flex gap-2"><span className="text-emerald-500 font-bold">→</span> ZILLIZ_URI + ZILLIZ_TOKEN</li>
                  <li className="flex gap-2"><span className="text-emerald-500 font-bold">→</span> SESSION_SECRET, QSTASH signing keys</li>
                </ul>
              </div>
            </div>
            <div className="mt-4 rounded-xl bg-emerald-500/5 border border-emerald-500/20 p-4 flex items-start gap-3">
              <span className="text-lg">💻</span>
              <p className="text-xs text-emerald-600 dark:text-emerald-400 leading-relaxed">
                <strong>Local development</strong> works with zero external services — Redis falls back to in-memory storage, QStash runs directly, and embeddings use hash-based vectors. Just <code className="rounded bg-emerald-500/10 px-1">pip install -r requirements.txt</code> and run.
              </p>
            </div>
          </section>
        </div>

        {/* Footer */}
        <footer className="mt-24 flex flex-wrap items-center justify-center gap-3 border-t pt-8">
          <ServiceLogo name="nextjs" />
          <ServiceLogo name="fastapi" />
          <ServiceLogo name="upstash" />
          <ServiceLogo name="milvus" />
          <ServiceLogo name="openrouter" />
          <span className="text-xs text-muted-foreground">Built with ❤️</span>
        </footer>
      </div>
    </div>
  );
}
