"use client";

import { upload } from "@vercel/blob/client";
import {
  Bot,
  CheckCircle2,
  ChevronRight,
  FileText,
  FileUp,
  Loader2,
  Moon,
  Send,
  Shield,
  Sparkles,
  Sun,
  Upload,
  X,
} from "lucide-react";
import { FormEvent, useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Citation,
  ChatMessage,
  DocumentRecord,
  createChatSession,
  createDocument,
  enqueueIngestion,
  getDocument,
  streamChat,
} from "@/lib/api";
import { MAX_UPLOAD_BYTES } from "@/lib/env";
import { cn } from "@/lib/utils";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import { Skeleton } from "@/components/ui/skeleton";

type SessionState = {
  token: string;
  documentId?: string;
  chatSessionId?: string;
};

function sessionToken() {
  const key = "agentic-rag-session";
  const existing = sessionStorage.getItem(key);
  if (existing) return existing;
  const token = crypto.randomUUID() + crypto.randomUUID();
  sessionStorage.setItem(key, token);
  return token;
}

function formatBytes(bytes: number) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

const SUGGESTIONS = [
  "Summarize the key findings of this document",
  "What are the main arguments presented?",
  "Explain the methodology used",
  "What conclusions does the author reach?",
];

function TypingIndicator() {
  return (
    <div className="flex items-end gap-3 animate-fade-in">
      <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-primary/10">
        <Bot className="h-4 w-4 text-primary" />
      </div>
      <div className="rounded-2xl rounded-bl-md border bg-card px-4 py-3">
        <div className="flex gap-1.5">
          <span className="h-2 w-2 rounded-full bg-muted-foreground/40 animate-pulse-gentle" style={{ animationDelay: "0ms" }} />
          <span className="h-2 w-2 rounded-full bg-muted-foreground/40 animate-pulse-gentle" style={{ animationDelay: "300ms" }} />
          <span className="h-2 w-2 rounded-full bg-muted-foreground/40 animate-pulse-gentle" style={{ animationDelay: "600ms" }} />
        </div>
      </div>
    </div>
  );
}

function StatusIcon({ status }: { status: string }) {
  switch (status) {
    case "ready":
      return <CheckCircle2 className="h-4 w-4 text-emerald-500" />;
    case "ingesting":
      return <Loader2 className="h-4 w-4 text-primary animate-spin" />;
    case "failed":
      return <X className="h-4 w-4 text-destructive" />;
    default:
      return <FileText className="h-4 w-4 text-muted-foreground" />;
  }
}

function StatusBadge({ status }: { status: string }) {
  const config: Record<string, { label: string; variant: "default" | "secondary" | "destructive" | "outline" }> = {
    uploaded: { label: "Uploaded", variant: "secondary" },
    ingesting: { label: "Processing", variant: "default" },
    ready: { label: "Ready", variant: "default" },
    failed: { label: "Failed", variant: "destructive" },
    deleted: { label: "Deleted", variant: "outline" },
  };
  const { label, variant } = config[status] ?? { label: status, variant: "outline" as const };
  return <Badge variant={variant} className={cn(status === "ready" && "bg-emerald-500/15 text-emerald-600 dark:text-emerald-400 border-emerald-500/20")}>{label}</Badge>;
}

export function App() {
  const [session, setSession] = useState<SessionState | null>(null);
  const [documentRecord, setDocumentRecord] = useState<DocumentRecord | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [draft, setDraft] = useState("");
  const [answerDraft, setAnswerDraft] = useState("");
  const [sending, setSending] = useState(false);
  const [isDark, setIsDark] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const [showCitations, setShowCitations] = useState(false);
  const [mobileSidebar, setMobileSidebar] = useState(true);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    if (window.matchMedia("(prefers-color-scheme: dark)").matches) {
      setIsDark(true);
      document.documentElement.classList.add("dark");
    }
  }, []);

  useEffect(() => {
    const token = sessionToken();
    const savedDocumentId = sessionStorage.getItem("agentic-rag-document-id") ?? undefined;
    const savedChatSessionId = sessionStorage.getItem("agentic-rag-chat-session-id") ?? undefined;
    setSession({ token, documentId: savedDocumentId, chatSessionId: savedChatSessionId });
  }, []);

  useEffect(() => {
    if (!session?.documentId) return;
    let active = true;
    const load = async () => {
      try {
        const doc = await getDocument(session.documentId!);
        if (active) setDocumentRecord(doc);
      } catch (err) {
        if (active) setError(err instanceof Error ? err.message : "Could not load document");
      }
    };
    load();
    const interval = setInterval(load, 2500);
    return () => {
      active = false;
      clearInterval(interval);
    };
  }, [session?.documentId]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, answerDraft]);

  const citations = useMemo<Citation[]>(() => messages.flatMap((m) => m.citations ?? []), [messages]);

  useEffect(() => {
    if (citations.length > 0) setShowCitations(true);
  }, [citations.length]);

  const toggleDark = useCallback(() => {
    setIsDark((prev) => {
      const next = !prev;
      document.documentElement.classList.toggle("dark", next);
      return next;
    });
  }, []);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const dropped = e.dataTransfer.files[0];
    if (dropped) setFile(dropped);
  }, []);

  async function handleUpload() {
    if (!file || !session) return;
    setError(null);
    if (file.size > MAX_UPLOAD_BYTES) {
      setError("PDF exceeds the 25MB upload limit.");
      return;
    }
    if (file.type !== "application/pdf" && !file.name.toLowerCase().endsWith(".pdf")) {
      setError("Only PDF files are supported.");
      return;
    }
    setUploading(true);
    try {
      const blob = await upload(file.name, file, { access: "public", handleUploadUrl: "/api/upload" });
      const doc = await createDocument({
        filename: file.name,
        blobUrl: blob.url,
        sizeBytes: file.size,
        sessionToken: session.token,
      });
      sessionStorage.setItem("agentic-rag-document-id", doc.id);
      setSession({ ...session, documentId: doc.id });
      setDocumentRecord(doc);
      if (doc.ingestion_job_id) await enqueueIngestion(doc.ingestion_job_id);
      setFile(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Upload failed");
    } finally {
      setUploading(false);
    }
  }

  async function ensureChatSession() {
    if (!session || !documentRecord) throw new Error("Upload a PDF first.");
    if (session.chatSessionId) return session.chatSessionId;
    const chat = await createChatSession(documentRecord.id, session.token);
    sessionStorage.setItem("agentic-rag-chat-session-id", chat.id);
    setSession({ ...session, chatSessionId: chat.id });
    return chat.id;
  }

  async function sendMessage(event: FormEvent) {
    event.preventDefault();
    if (!draft.trim() || !session) return;
    const content = draft.trim();
    setDraft("");
    setError(null);
    setSending(true);
    setAnswerDraft("");
    setMobileSidebar(false);
    const optimisticUser: ChatMessage = {
      id: crypto.randomUUID(),
      session_id: session.chatSessionId ?? "pending",
      role: "user",
      content,
      citations: [],
      created_at: new Date().toISOString(),
    };
    setMessages((current) => [...current, optimisticUser]);
    try {
      const chatSessionId = await ensureChatSession();
      const finalMessage = await streamChat({
        sessionId: chatSessionId,
        sessionToken: session.token,
        content,
        onDelta: (value) => setAnswerDraft(value),
      });
      if (finalMessage) setMessages((current) => [...current, finalMessage]);
      setAnswerDraft("");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Chat request failed");
    } finally {
      setSending(false);
    }
  }

  function handleSuggestion(suggestion: string) {
    setDraft(suggestion);
    textareaRef.current?.focus();
  }

  const isReady = documentRecord?.status === "ready";

  return (
    <div className="flex h-dvh overflow-hidden bg-background">
      {/* Sidebar */}
      <aside
        className={cn(
          "flex w-full flex-col border-r bg-card transition-transform duration-300 lg:w-[380px] lg:translate-x-0 lg:flex-shrink-0",
          mobileSidebar ? "translate-x-0" : "-translate-x-full lg:translate-x-0"
        )}
      >
        {/* Sidebar header */}
        <div className="flex items-center justify-between border-b px-5 py-4">
          <div className="flex items-center gap-3">
            <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-gradient-to-br from-primary to-primary/60 shadow-sm">
              <Sparkles className="h-5 w-5 text-primary-foreground" />
            </div>
            <div>
              <h1 className="text-base font-semibold tracking-tight">Agentic PDF RAG</h1>
              <p className="text-xs text-muted-foreground">Chat with your documents</p>
            </div>
          </div>
          <Button variant="ghost" size="icon" onClick={toggleDark} className="h-8 w-8">
            {isDark ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
          </Button>
        </div>

        <ScrollArea className="flex-1">
          <div className="space-y-5 p-5">
            {/* Upload area */}
            <Card className="overflow-hidden">
              <CardHeader className="pb-3">
                <CardTitle className="flex items-center gap-2 text-sm">
                  <Upload className="h-4 w-4" />
                  Upload PDF
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <div
                  onDragOver={handleDragOver}
                  onDragLeave={handleDragLeave}
                  onDrop={handleDrop}
                  onClick={() => fileInputRef.current?.click()}
                  className={cn(
                    "group relative flex cursor-pointer flex-col items-center justify-center gap-2 rounded-xl border-2 border-dashed px-4 py-8 transition-all duration-200",
                    isDragging
                      ? "border-primary bg-primary/5 scale-[1.02]"
                      : "border-muted-foreground/20 hover:border-primary/50 hover:bg-primary/5",
                    file && "border-primary/30 bg-primary/5"
                  )}
                >
                  <div className={cn(
                    "flex h-12 w-12 items-center justify-center rounded-full transition-colors",
                    isDragging ? "bg-primary/15" : "bg-muted",
                    file && "bg-primary/15"
                  )}>
                    {file ? (
                      <FileText className="h-6 w-6 text-primary" />
                    ) : (
                      <FileUp className="h-6 w-6 text-muted-foreground transition-colors group-hover:text-primary" />
                    )}
                  </div>
                  {file ? (
                    <div className="text-center">
                      <p className="text-sm font-medium truncate max-w-[200px]">{file.name}</p>
                      <p className="text-xs text-muted-foreground">{formatBytes(file.size)}</p>
                    </div>
                  ) : (
                    <div className="text-center">
                      <p className="text-sm font-medium">Drop a PDF here</p>
                      <p className="text-xs text-muted-foreground">or click to browse (max 25MB)</p>
                    </div>
                  )}
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept="application/pdf"
                    onChange={(e) => setFile(e.target.files?.[0] ?? null)}
                    className="hidden"
                  />
                </div>

                <Button
                  className="w-full"
                  disabled={!file || uploading}
                  onClick={handleUpload}
                  size="sm"
                >
                  {uploading ? (
                    <>
                      <Loader2 className="h-4 w-4 animate-spin" />
                      Uploading...
                    </>
                  ) : (
                    <>
                      <Upload className="h-4 w-4" />
                      Upload & Process
                    </>
                  )}
                </Button>
              </CardContent>
            </Card>

            {/* Document status */}
            {documentRecord && (
              <Card className="animate-slide-up overflow-hidden">
                <CardHeader className="pb-3">
                  <CardTitle className="flex items-center justify-between text-sm">
                    <span className="flex items-center gap-2">
                      <StatusIcon status={documentRecord.status} />
                      Document
                    </span>
                    <StatusBadge status={documentRecord.status} />
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-3">
                  <div className="flex items-center gap-2 rounded-lg bg-muted/50 p-2.5">
                    <FileText className="h-4 w-4 shrink-0 text-muted-foreground" />
                    <span className="truncate text-sm">{documentRecord.filename}</span>
                  </div>

                  {documentRecord.status === "ingesting" && (
                    <div className="space-y-2">
                      <Progress value={undefined} className="h-1.5 [&>div]:animate-pulse" />
                      <p className="text-xs text-center text-muted-foreground">Processing document...</p>
                    </div>
                  )}

                  <div className="grid grid-cols-3 gap-2">
                    <div className="rounded-lg bg-muted/50 p-2 text-center">
                      <p className="text-xs text-muted-foreground">Pages</p>
                      <p className="text-sm font-semibold">{documentRecord.page_count ?? <Skeleton className="mx-auto mt-1 h-4 w-6" />}</p>
                    </div>
                    <div className="rounded-lg bg-muted/50 p-2 text-center">
                      <p className="text-xs text-muted-foreground">Chunks</p>
                      <p className="text-sm font-semibold">{documentRecord.chunk_count ?? <Skeleton className="mx-auto mt-1 h-4 w-6" />}</p>
                    </div>
                    <div className="rounded-lg bg-muted/50 p-2 text-center">
                      <p className="text-xs text-muted-foreground">Size</p>
                      <p className="text-sm font-semibold">{formatBytes(documentRecord.size_bytes)}</p>
                    </div>
                  </div>

                  {documentRecord.status === "ready" && (
                    <div className="flex items-center gap-2 rounded-lg bg-emerald-500/10 px-3 py-2 text-emerald-600 dark:text-emerald-400 animate-fade-in">
                      <CheckCircle2 className="h-4 w-4" />
                      <span className="text-xs font-medium">Ready for questions</span>
                    </div>
                  )}

                  {documentRecord.error && (
                    <p className="text-xs text-destructive">{documentRecord.error}</p>
                  )}
                </CardContent>
              </Card>
            )}

            {/* Error */}
            {error && (
              <div className="animate-slide-up rounded-lg border border-destructive/30 bg-destructive/5 px-3 py-2">
                <p className="text-sm text-destructive">{error}</p>
              </div>
            )}

            {/* Privacy notice */}
            <div className="flex items-start gap-2 rounded-lg bg-muted/30 px-3 py-2.5">
              <Shield className="mt-0.5 h-3.5 w-3.5 shrink-0 text-muted-foreground" />
              <p className="text-xs leading-relaxed text-muted-foreground">
                Files are retained for 30 days and scoped to this browser session only. No accounts or sharing.
              </p>
            </div>
          </div>
        </ScrollArea>
      </aside>

      {/* Main chat area */}
      <main className="flex min-w-0 flex-1 flex-col">
        {/* Chat header */}
        <div className="flex items-center justify-between border-b px-4 py-3">
          <div className="flex items-center gap-3">
            <Button
              variant="ghost"
              size="icon"
              className="h-8 w-8 lg:hidden"
              onClick={() => setMobileSidebar(!mobileSidebar)}
            >
              <ChevronRight className={cn("h-4 w-4 transition-transform", mobileSidebar && "rotate-180")} />
            </Button>
            <div className="flex items-center gap-2">
              {documentRecord ? (
                <>
                  <FileText className="h-4 w-4 text-muted-foreground" />
                  <span className="truncate text-sm font-medium">{documentRecord.filename}</span>
                  <StatusBadge status={documentRecord.status} />
                </>
              ) : (
                <span className="text-sm text-muted-foreground">No document loaded</span>
              )}
            </div>
          </div>
          {citations.length > 0 && (
            <Button
              variant="outline"
              size="sm"
              onClick={() => setShowCitations(!showCitations)}
              className="hidden gap-1.5 md:flex"
            >
              <FileText className="h-3.5 w-3.5" />
              Citations
              <Badge variant="secondary" className="ml-1 h-5 px-1.5 text-xs">
                {citations.length}
              </Badge>
            </Button>
          )}
        </div>

        {/* Messages area */}
        <ScrollArea className="flex-1">
          <div className="mx-auto max-w-3xl px-4 py-6">
            {messages.length === 0 && !sending ? (
              <div className="flex flex-col items-center justify-center py-16 animate-fade-in">
                <div className="mb-6 flex h-16 w-16 items-center justify-center rounded-2xl bg-gradient-to-br from-primary/20 to-primary/5">
                  <Bot className="h-8 w-8 text-primary" />
                </div>
                <h2 className="mb-2 text-xl font-semibold tracking-tight">
                  {isReady ? "What would you like to know?" : "Upload a PDF to get started"}
                </h2>
                <p className="mb-8 max-w-md text-center text-sm text-muted-foreground">
                  {isReady
                    ? "Ask questions about your document. Answers are grounded with page-level citations."
                    : "Drop a PDF in the sidebar to start chatting with your document."}
                </p>
                {isReady && (
                  <div className="grid gap-2 sm:grid-cols-2">
                    {SUGGESTIONS.map((suggestion) => (
                      <button
                        key={suggestion}
                        onClick={() => handleSuggestion(suggestion)}
                        className="rounded-xl border bg-card px-4 py-3 text-left text-sm text-muted-foreground transition-colors hover:bg-accent hover:text-accent-foreground"
                      >
                        <ChevronRight className="mr-2 inline h-3.5 w-3.5 text-primary" />
                        {suggestion}
                      </button>
                    ))}
                  </div>
                )}
              </div>
            ) : (
              <div className="space-y-4">
                {messages.map((message) => (
                  <div
                    key={message.id}
                    className={cn(
                      "flex items-end gap-3 animate-slide-up",
                      message.role === "user" ? "flex-row-reverse" : ""
                    )}
                  >
                    <div
                      className={cn(
                        "flex h-7 w-7 shrink-0 items-center justify-center rounded-full",
                        message.role === "user"
                          ? "bg-primary text-primary-foreground"
                          : "bg-primary/10 text-primary"
                      )}
                    >
                      {message.role === "user" ? (
                        <span className="text-xs font-semibold">U</span>
                      ) : (
                        <Bot className="h-3.5 w-3.5" />
                      )}
                    </div>
                    <div
                      className={cn(
                        "max-w-[80%] rounded-2xl border px-4 py-2.5 text-sm leading-relaxed",
                        message.role === "user"
                          ? "rounded-br-md bg-primary text-primary-foreground border-primary"
                          : "rounded-bl-md bg-card"
                      )}
                    >
                      <div className="whitespace-pre-wrap">{message.content}</div>
                      {message.citations.length > 0 && (
                        <div className="mt-2 flex flex-wrap gap-1.5 border-t border-border/50 pt-2">
                          {message.citations.map((c, i) => (
                            <Badge key={`${c.chunk_id}-${i}`} variant="outline" className="text-xs font-normal">
                              p.{c.page_start ?? "?"}–{c.page_end ?? "?"}
                            </Badge>
                          ))}
                        </div>
                      )}
                    </div>
                  </div>
                ))}
                {answerDraft && (
                  <div className="flex items-end gap-3 animate-slide-up">
                    <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-primary/10 text-primary">
                      <Bot className="h-3.5 w-3.5" />
                    </div>
                    <div className="max-w-[80%] rounded-2xl rounded-bl-md border bg-card px-4 py-2.5 text-sm leading-relaxed">
                      <div className="whitespace-pre-wrap">{answerDraft}</div>
                    </div>
                  </div>
                )}
                {sending && !answerDraft && <TypingIndicator />}
                <div ref={messagesEndRef} />
              </div>
            )}
          </div>
        </ScrollArea>

        {/* Composer */}
        <div className="border-t bg-card">
          <form onSubmit={sendMessage} className="mx-auto flex max-w-3xl items-end gap-3 px-4 py-3">
            <div className="relative flex-1">
              <textarea
                ref={textareaRef}
                value={draft}
                onChange={(e) => {
                  setDraft(e.target.value);
                  e.target.style.height = "auto";
                  e.target.style.height = Math.min(e.target.scrollHeight, 120) + "px";
                }}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    if (draft.trim() && isReady && !sending) {
                      sendMessage(e as unknown as FormEvent);
                    }
                  }
                }}
                placeholder={isReady ? "Ask about your document... (Enter to send)" : "Upload a PDF first..."}
                disabled={!isReady || sending}
                rows={1}
                className="min-h-[40px] max-h-[120px] w-full resize-none rounded-xl border bg-background px-4 py-2.5 pr-12 text-sm outline-none transition-colors placeholder:text-muted-foreground focus-visible:ring-1 focus-visible:ring-ring disabled:opacity-50"
              />
            </div>
            <Button
              type="submit"
              size="icon"
              disabled={!draft.trim() || !isReady || sending}
              className="h-10 w-10 shrink-0 rounded-xl"
            >
              {sending ? <Loader2 className="h-4 w-4 animate-spin" /> : <Send className="h-4 w-4" />}
            </Button>
          </form>
        </div>
      </main>

      {/* Citations panel */}
      {showCitations && citations.length > 0 && (
        <aside className="hidden w-[320px] shrink-0 flex-col border-l bg-card md:flex animate-slide-in-right">
          <div className="flex items-center justify-between border-b px-4 py-3">
            <h2 className="text-sm font-semibold">Citations</h2>
            <Button variant="ghost" size="icon" className="h-7 w-7" onClick={() => setShowCitations(false)}>
              <X className="h-3.5 w-3.5" />
            </Button>
          </div>
          <ScrollArea className="flex-1">
            <div className="space-y-1 p-3">
              {citations.map((citation, index) => (
                <div key={`${citation.chunk_id}-${index}`} className="animate-fade-in">
                  <Card className="overflow-hidden">
                    <CardContent className="p-3">
                      <div className="flex items-start justify-between gap-2">
                        <div className="min-w-0">
                          <p className="truncate text-xs font-medium">{citation.filename}</p>
                          <p className="text-xs text-muted-foreground">
                            Pages {citation.page_start ?? "?"}–{citation.page_end ?? "?"}
                          </p>
                        </div>
                        <Badge variant="outline" className="shrink-0 text-xs">
                          {citation.source_type === "web" ? "Web" : "PDF"}
                        </Badge>
                      </div>
                      {citation.score != null && (
                        <div className="mt-2 flex items-center gap-2">
                          <div className="h-1.5 flex-1 rounded-full bg-muted overflow-hidden">
                            <div
                              className="h-full rounded-full bg-primary transition-all"
                              style={{ width: `${Math.round(citation.score * 100)}%` }}
                            />
                          </div>
                          <span className="text-xs text-muted-foreground">{Math.round(citation.score * 100)}%</span>
                        </div>
                      )}
                    </CardContent>
                  </Card>
                </div>
              ))}
            </div>
          </ScrollArea>
        </aside>
      )}
    </div>
  );
}
