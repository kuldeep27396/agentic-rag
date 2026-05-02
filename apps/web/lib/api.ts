import { apiBaseUrl } from "./env";

export type DocumentStatus = "uploaded" | "ingesting" | "ready" | "failed" | "deleted";

export type DocumentRecord = {
  id: string;
  filename: string;
  size_bytes: number;
  status: DocumentStatus;
  page_count?: number;
  chunk_count?: number;
  error?: string;
  expires_at: string;
  ingestion_job_id?: string;
};

export type Citation = {
  chunk_id: string;
  filename: string;
  page_start?: number;
  page_end?: number;
  source_type: "pdf" | "web";
  score?: number;
  url?: string;
};

export type ChatMessage = {
  id: string;
  session_id: string;
  role: "user" | "assistant";
  content: string;
  citations: Citation[];
  created_at: string;
};

export async function createDocument(input: {
  filename: string;
  blobUrl: string;
  sizeBytes: number;
  sessionToken: string;
}) {
  const response = await fetch(`${apiBaseUrl()}/v1/documents`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      filename: input.filename,
      blob_url: input.blobUrl,
      size_bytes: input.sizeBytes,
      content_type: "application/pdf",
      session_token: input.sessionToken
    })
  });
  if (!response.ok) throw new Error(await response.text());
  return (await response.json()) as DocumentRecord;
}

export async function getDocument(documentId: string) {
  const response = await fetch(`${apiBaseUrl()}/v1/documents/${documentId}`, { cache: "no-store" });
  if (!response.ok) throw new Error(await response.text());
  return (await response.json()) as DocumentRecord;
}

export async function createChatSession(documentId: string, sessionToken: string) {
  const response = await fetch(`${apiBaseUrl()}/v1/chat/sessions`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ document_id: documentId, session_token: sessionToken })
  });
  if (!response.ok) throw new Error(await response.text());
  return (await response.json()) as { id: string; document_id: string; expires_at: string };
}

export async function enqueueIngestion(jobId: string) {
  const response = await fetch("/api/ingest", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ jobId })
  });
  if (!response.ok) throw new Error(await response.text());
}

export async function streamChat(input: {
  sessionId: string;
  sessionToken: string;
  content: string;
  onDelta: (value: string) => void;
}) {
  const response = await fetch(`${apiBaseUrl()}/v1/chat/sessions/${input.sessionId}/messages/stream`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      session_token: input.sessionToken,
      content: input.content,
      hybrid: true
    })
  });
  if (!response.ok || !response.body) throw new Error(await response.text());

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let finalMessage: ChatMessage | null = null;
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const events = buffer.split("\n\n");
    buffer = events.pop() ?? "";
    for (const event of events) {
      const dataLine = event.split("\n").find((line) => line.startsWith("data: "));
      if (!dataLine) continue;
      const payload = JSON.parse(dataLine.slice(6));
      if (payload.type === "delta") input.onDelta(payload.content);
      if (payload.role === "assistant") finalMessage = payload as ChatMessage;
    }
  }
  return finalMessage;
}
