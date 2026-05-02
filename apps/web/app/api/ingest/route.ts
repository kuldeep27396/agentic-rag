import { NextResponse } from "next/server";
import { apiBaseUrl } from "@/lib/env";

export async function POST(request: Request) {
  const { jobId } = (await request.json()) as { jobId?: string };
  if (!jobId) {
    return NextResponse.json({ error: "Missing jobId" }, { status: 400 });
  }

  const destination = `${apiBaseUrl()}/v1/ingestion/jobs/${jobId}/run`;
  const qstashToken = process.env.QSTASH_TOKEN;

  if (!qstashToken) {
    const response = await fetch(destination, { method: "POST" });
    if (!response.ok) {
      return NextResponse.json({ error: await response.text() }, { status: response.status });
    }
    return NextResponse.json({ queued: false, mode: "direct-dev" });
  }

  const response = await fetch("https://qstash.upstash.io/v2/publish/" + destination, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${qstashToken}`,
      "content-type": "application/json"
    },
    body: JSON.stringify({ jobId })
  });
  if (!response.ok) {
    return NextResponse.json({ error: await response.text() }, { status: response.status });
  }
  return NextResponse.json({ queued: true });
}
