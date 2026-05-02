export function apiBaseUrl() {
  return process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";
}

export const MAX_UPLOAD_BYTES = 25 * 1024 * 1024;

