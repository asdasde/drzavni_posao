import { mockSearch } from "./mock";
import type { SearchParams, SearchResponse } from "./types";

const API_BASE_URL: string | undefined = import.meta.env.VITE_API_BASE_URL || undefined;

export async function search(params: SearchParams): Promise<SearchResponse> {
  if (!API_BASE_URL) {
    return mockSearch(params);
  }
  const started = performance.now();
  const response = await fetch(`${API_BASE_URL.replace(/\/$/, "")}/search`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
  });
  if (!response.ok) {
    const detail = await response
      .json()
      .then((body) => (typeof body?.detail === "string" ? body.detail : null))
      .catch(() => null);
    throw new Error(detail ?? `Pretraga nije uspela (HTTP ${response.status})`);
  }
  const payload = await response.json();
  return {
    query: params.query,
    method: params.method,
    took_seconds: payload.took_seconds ?? (performance.now() - started) / 1000,
    results: payload.results ?? payload,
  };
}
