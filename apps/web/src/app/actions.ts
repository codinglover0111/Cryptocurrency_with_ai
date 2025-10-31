"use server";

import "server-only";

const DEFAULT_API_BASE = "http://localhost:8000/api";

function getApiBase() {
  return process.env.BACKEND_API_BASE_URL?.replace(/\/$/, "") ?? DEFAULT_API_BASE;
}

async function fetchJson<T>(path: string, init?: RequestInit): Promise<T> {
  const base = getApiBase();
  const url = path.startsWith("http") ? path : `${base}${path.startsWith("/") ? "" : "/"}${path}`;
  const response = await fetch(url, {
    ...init,
    headers: {
      Accept: "application/json",
      ...(init?.headers ?? {})
    },
    next: {
      revalidate: 5
    }
  });

  if (!response.ok) {
    throw new Error(`Request failed (${response.status} ${response.statusText})`);
  }

  return (await response.json()) as T;
}

export async function loadStatus() {
  return fetchJson<unknown>("/status");
}

export async function loadStats() {
  return fetchJson<unknown>("/stats");
}

export async function loadJournals(params?: {
  symbol?: string;
  types?: string;
  limit?: number;
  todayOnly?: boolean;
}) {
  const search = new URLSearchParams();
  if (params?.symbol) search.set("symbol", params.symbol);
  if (params?.types) search.set("types", params.types);
  if (params?.limit) search.set("limit", params.limit.toString());
  if (params?.todayOnly === false) search.set("today_only", "0");
  const suffix = search.toString() ? `?${search.toString()}` : "";
  return fetchJson<unknown>(`/journals${suffix}`);
}
