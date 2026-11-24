/**
 * Goal 30 — minimal, deterministic frontend integration with the existing
 * VoxSearch search API (`POST {API_PREFIX}/search`).
 *
 * The backend contract is taken from `backend/app/api/search.py`:
 * request `{ query, top_k }`; response `{ query, top_k, count, results }` with
 * each hit carrying `id`, `text`, `start_ms`, `end_ms`, `score`, ...
 *
 * This module contains NO React and NO mock fallback. An injected fetch-like
 * function makes every path testable with Node's built-in test runner.
 */
export const DEFAULT_API_BASE_URL = "http://localhost:8000";
export const API_PREFIX = "/api/v1";
export const DEFAULT_TOP_K = 10;

export type SearchResultSnippet = {
  id: string;
  text: string;
  start_ms: number;
  end_ms: number;
  score: number;
  // Optional metadata actually present in the backend SearchHit schema.
  video_id?: string;
  transcript_asset_id?: string;
  chunk_index?: number;
  duration_ms?: number;
  model_id?: string;
  timestamp_precision?: string;
  youtube_video_id?: string;
  distance?: number;
};

export type SearchApiResponse = {
  query: string | null;
  top_k: number;
  count: number;
  results: SearchResultSnippet[];
};

export type FetchResponseLike = {
  ok: boolean;
  status: number;
  json(): Promise<unknown>;
};

export type FetchLike = (
  url: string,
  init: RequestInit,
) => Promise<FetchResponseLike>;

export class SearchApiError extends Error {
  readonly status?: number;

  constructor(message: string, status?: number) {
    super(message);
    this.name = "SearchApiError";
    this.status = status;
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/** Build the versioned endpoint URL, tolerating a trailing slash in the base. */
export function buildSearchUrl(baseUrl: string = DEFAULT_API_BASE_URL): string {
  return `${baseUrl.replace(/\/+$/, "")}${API_PREFIX}/search`;
}

export type SearchFilters = {
  language?: string | null;
  maxDurationSeconds?: number | null;
};

/** Validate and narrow Goal 33 filters to the backend's exact field names. */
export function normalizeFilters(filters?: SearchFilters): {
  language?: string;
  max_duration_seconds?: number;
} {
  if (!filters) return {};
  const out: { language?: string; max_duration_seconds?: number } = {};
  if (filters.language !== undefined && filters.language !== null) {
    if (typeof filters.language !== "string" || filters.language.length > 16) {
      throw new SearchApiError("Invalid language filter.");
    }
    const cleaned = filters.language.trim();
    if (cleaned.length > 0) out.language = cleaned;
  }
  if (filters.maxDurationSeconds !== undefined && filters.maxDurationSeconds !== null) {
    const value = filters.maxDurationSeconds;
    if (
      typeof value !== "number" ||
      !Number.isInteger(value) ||
      value < 1 ||
      value > 86400
    ) {
      throw new SearchApiError("Invalid duration filter.");
    }
    out.max_duration_seconds = value;
  }
  return out;
}

/** Build the exact JSON body expected by the backend for a text query. */
export function buildSearchRequest(
  query: string,
  topK: number = DEFAULT_TOP_K,
  filters?: SearchFilters,
) {
  return { query, top_k: topK, ...normalizeFilters(filters) };
}

/** Convert a safe backend error payload into a short, credential-free message. */
export function describeApiError(status: number, payload: unknown): string {
  if (status === 422) {
    if (isRecord(payload) && typeof payload.detail === "string") {
      return payload.detail;
    }
    return "The search request was rejected as invalid.";
  }
  if (status === 503) {
    return "Search is temporarily unavailable. Please try again.";
  }
  return "The search API could not complete the request.";
}

/** Validate and narrow an untrusted JSON body into the Goal 30 response shape. */
export function parseSearchResponse(payload: unknown): SearchApiResponse {
  if (!isRecord(payload)) {
    throw new SearchApiError("The search API returned an invalid response.");
  }
  const { query, top_k: topK, count, results } = payload;
  if (!Number.isInteger(count) || (count as number) < 0) {
    throw new SearchApiError("The search API returned an invalid result count.");
  }
  if (!Number.isInteger(topK) || (topK as number) < 1) {
    throw new SearchApiError("The search API returned an invalid top_k.");
  }
  if (query !== null && typeof query !== "string") {
    throw new SearchApiError("The search API returned an invalid query field.");
  }
  if (!Array.isArray(results) || results.length !== count) {
    throw new SearchApiError("The search API returned an invalid result list.");
  }

  const parsed: SearchResultSnippet[] = results.map((item) => {
    if (!isRecord(item)) {
      throw new SearchApiError("The search API returned a malformed result.");
    }
    const { id, text, start_ms: startMs, end_ms: endMs, score } = item;
    if (typeof id !== "string" || typeof text !== "string") {
      throw new SearchApiError("The search API returned a malformed result.");
    }
    if (
      !Number.isInteger(startMs) ||
      (startMs as number) < 0 ||
      !Number.isInteger(endMs) ||
      (endMs as number) < 0
    ) {
      throw new SearchApiError("The search API returned invalid timestamps.");
    }
    if (typeof score !== "number" || !Number.isFinite(score)) {
      throw new SearchApiError("The search API returned an invalid score.");
    }

    // Optional SearchHit metadata: reject malformed values, omit absent ones.
    const optional: Record<string, unknown> = {};
    for (const key of [
      "video_id",
      "transcript_asset_id",
      "model_id",
      "timestamp_precision",
      "youtube_video_id",
    ] as const) {
      const value = item[key];
      if (value !== undefined) {
        if (typeof value !== "string") {
          throw new SearchApiError("The search API returned invalid metadata.");
        }
        optional[key] = value;
      }
    }
    for (const key of ["chunk_index", "duration_ms"] as const) {
      const value = item[key];
      if (value !== undefined) {
        if (!Number.isInteger(value) || (value as number) < 0) {
          throw new SearchApiError("The search API returned invalid metadata.");
        }
        optional[key] = value;
      }
    }
    if (item.distance !== undefined) {
      if (
        typeof item.distance !== "number" ||
        !Number.isFinite(item.distance)
      ) {
        throw new SearchApiError("The search API returned an invalid distance.");
      }
      optional.distance = item.distance;
    }

    return {
      id,
      text,
      start_ms: startMs as number,
      end_ms: endMs as number,
      score,
      ...optional,
    };
  });

  return {
    query: query as string | null,
    top_k: topK as number,
    count: count as number,
    results: parsed,
  };
}

/**
 * Small, framework-free request lifecycle tracker used by the homepage.
 * Extracted so stale-response and abort races are deterministically testable.
 */
export type RequestTracker = {
  next(): { id: number; controller: AbortController };
  isCurrent(id: number): boolean;
  abortAll(): void;
};

export function createRequestTracker(): RequestTracker {
  let counter = 0;
  let active: AbortController | null = null;
  return {
    next() {
      active?.abort(); // supersede any previous request
      const id = ++counter;
      active = new AbortController();
      return { id, controller: active };
    },
    isCurrent(id: number) {
      return id === counter;
    },
    abortAll() {
      active?.abort();
      active = null;
      counter += 1; // makes every in-flight response stale
    },
  };
}

export class SearchApiClient {
  private readonly _fetch: FetchLike;
  private readonly _baseUrl: string;

  constructor(fetchImpl: FetchLike = fetch as FetchLike, baseUrl?: string) {
    this._fetch = fetchImpl;
    this._baseUrl = baseUrl ?? DEFAULT_API_BASE_URL;
  }

  async search(
    query: string,
    signal?: AbortSignal,
    filters?: SearchFilters,
  ): Promise<SearchApiResponse> {
    let response: FetchResponseLike;
    try {
      response = await this._fetch(buildSearchUrl(this._baseUrl), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(buildSearchRequest(query, DEFAULT_TOP_K, filters)),
        signal,
      });
    } catch (error) {
      if (error instanceof Error && error.name === "AbortError") throw error;
      throw new SearchApiError("Network error: could not reach the search API.");
    }

    let payload: unknown;
    try {
      payload = await response.json();
    } catch {
      throw new SearchApiError("The search API returned an invalid response.");
    }

    if (!response.ok) {
      throw new SearchApiError(
        describeApiError(response.status, payload),
        response.status,
      );
    }
    return parseSearchResponse(payload);
  }
}
