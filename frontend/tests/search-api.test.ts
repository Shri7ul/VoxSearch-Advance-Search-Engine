import test from "node:test";
import assert from "node:assert/strict";

import {
  DEFAULT_TOP_K,
  SearchApiClient,
  SearchApiError,
  buildSearchRequest,
  buildSearchUrl,
  normalizeFilters,
  createRequestTracker,
  describeApiError,
  parseSearchResponse,
  type FetchLike,
  type FetchResponseLike,
} from "../lib/search-api.ts";

const validPayload = {
  query: "latency",
  top_k: 10,
  count: 1,
  results: [
    {
      id: "9a7d6f2e-0000-4000-8000-000000000001",
      text: "The latency number was about 40 milliseconds.",
      start_ms: 1200,
      end_ms: 4300,
      score: 0.87,
    },
  ],
};

function fakeFetch(response: FetchResponseLike, calls: unknown[] = []): FetchLike {
  return async (url, init) => {
    calls.push({ url, init });
    return response;
  };
}

function okResponse(payload: unknown): FetchResponseLike {
  return { ok: true, status: 200, json: async () => payload };
}

test("URL and request body match the backend contract", () => {
  assert.equal(
    buildSearchUrl("http://localhost:8000/"),
    "http://localhost:8000/api/v1/search",
  );
  assert.deepEqual(buildSearchRequest("latency"), {
    query: "latency",
    top_k: DEFAULT_TOP_K,
  });
});

test("client sends a POST JSON request and returns parsed results", async () => {
  const calls: unknown[] = [];
  const client = new SearchApiClient(fakeFetch(okResponse(validPayload), calls));
  const result = await client.search("  latency  ");
  assert.equal(result.count, 1);
  assert.equal(result.results[0].text, validPayload.results[0].text);
  const call = calls[0] as { url: string; init: RequestInit };
  assert.equal(call.url, "http://localhost:8000/api/v1/search");
  assert.equal(call.init.method, "POST");
  assert.deepEqual(JSON.parse(call.init.body as string), {
    query: "  latency  ",
    top_k: 10,
  });
});

test("Goal 33 filters use the exact backend field names", () => {
  assert.deepEqual(
    buildSearchRequest("latency", 10, {
      language: "en",
      maxDurationSeconds: 900,
    }),
    { query: "latency", top_k: 10, language: "en", max_duration_seconds: 900 },
  );
  assert.deepEqual(buildSearchRequest("latency"), {
    query: "latency",
    top_k: 10,
  });
  assert.deepEqual(buildSearchRequest("latency", 10, { language: "" }), {
    query: "latency",
    top_k: 10,
  });
});

test("invalid filter values are rejected before any request", () => {
  for (const filters of [
    { language: "x".repeat(17) },
    { language: 123 },
    { maxDurationSeconds: 0 },
    { maxDurationSeconds: 90000 },
    { maxDurationSeconds: 1.5 },
    { maxDurationSeconds: "900" },
  ]) {
    assert.throws(() => normalizeFilters(filters), SearchApiError);
  }
});

test("client sends filters in the JSON body", async () => {
  const calls: unknown[] = [];
  const client = new SearchApiClient(fakeFetch(okResponse(validPayload), calls));
  await client.search("query", undefined, { language: "bn", maxDurationSeconds: 300 });
  const call = calls[0] as { init: RequestInit };
  assert.deepEqual(JSON.parse(call.init.body as string), {
    query: "query",
    top_k: 10,
    language: "bn",
    max_duration_seconds: 300,
  });
});

test("422 errors expose only the safe backend detail", async () => {
  const response: FetchResponseLike = {
    ok: false,
    status: 422,
    json: async () => ({ detail: "query must not be empty" }),
  };
  const client = new SearchApiClient(fakeFetch(response));
  await assert.rejects(
    () => client.search("query"),
    (error: unknown) => {
      assert.ok(error instanceof SearchApiError);
      assert.equal(error.status, 422);
      assert.equal(error.message, "query must not be empty");
      return true;
    },
  );
});

test("503 and 500 errors use generic safe messages", () => {
  assert.match(describeApiError(503, {}), /temporarily unavailable/);
  assert.match(describeApiError(500, { detail: "dsn=secret" }), /could not complete/);
  assert.ok(!describeApiError(500, { detail: "dsn=secret" }).includes("secret"));
});

test("network failures are explicit and never fall back to mock data", async () => {
  const failing: FetchLike = async () => {
    throw new TypeError("fetch failed");
  };
  const client = new SearchApiClient(failing);
  await assert.rejects(
    () => client.search("query"),
    /Network error: could not reach the search API/,
  );
});

test("abort errors propagate so the UI can ignore stale requests", async () => {
  const aborting: FetchLike = async () => {
    const error = new Error("aborted");
    error.name = "AbortError";
    throw error;
  };
  const client = new SearchApiClient(aborting);
  await assert.rejects(
    () => client.search("query"),
    (error: unknown) => error instanceof Error && error.name === "AbortError",
  );
});

test("malformed success payloads are rejected", () => {
  const badPayloads: unknown[] = [
    null,
    { ...validPayload, count: 2 },
    { ...validPayload, results: "not-a-list" },
    { ...validPayload, results: [{ ...validPayload.results[0], score: "high" }] },
    { ...validPayload, results: [{ ...validPayload.results[0], start_ms: -1 }] },
  ];
  for (const payload of badPayloads) {
    assert.throws(() => parseSearchResponse(payload), SearchApiError);
  }
});

test("empty result set is a valid success response", async () => {
  const empty = { query: "missing", top_k: 10, count: 0, results: [] };
  const client = new SearchApiClient(fakeFetch(okResponse(empty)));
  const result = await client.search("missing");
  assert.equal(result.count, 0);
  assert.deepEqual(result.results, []);
});

test("request tracker supersedes and aborts older requests", () => {
  const tracker = createRequestTracker();
  const first = tracker.next();
  const second = tracker.next();
  assert.equal(first.controller.signal.aborted, true);
  assert.equal(second.controller.signal.aborted, false);
  assert.equal(tracker.isCurrent(first.id), false);
  assert.equal(tracker.isCurrent(second.id), true);
});

test("abortAll invalidates every in-flight response", () => {
  const tracker = createRequestTracker();
  const active = tracker.next();
  tracker.abortAll();
  assert.equal(active.controller.signal.aborted, true);
  assert.equal(tracker.isCurrent(active.id), false);
  assert.equal(tracker.next().controller.signal.aborted, false);
});

test("422 with structured detail never exposes raw input values", () => {
  assert.equal(
    describeApiError(422, {
      detail: [{ msg: "bad", input: { query: "secret-user-text" } }],
    }),
    "The search request was rejected as invalid.",
  );
  assert.ok(!describeApiError(422, { detail: ["secret"] }).includes("secret"));
});

test("known SearchHit metadata is preserved and unknown fields are not invented", () => {
  const payload = {
    ...validPayload,
    results: [
      {
        ...validPayload.results[0],
        distance: 0.13,
        video_id: "video-id",
        transcript_asset_id: "asset-id",
        chunk_index: 3,
        model_id: "BAAI/bge-m3",
        word_count: 9, // real backend field, not used by the UI
        unexpected_field: "ignored",
      },
    ],
  };
  const parsed = parseSearchResponse(payload);
  const hit = parsed.results[0];
  assert.equal(hit.video_id, "video-id");
  assert.equal(hit.transcript_asset_id, "asset-id");
  assert.equal(hit.chunk_index, 3);
  assert.equal(hit.distance, 0.13);
  assert.ok(!("unexpected_field" in hit));
  assert.ok(!("word_count" in hit));
});

test("malformed optional metadata is rejected", () => {
  for (const extra of [
    { video_id: 123 },
    { chunk_index: -1 },
    { distance: Number.NaN },
    { model_id: null },
  ]) {
    const payload = {
      ...validPayload,
      results: [{ ...validPayload.results[0], ...extra }],
    };
    assert.throws(() => parseSearchResponse(payload), SearchApiError);
  }
});

test("invalid JSON is reported without leaking internals", async () => {
  const response: FetchResponseLike = {
    ok: true,
    status: 200,
    json: async () => {
      throw new Error("unexpected token at /backend/path");
    },
  };
  const client = new SearchApiClient(fakeFetch(response));
  await assert.rejects(
    () => client.search("query"),
    (error: unknown) => {
      assert.ok(error instanceof SearchApiError);
      assert.ok(!error.message.includes("/backend/path"));
      return true;
    },
  );
});
