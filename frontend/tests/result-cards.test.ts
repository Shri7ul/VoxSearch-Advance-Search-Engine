import test from "node:test";
import assert from "node:assert/strict";

import {
  formatTimestamp,
  toResultCardModel,
} from "../lib/result-cards.ts";
import type { SearchResultSnippet } from "../lib/search-api.ts";

const base: SearchResultSnippet = {
  id: "chunk-0001",
  text: "The measured latency was about forty milliseconds.",
  start_ms: 1200,
  end_ms: 4300,
  score: 0.87,
};

test("timestamp formatting is deterministic and bounded-safe", () => {
  assert.equal(formatTimestamp(0), "0:00.000");
  assert.equal(formatTimestamp(1200), "0:01.200");
  assert.equal(formatTimestamp(61045), "1:01.045");
  assert.equal(formatTimestamp(-1), "—");
  assert.equal(formatTimestamp(1.5), "—");
});

test("card maps actual SearchHit fields with a 1-based rank", () => {
  const card = toResultCardModel(
    {
      ...base,
      video_id: "abcdefgh-1234-5678-9012-abcdefghijkl",
      transcript_asset_id: "11112222-3333-4444-5555-666677778888",
      chunk_index: 4,
      model_id: "BAAI/bge-m3",
      timestamp_precision: "word",
    },
    0,
  );
  assert.equal(card.rank, 1);
  assert.equal(card.videoLabel, "abcdefgh…");
  assert.equal(card.transcriptLabel, "11112222…");
  assert.equal(card.chunkLabel, "Chunk 4");
  assert.equal(card.timing, "0:01.200 – 0:04.300");
  assert.equal(card.relevance, "Relevance 0.870");
  assert.equal(card.modelLabel, "BAAI/bge-m3");
  assert.equal(card.precisionLabel, "word");
  assert.equal(card.text, base.text);
});

test("missing optional metadata is omitted, not fabricated", () => {
  const card = toResultCardModel(base, 2);
  assert.equal(card.rank, 3);
  assert.equal(card.videoLabel, null);
  assert.equal(card.transcriptLabel, null);
  assert.equal(card.chunkLabel, null);
  assert.equal(card.modelLabel, null);
  assert.equal(card.precisionLabel, null);
});

test("unicode and long transcript text are preserved verbatim", () => {
  const text = "ল্যাটেন্সি — " + "word ".repeat(500);
  const card = toResultCardModel({ ...base, text }, 0);
  assert.equal(card.text, text);
  assert.ok(card.text.includes("ল্যাটেন্সি"));
});

test("blank optional metadata does not produce empty labels", () => {
  const card = toResultCardModel(
    {
      ...base,
      video_id: "   ",
      transcript_asset_id: "",
      model_id: " \t ",
      timestamp_precision: "",
    },
    0,
  );
  assert.equal(card.videoLabel, null);
  assert.equal(card.transcriptLabel, null);
  assert.equal(card.modelLabel, null);
  assert.equal(card.precisionLabel, null);
});

test("multiple results keep distinct keys and ranks", () => {
  const cards = [base, { ...base, id: "chunk-0002" }].map(toResultCardModel);
  assert.equal(new Set(cards.map((c) => c.id)).size, 2);
  assert.deepEqual(cards.map((c) => c.rank), [1, 2]);
});

test("card model exposes no links or filter state", () => {
  const card = toResultCardModel(base, 0);
  assert.ok(!("href" in card));
  assert.ok(!("filters" in card));
  assert.ok(!("embedding" in card));
});
