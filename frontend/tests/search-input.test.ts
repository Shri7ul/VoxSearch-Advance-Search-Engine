import test from "node:test";
import assert from "node:assert/strict";

import {
  MAX_QUERY_CHARACTERS,
  formatAcceptedMessage,
  validateQuery,
} from "../lib/search-input.ts";

test("valid query keeps verbatim text, including surrounding whitespace", () => {
  assert.deepEqual(validateQuery("  latency  "), {
    ok: true,
    value: "  latency  ",
  });
});

test("unicode and punctuation are accepted unchanged", () => {
  const query = "কেন latency বেশি? — exact term: BGE-M3";
  assert.deepEqual(validateQuery(query), { ok: true, value: query });
});

test("non-string input is rejected", () => {
  for (const value of [null, undefined, 123, true, {}, ["query"]]) {
    const result = validateQuery(value);
    assert.equal(result.ok, false);
    assert.equal(result.error, "Query must be text.");
  }
});

test("empty and whitespace-only queries are rejected", () => {
  for (const value of ["", "   ", "\t\n "]) {
    const result = validateQuery(value);
    assert.equal(result.ok, false);
    assert.equal(result.error, "Enter a search query.");
  }
});

test("accepted status message preserves the query verbatim", () => {
  const query = "  latency  ";
  const message = formatAcceptedMessage(query);
  assert.ok(message.includes(query));
  assert.match(message, /Live results arrive in Goal 30/);
  assert.equal(formatAcceptedMessage(query), message);
});

test("control characters are not silently rewritten", () => {
  const query = "line\nbreak\tand\u0000nul";
  assert.deepEqual(validateQuery(query), { ok: true, value: query });
});

test("over-long query is rejected at the documented boundary", () => {
  const tooLong = "x".repeat(MAX_QUERY_CHARACTERS + 1);
  const result = validateQuery(tooLong);
  assert.equal(result.ok, false);
  assert.match(result.error ?? "", /8192/);

  const exactlyMax = "x".repeat(MAX_QUERY_CHARACTERS);
  assert.equal(validateQuery(exactlyMax).ok, true);
});
