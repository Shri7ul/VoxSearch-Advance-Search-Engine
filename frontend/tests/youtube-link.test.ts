import test from "node:test";
import assert from "node:assert/strict";

import {
  buildYouTubeTimestampUrl,
  formatYouTubeSeconds,
} from "../lib/youtube-link.ts";

const VALID_ID = "dQw4w9WgXcQ";

test("builds an https watch URL with a whole-second timestamp", () => {
  assert.equal(
    buildYouTubeTimestampUrl(VALID_ID, 125400),
    `https://www.youtube.com/watch?v=${VALID_ID}&t=125s`,
  );
});

test("millisecond conversion floors and never seeks past the start", () => {
  assert.equal(formatYouTubeSeconds(0), 0);
  assert.equal(formatYouTubeSeconds(999), 0);
  assert.equal(formatYouTubeSeconds(1000), 1);
  assert.equal(formatYouTubeSeconds(61045), 61);
});

test("invalid video IDs produce no link", () => {
  for (const id of [
    "",
    "short",
    "too-long-video-id",
    "has space here!",
    "javascript:alert(1)",
    null,
    undefined,
    123,
    {},
  ]) {
    assert.equal(buildYouTubeTimestampUrl(id, 1000), null);
  }
});

test("invalid timestamps produce no link", () => {
  for (const ms of [-1, 1.5, Number.NaN, Number.POSITIVE_INFINITY, "1000", null]) {
    assert.equal(buildYouTubeTimestampUrl(VALID_ID, ms), null);
  }
});

test("URL is fixed-scheme and contains no injected query parameters", () => {
  const url = buildYouTubeTimestampUrl(VALID_ID, 0);
  assert.ok(url?.startsWith("https://www.youtube.com/watch?v="));
  assert.ok(!url?.includes('"'));
  assert.ok(!url?.includes("<"));
  assert.equal(url?.split("&").length, 2);
});

test("underscore and hyphen are valid YouTube id characters", () => {
  assert.ok(buildYouTubeTimestampUrl("a_b-cD12345", 2000));
});
