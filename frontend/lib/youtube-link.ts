/**
 * Goal 32 — safe, deterministic YouTube timestamp links.
 *
 * A link is produced ONLY when BOTH a verified 11-character YouTube video id
 * and a valid non-negative integer millisecond timestamp are available.
 * Anything else yields `null` (no fabricated link, no unsafe URL).
 */
const YOUTUBE_ID = /^[A-Za-z0-9_-]{11}$/;
const BASE = "https://www.youtube.com/watch";

export function formatYouTubeSeconds(startMs: unknown): number | null {
  if (
    typeof startMs !== "number" ||
    !Number.isFinite(startMs) ||
    !Number.isInteger(startMs) ||
    startMs < 0
  ) {
    return null;
  }
  // YouTube's `t` is whole seconds; floor never seeks past the chunk start.
  return Math.floor(startMs / 1000);
}

export function buildYouTubeTimestampUrl(
  videoId: unknown,
  startMs: unknown,
): string | null {
  if (typeof videoId !== "string" || !YOUTUBE_ID.test(videoId)) return null;
  const seconds = formatYouTubeSeconds(startMs);
  if (seconds === null) return null;
  return `${BASE}?v=${encodeURIComponent(videoId)}&t=${seconds}s`;
}
