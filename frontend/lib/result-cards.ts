/**
 * Goal 31 — pure view-model construction for transcript result cards.
 *
 * Only fields actually returned by the backend SearchHit schema are consumed.
 * No links, navigation or filters are produced (Goals 32–33). Duplicate-safe
 * keys and deterministic formatting keep rendering stable across requests.
 */
import type { SearchResultSnippet } from "./search-api.ts";
import { buildYouTubeTimestampUrl } from "./youtube-link.ts";

export type ResultCardModel = {
  id: string;
  rank: number;
  videoLabel: string | null;
  transcriptLabel: string | null;
  chunkLabel: string | null;
  text: string;
  timing: string;
  timestampUrl: string | null;
  relevance: string | null;
  modelLabel: string | null;
  precisionLabel: string | null;
};

/** Format integer milliseconds as `m:ss.mmm` without locale dependence. */
export function formatTimestamp(milliseconds: number): string {
  if (!Number.isInteger(milliseconds) || milliseconds < 0) {
    return "—";
  }
  const minutes = Math.floor(milliseconds / 60000);
  const seconds = Math.floor((milliseconds % 60000) / 1000);
  const millis = milliseconds % 1000;
  return `${minutes}:${String(seconds).padStart(2, "0")}.${String(millis).padStart(3, "0")}`;
}

function nonEmpty(value: string | undefined): string | null {
  if (typeof value !== "string" || value.trim().length === 0) return null;
  return value;
}

function shortId(value: string | undefined): string | null {
  const clean = nonEmpty(value);
  if (clean === null) return null;
  return clean.length > 8 ? `${clean.slice(0, 8)}…` : clean;
}

/** Build one deterministic card model from a parsed SearchHit. */
export function toResultCardModel(
  result: SearchResultSnippet,
  index: number,
): ResultCardModel {
  return {
    id: result.id,
    rank: index + 1,
    videoLabel: shortId(result.video_id),
    transcriptLabel: shortId(result.transcript_asset_id),
    chunkLabel:
      typeof result.chunk_index === "number" ? `Chunk ${result.chunk_index}` : null,
    text: result.text,
    timing: `${formatTimestamp(result.start_ms)} – ${formatTimestamp(result.end_ms)}`,
    timestampUrl: buildYouTubeTimestampUrl(
      result.youtube_video_id,
      result.start_ms,
    ),
    relevance:
      typeof result.score === "number" && Number.isFinite(result.score)
        ? `Relevance ${result.score.toFixed(3)}`
        : null,
    modelLabel: nonEmpty(result.model_id),
    precisionLabel: nonEmpty(result.timestamp_precision),
  };
}
