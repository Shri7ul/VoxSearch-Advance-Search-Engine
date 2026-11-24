/**
 * Goal 29 — pure, framework-free query input validation for the search UI.
 *
 * This module deliberately contains NO network logic: calling the search API
 * belongs to Goal 30. It is TypeScript so Node 24's native type stripping can
 * run the tests in `tests/search-input.test.ts` without installing packages.
 */
export const MAX_QUERY_CHARACTERS = 8192;

export type QueryValidation =
  | { ok: true; value: string }
  | { ok: false; error: string };

/** Build the Goal 29 local-validation status message (no search is performed). */
export function formatAcceptedMessage(value: string): string {
  return `Query accepted: “${value}”. Live results arrive in Goal 30.`;
}

/** Validate a raw text-box value, preserving the verbatim query on success. */
export function validateQuery(raw: unknown): QueryValidation {
  if (typeof raw !== "string") {
    return { ok: false, error: "Query must be text." };
  }
  if (raw.trim().length === 0) {
    return { ok: false, error: "Enter a search query." };
  }
  if (raw.length > MAX_QUERY_CHARACTERS) {
    return {
      ok: false,
      error: `Query must be at most ${MAX_QUERY_CHARACTERS} characters.`,
    };
  }
  return { ok: true, value: raw };
}
