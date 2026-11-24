# VoxSearch Frontend — Goal 29 search homepage

Next.js + TypeScript search homepage.

- Goal 29: query text box with deterministic validation (`lib/search-input.ts`)
  and accessible local status.
- Goal 30: `lib/search-api.ts` integrates with `POST /api/v1/search` — loading,
  success, API/network error states, duplicate-request suppression and stale
  response protection. Results are rendered as plain text only.
- Goal 31: accessible, responsive result cards (`lib/result-cards.ts`) showing
  video / transcript identity, chunk index, verbatim text, plain-text timing and
  relevance. Timestamp links (32) and filters (33) are NOT implemented.

## Layout

- `app/layout.tsx` — root metadata and document shell.
- `app/page.tsx` — the homepage client component.
- `app/globals.css` — minimal dark theme.
- `lib/search-input.ts` — pure validation, no React and no network.
- `lib/search-api.ts` — injectable-fetch search API client (Goal 30).
- `lib/result-cards.ts` — pure card view-model and timestamp formatter (Goal 31).
- `tests/search-input.test.ts`, `tests/search-api.test.ts`,
  `tests/result-cards.test.ts` — Node built-in tests.

Set `NEXT_PUBLIC_API_BASE_URL` to override the default `http://localhost:8000`
backend base URL. No credentials belong in this variable.

## Commands (from `frontend/`)

```
npm install     # one-time, only when deliberately provisioning Node deps
npm test        # node --test tests/*.test.ts (no dependencies required)
npm run dev     # Next.js dev server after install
```

Goals 29–30 were validated with `npm test` (Node 24 native TypeScript support,
mocked fetch boundary). No packages were installed; real browser rendering,
`next build`, and live API calls remain unverified in this environment.
