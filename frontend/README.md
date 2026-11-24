# VoxSearch Frontend — Goal 29 search homepage

Next.js + TypeScript search homepage. This goal is the **search UI only**:

- query text box with deterministic validation (`lib/search-input.ts`);
- submit button and local status message;
- no API call, no result cards, no timestamp links, no filters — those are
  Goals 30–33.

## Layout

- `app/layout.tsx` — root metadata and document shell.
- `app/page.tsx` — the homepage client component.
- `app/globals.css` — minimal dark theme.
- `lib/search-input.ts` — pure validation, no React and no network.
- `tests/search-input.test.ts` — Node's built-in test runner tests.

## Commands (from `frontend/`)

```
npm install     # one-time, only when deliberately provisioning Node deps
npm test        # node --test tests/*.test.ts (no dependencies required)
npm run dev     # Next.js dev server after install
```

Goal 29 was validated with `npm test` (Node 24 native TypeScript support).
No packages were installed as part of this goal; real browser rendering and
`next build` remain unverified in this environment.
