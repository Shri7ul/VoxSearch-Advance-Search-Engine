# VoxSearch

A YouTube-first semantic video search engine that retrieves **spoken-content segments** and returns **timestamped video links**. Instead of matching only video titles or descriptions, VoxSearch is designed to search inside what was actually said in a video and link directly to the moment in the video where a matching phrase occurs.

> **Status:** The project has moved beyond initial scaffolding. Goals 01–13 foundations are implemented and version-controlled: a FastAPI backend, the `voxsearch` relational schema with four Alembic revisions, transcript ingestion / normalization / segmentation, word-level timestamp alignment (**Goal 12 — ACCEPTED**), and deterministic sentence-aware chunking (**Goal 13 — ACCEPTED**). **Goal 14 — ACCEPTED** has prepared and validated an **isolated, CPU-only BGE-M3 environment** (installation / smoke-test validation ONLY). **Goal 15 — ACCEPTED** adds dense **BGE-M3 embedding generation** (1024-dimensional vectors) as a torch-free contract layer plus an isolated worker. **Goal 16 — ACCEPTED** adds **PostgreSQL + pgvector persistence** for 1024-dimensional BGE-M3 embeddings (single `voxsearch.video_chunks` table, dependency-free `VectorType` UDT, atomic persistence service with idempotent upsert; live PostgreSQL/pgvector runtime verification remains pending due to lack of a local instance; unapplied Alembic migration). Cosine similarity search (**Goal 17 — ACCEPTED**: pgvector `<=>` cosine-distance retrieval with strict query-vector validation) and the search HTTP endpoint (**Goal 18 — ACCEPTED**: `POST /api/v1/search`) and natural-language query embedding (**Goal 19 — ACCEPTED**: isolated-worker generated query vector, no model loaded in the API process; real-weights inference not verified) and top-10 vector retrieval orchestration (**Goal 20 — ACCEPTED**; live DB verification pending) and the structured JSON response schema (**Goal 21 - ACCEPTED**; live DB verification pending) and PostgreSQL full-text keyword retrieval (**Goal 22 — ACCEPTED**; offline SQL only, live DB verification pending) and dual dense+lexical retrieval (**Goal 23 - ACCEPTED**) and RRF combined ranking (**Goal 24 - ACCEPTED**) and duplicate removal (**Goal 25 - ACCEPTED**), and BGE reranker setup (**Goal 26 - ACCEPTED**; isolated worker only, real model load not verified). Reranking, the search homepage UI (Goal 29 is ACCEPTED) and its backend API integration (Goal 30 is ACCEPTED) and result cards (Goal 31 is ACCEPTED) and YouTube timestamp links (Goal 32 is ACCEPTED) and search refinement filters (Goal 33 is ACCEPTED), and YouTube API authentication setup (Goal 34 is ACCEPTED) and bounded video discovery (Goal 35 is ACCEPTED) and channel registration (Goal 36 is ACCEPTED) and creator OAuth workflow (Goal 37 is ACCEPTED) and authorized transcript adapter (Goal 38 is ACCEPTED) and Redis queue setup (Goal 39 is ACCEPTED) and Celery background-job setup (Goal 40 is ACCEPTED) and bounded failed-job retries (Goal 41 is ACCEPTED), YouTube ingestion, background workers and deployment remain **planned**. See `progress.md` for the canonical numbered Goal tracker and `backend/README.md` for implementation detail.

---

## Project description

VoxSearch is intended to be a search engine over the *transcripts* of YouTube videos. A user types a natural-language query, and the system returns the most semantically relevant short transcript segments together with a deep link to the exact timestamp in the source video (e.g. `https://www.youtube.com/watch?v=VIDEO_ID&t=SECONDS`).

The project is organized as a monorepo so that the web frontend, API backend, machine-learning components, background workers, database schema, and deployment infrastructure can evolve together in a single version-controlled codebase.

---

## Core objective

Enable **semantic, timestamp-accurate retrieval of spoken content** from YouTube videos:

1. Ingest authorized video transcripts while preserving word/segment timings.
2. Turn transcript segments into dense vector embeddings.
3. Retrieve candidate segments for a query using hybrid (dense + lexical) search.
4. Re-rank candidates for relevance.
5. Return results as timestamped, deep-linked video segments.

The emphasis is on **finding the right moment inside a video**, not merely the right video.

---

## Main features (planned)

- **Semantic search over spoken content** — query by meaning, not just keywords.
- **Timestamped YouTube deep links** — jump straight to the relevant moment.
- **Hybrid retrieval** — combine dense vector search with full-text/lexical matching and fuse rankings.
- **Reranking** — a cross-encoder reranker to improve result ordering.
- **Transcript processing pipeline** — normalization, sentence segmentation, chunking, and timestamp preservation.
- **Background ingestion & indexing** — queue-driven workers so indexing runs without manual steps.
- **YouTube integration** — video discovery, channel registry, and authorized transcript sources.
- **Evaluation tooling** — retrieval metrics (Recall@10, MRR, nDCG), timestamp accuracy, and latency benchmarks.
- **Search frontend** — a web UI with result cards, timestamp links, and basic filters.

> Most of these features remain **planned**. The transcript processing pipeline (normalization, sentence segmentation, chunking, timestamp preservation) is now implemented (Goals 09–13); at the Goal 01 stage none were implemented.

---

## Planned technology stack

> The stack below is **indicative of the target architecture**. Only the FastAPI / SQLAlchemy / psycopg / Alembic / pytest backend subset is currently configured and installed (see `backend/requirements.txt`); the remaining layers are planned.

| Layer | Planned technology |
| --- | --- |
| Frontend | Next.js (React, TypeScript) |
| Backend API | FastAPI (Python) |
| Database | PostgreSQL (via Supabase) with `pgvector` for embeddings |
| Auth / data platform | Supabase (URL + anon/service-role keys) |
| Caching / queue | Redis, Celery workers |
| Embeddings | `BGE-M3` (configurable via `EMBEDDING_MODEL`) |
| Reranker | BGE reranker (configurable via `RERANKER_MODEL`) |
| Transcript source | YouTube Data API (`YOUTUBE_API_KEY`) |
| Containerization / local dev | Docker, Docker Compose |
| Monitoring | Logs and metrics (to be defined) |

Backend Python dependencies are pinned and installed in the standalone `backend/.venv` (Python 3.12.10), and the relational schema exists as SQLAlchemy models plus Alembic migrations. No live Supabase/PostgreSQL project, Redis, ML model, or production service is configured, and no production database has been migrated.

---

## High-level architecture (planned)

```
                  ┌──────────────┐
   user query ──► │   frontend   │  (Next.js search UI)
                  └──────┬───────┘
                         │ HTTP
                  ┌──────▼───────┐
                  │   backend    │  (FastAPI /search endpoint)
                  └──────┬───────┘
                         │
        ┌────────────────┼─────────────────┐
        │                │                 │
  ┌─────▼─────┐   ┌──────▼──────┐   ┌──────▼──────┐
  │ embeddings│   │  retrieval  │   │  reranker   │
  │  (ML)     │   │  (hybrid)   │   │   (ML)      │
  └─────┬─────┘   └──────┬──────┘   └──────┬──────┘
        │                │                 │
        └────────────────┼─────────────────┘
                         │
                  ┌──────▼───────┐
                  │   database   │  Postgres + pgvector (Supabase)
                  └──────▲───────┘
                         │
        ┌────────────────┴─────────────────┐
        │            workers               │  discovery → ingestion → indexing
        │   (Redis queue + Celery)         │  (+ compliance)
        └────────────────▲─────────────────┘
                         │
                  ┌──────┴───────┐
                  │ YouTube API  │  authorized transcript sources
                  └──────────────┘
```

Flow: authorized transcripts are discovered and ingested by background workers, chunked with preserved timestamps, embedded, and indexed into the vector database. At query time the backend embeds the query, retrieves candidates via hybrid search, reranks them, and returns timestamped segments to the frontend.

This is a **design sketch only** — the pipeline is not implemented.

---

## Repository structure

```
voxsearch/
├── frontend/            # Next.js web application (planned)
├── backend/             # FastAPI service (planned)
├── ml/
│   ├── embeddings/      # Embedding model integration (planned)
│   ├── reranker/        # Reranking model integration (planned)
│   ├── asr/             # Speech/transcript handling (planned)
│   ├── alignment/       # Timestamp alignment (planned)
│   └── evaluation/      # Retrieval & timestamp evaluation (planned)
├── workers/
│   ├── discovery/       # Video/channel discovery (planned)
│   ├── ingestion/       # Transcript ingestion (planned)
│   ├── indexing/        # Embedding & index updates (planned)
│   └── compliance/      # Source authorization/compliance (planned)
├── database/
│   ├── migrations/      # Schema migrations (planned; none created yet)
│   └── seeds/           # Seed data (planned; none created yet)
├── infrastructure/
│   ├── docker/          # Dockerfiles / compose (planned)
│   └── monitoring/      # Logs & metrics config (planned)
├── docs/                # Project documentation
├── tests/               # Test suites (planned)
├── .gitignore           # VCS exclusions
├── .env.example         # Environment variable template (placeholders only)
└── README.md            # This file
```

Empty directories are kept in version control using `.gitkeep` placeholder files.

---

## Development roadmap

The roadmap is tracked in phases; the canonical numbered Goal tracker is `progress.md`. Goals 12, 13 and 14 are ACCEPTED (Goal 14 is committed and pushed at `969c5b5`); Goal 15 (dense BGE-M3 embedding generation) is ACCEPTED; Goal 16 (vector database persistence) is ACCEPTED (live PostgreSQL verification pending).

| Phase | Goals | Focus | Status |
| --- | --- | --- | --- |
| Phase 1 — Project Foundation | 01–04 | Repo init, FastAPI setup, Next.js setup, Docker | **01, 02 implemented; 03–04 planned** |
| Phase 2 — Database Foundation | 05–08 | PostgreSQL, pgvector, video & transcript tables | **05, 07, 08 implemented; 06 (pgvector) planned** |
| Phase 3 — Transcript Processing | 09–13 | Normalization, segmentation, timestamps, chunking | **09–11 implemented; 12–13 ACCEPTED** |
| Phase 4 — Embedding Engine | 14–17 | BGE-M3, vector insert, cosine search | **14–17 ACCEPTED** |
| Phase 5 — Semantic Search API | 18–21 | `/search` endpoint, query embedding, response schema | **18–21 ACCEPTED** |
| Phase 6 — Hybrid Search | 22–25 | PostgreSQL FTS, dense + lexical, RRF, dedup | **22–25 ACCEPTED** |
| Phase 7 — Reranking Engine | 26–28 | BGE reranker, candidate reranking, context | **26–28 ACCEPTED** |
| Phase 8 — Search Frontend | 29–33 | Search UI, API integration, result cards, links | **29–33 ACCEPTED** |
| Phase 9 — YouTube Integration | 34–38 | YouTube API, discovery, channel registry, auth | **34–38 ACCEPTED** |
| Phase 10 — Background Processing | 39–42 | Redis, Celery, retries, indexing automation | **39–41 ACCEPTED; 42 planned** |
| Phase 11 — Evaluation & Optimization | 43–46 | Test queries, Recall/MRR/nDCG, timestamp MAE, P95 | Planned |
| Phase 12 — Production Deployment | 47–50 | Production Docker, monitoring, backup/security, launch | Planned |

> Roadmap detail lives in `progress.md`. This README summarizes it for orientation.

---

## Current development status

**Goals 12, 13 and 14 are ACCEPTED** (canonical tracker: `progress.md`); Goal 14 is committed and pushed at `969c5b5`. **Goal 15 (dense BGE-M3 embedding generation) is ACCEPTED.** **Goal 16 (PostgreSQL + pgvector persistence) is ACCEPTED** (live PostgreSQL/pgvector runtime verification pending). The repository now contains the FastAPI backend (`GET /health`, `GET /health/db`), the `voxsearch` relational schema with five Alembic revisions (new head `c9d4f1a6b2e3`), the transcript ingestion / normalization / segmentation / word-alignment / chunking pipeline, the torch-free dense embedding contract layer plus isolated worker, the single `video_chunks` table definition with dependency-free `VectorType(1024)` UDT, the atomic chunk persistence service with idempotent upsert, and an offline test suite. Cosine similarity search (Goal 17 is ACCEPTED) and the `POST /api/v1/search` endpoint (Goal 18 is ACCEPTED; it accepts `query` text or a precomputed `query_vector`), natural-language query embedding (Goal 19 is ACCEPTED via the isolated worker; real-weights inference not verified), top-10 vector retrieval orchestration (Goal 20 is ACCEPTED; live DB verification pending), structured JSON search response schema (Goal 21 is ACCEPTED; live DB verification pending), PostgreSQL full-text keyword retrieval (Goal 22 is ACCEPTED; offline SQL only) and dual dense+lexical retrieval (Goal 23 is ACCEPTED), RRF combined ranking (Goal 24 is ACCEPTED), duplicate removal (Goal 25 is ACCEPTED), BGE reranker setup (Goal 26 is ACCEPTED; real model load not verified) and candidate reranking (Goal 27 is ACCEPTED; real inference not verified) and context-aware matching (Goal 28 is ACCEPTED; real inference not verified), the search homepage UI (Goal 29 is IMPLEMENTED — PENDING REVIEW), YouTube ingestion, background workers and deployment remain planned. `backend/.env` is local-only and never committed; production Supabase has NOT been touched by Goals 12–16, and the pending migrations require separate explicit approval.

The section below is the **historical Goal 01 (Repository Initialization) record**, retained for traceability; it describes the initial scaffolding commit, not the current state.

### Historical — Goal 01 — Repository Initialization ✅

Completed in that change:

- Monorepo directory structure with placeholder files.
- `README.md` with project overview and roadmap.
- `.gitignore` covering Python, Node/Next.js, secrets, ML artifacts, datasets, and local dev artifacts.
- `.env.example` with placeholder variables only.

Intentionally out of scope for Goal 01 (since addressed in later goals, to the extent described above):

- No backend, frontend, database tables, or Supabase configuration.
- No ML models, workers, or API endpoints implemented.
- No packages installed.
- See `.env.example` for the configuration variables that will be needed later; copy it to `.env` and fill in real values only in a local, untracked environment.

Later goals cover the remaining roadmap phases (see `progress.md`).
