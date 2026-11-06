# VoxSearch — Backend

FastAPI backend for VoxSearch.

> **Scope:** FastAPI application initialization, centralized settings/logging
> and `GET /health` (Goal 02), a reproducible **standalone** Python 3.12
> environment decoupled from Conda (Goals 03–04), a read-only Supabase /
> PostgreSQL connection, the Alembic schema & migration foundation, and the
> repository & service layers (Goals 05–08), plus the **validated transcript
> ingestion foundation** (Goal 09). ASR, embeddings, reranking, and semantic
> search are **not** implemented; Goal 09 stays limited to transcript
> processing foundations.

## Structure

```
backend/
├── app/
│   ├── __init__.py
│   ├── main.py            # FastAPI app factory, metadata, lifespan
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py      # pydantic-settings configuration
│   │   └── logging.py     # centralized logging setup
│   └── api/
│       ├── __init__.py
│       └── health.py      # GET /health
├── tests/
│   └── test_health.py     # pytest suite
├── .venv/                 # standalone Python 3.12 venv (git-ignored, NOT committed)
├── requirements.txt       # exact direct-dependency pins (6 packages)
├── requirements-lock.txt  # fully resolved environment snapshot (29 packages)
├── pytest.ini
├── .env.example           # copy to .env (not committed)
└── README.md
```

Run all commands below **from the `backend/` directory**.

---

## 1. Standalone Python 3.12 prerequisite

**Python 3.12 is the project baseline, installed as a standalone python.org /
winget runtime — NOT from a Conda environment.** Python 3.14 passed the early
backend tests but is *not* assumed suitable for the future ML stack.

If Python 3.12 is not present, install the official standalone runtime:

```powershell
winget install --id Python.Python.3.12 -e
```

This does not modify or remove any existing Conda environments.

Verify the interpreter is registered and resolves to a standalone (non-Conda)
installation:

```powershell
py -0p
py -3.12 --version
py -3.12 -c "import sys; print(sys.base_prefix)"
```

`sys.base_prefix` must point at the standalone install, e.g.
`C:\Users\USER\AppData\Local\Programs\Python\Python312` — **not** at
`anaconda3\envs\...`. Do not substitute Python 3.13 or 3.14.

## 2. Virtual environment (Windows PowerShell)

Create the environment from the standalone Python 3.12 (one time):

```powershell
py -3.12 -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks the activation script, allow it for the current session
only (no permanent policy change):

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Confirm activation and that the venv is independent of Conda:

```powershell
python --version
python -c "import sys; print(sys.executable); print(sys.base_prefix); print(sys.prefix)"
```

- `python --version` must report **3.12.x** (validated: `3.12.10`).
- `sys.base_prefix` must point at the standalone Python 3.12 install.
- `sys.prefix` must point at `backend\.venv`.
- Neither may reference `conda`, `anaconda`, or `ml_clean`.

To deactivate:

```powershell
deactivate
```

> On macOS / Linux use `source .venv/bin/activate` instead.
> `.venv/` is git-ignored — **never commit the virtual environment**.

## 3. Dependency installation

With `.venv` activated, choose one of the two workflows below.

### Standard installation (direct dependencies)

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

`requirements.txt` pins the six direct dependencies exactly. Transitive
versions are resolved by pip:

| Package | Pin |
| --- | --- |
| `fastapi` | 0.162.0 |
| `uvicorn[standard]` | 0.54.0 |
| `pydantic-settings` | 2.16.2 |
| `python-dotenv` | 1.2.4 |
| `pytest` | 9.2.0 |
| `httpx` | 0.28.1 |

### Reproducible installation (fully resolved lock)

`requirements-lock.txt` is a `pip freeze` snapshot of the exact, validated
environment (29 packages). Use it to reproduce the environment without
re-resolving transitive dependencies:

```powershell
python -m pip install --upgrade pip
pip install -r requirements-lock.txt
```

> `requirements.txt` remains the **direct dependency declaration**; do not
> replace it with the lock file. Regenerate the lock from a clean venv with
> `python -m pip freeze > requirements-lock.txt` whenever dependencies change.

No ML packages, PyTorch, Transformers, Whisper, database drivers, or other
future dependencies are included in this goal.

Verify integrity after either workflow:

```powershell
python -m pip check
python -m pip list
```

`pip check` must report `No broken requirements found.`

## 4. Environment configuration

Settings are loaded by `pydantic-settings` from environment variables and,
optionally, a local `.env` file. All values have safe defaults, so the app runs
with **no configuration at all**.

```powershell
copy .env.example .env
```

| Variable | Default | Purpose |
| --- | --- | --- |
| `APP_NAME` | `voxsearch-backend` | Service name (`/health` "service" field, OpenAPI title) |
| `APP_ENV` | `development` | Runtime environment label |
| `APP_VERSION` | `0.1.0` | Version exposed by `/health` and OpenAPI |
| `API_PREFIX` | `/api/v1` | Prefix reserved for future versioned routers |
| `LOG_LEVEL` | `INFO` | `DEBUG` / `INFO` / `WARNING` / `ERROR` / `CRITICAL` |

`.env` is git-ignored. **Never put credentials or secrets in `.env.example`.**

## 5. Running tests

With `.venv` activated:

```powershell
pytest -v
```

The suite uses FastAPI's `TestClient` and covers:

- `GET /health` / `GET /health/db` responses and application metadata.
- SQLAlchemy models, Alembic migrations, and database health (Goals 05–07).
- Repository and service layers and transaction ownership (Goal 08).
- Transcript input contract, validation, normalization, and atomic ingestion
  with real-session rollback (Goal 09).

From the repository root the full suite can be run with the standalone venv
interpreter (no live Supabase connection is required):

```powershell
.\backend\.venv\Scripts\python.exe -m pytest backend\tests -v
.\backend\.venv\Scripts\python.exe -m pip check
```

`pip check` must report `No broken requirements found.`

## 6. Starting the FastAPI server

With `.venv` activated:

```powershell
uvicorn app.main:app --reload
```

By default this serves `http://127.0.0.1:8000`. To bind explicitly:

```powershell
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Interactive API docs are available at `http://127.0.0.1:8000/docs`.

Verify the health endpoint (from a second terminal):

```powershell
curl http://127.0.0.1:8000/health
```

Expected `200 OK` response:

```json
{
  "status": "healthy",
  "service": "voxsearch-backend",
  "version": "0.1.0"
}
```

---

## 7. Goal 09 — Transcript processing foundation

A library-level transcript ingestion pipeline. It is a **foundation only**: no
ASR, no caption downloading/scraping, no embeddings, no reranking, no search,
and no HTTP routes were added (OpenAPI paths remain `/health` and `/health/db`).

### Modules

| File | Responsibility |
| --- | --- |
| `app/transcripts/contracts.py` | Pydantic 2 validated input contract + closed `TranscriptSource` set |
| `app/transcripts/validation.py` | Cross-segment / contextual validation, overlap policy, duration bound |
| `app/transcripts/normalization.py` | Deterministic, multilingual-safe text normalization |
| `app/services/transcript_ingestion_service.py` | Atomic ingestion orchestration (ONE transaction) |

Validation, normalization, and persistence are deliberately separated. The
service composes the **existing** repository layer inside a single
`transaction()`; it does not call `TranscriptService.create_transcript_asset()` /
`save_segments()` because each of those owns a separate commit and would break
all-or-nothing atomicity (documented technical reason in the module docstring).

### Input contract

`TranscriptInput`: `video_id`, `content_right_id`, `source_type`, `language`,
`segments` (>= 1). Each `TranscriptSegmentInput`: `start_time`, `end_time`
(seconds, float), `text`, optional `speaker`, optional `metadata`.
`source_type` is a closed set — `creator_provided`, `licensed_transcript`,
`user_uploaded`, `asr_generated` — mapping to the four authorized source
priorities in the architecture (§7.1). There is **no** arbitrary YouTube
caption source. Unknown fields are rejected (`extra="forbid"`), not discarded.

`start_time`/`end_time` are strict: only real numeric values are accepted (no
string coercion, no bools, no NaN/±Inf). Text is rejected if blank after
trimming, but the **original text is preserved verbatim** (never trimmed in the
contract). Optional `speaker` / `metadata` are accepted and surfaced in the
ingestion result but **not persisted** — the `transcript_segments` table has no
columns for them and adding a migration is out of scope for this foundation.

### Validation rules & overlap policy

Non-empty transcript; non-blank segment text; numeric finite timestamps;
`start_time >= 0`; `end_time > start_time`; strictly increasing `start_time`
across segments; every `end_time <= Video.duration_seconds` **when the duration
is known** (skipped when `NULL`). Invalid data raises a clear `ValidationError`
(Pydantic) or `TranscriptValidationError` (contextual, index-aware). Nothing is
silently repaired, reordered, clamped, or dropped.

**Overlap is allowed and preserved.** The architecture's word-level / forced
alignment (§7) and ASR/caption exports commonly produce overlapping windows;
rejecting them would discard legitimate alignment data. Overlapping segments
(with distinct starts) are persisted verbatim. Strictly increasing (not merely
non-decreasing) starts are required because the existing
`uq_transcript_segments_asset_start` unique constraint forbids two segments of
one asset sharing a start offset.

### Normalization

Unicode **NFC** composition (not NFKC, which could change meaning), then any run
of Unicode whitespace (spaces, tabs, line breaks, NBSP) collapses to a single
space, then trim. Punctuation, casing, and non-whitespace content are preserved;
there is no lowercasing, ASCII folding, stemming, or language-specific logic, so
Bangla, English, and mixed text are all safe. The function is pure,
deterministic, and idempotent, and returns **both** `original` and `normalized`
text. The ingestion service persists the normalized form into the existing
`segment_text` column and keeps the original in the `IngestionResult`; storing a
byte-exact original at rest would need a future schema column/migration.

### Authorization (fail closed)

Before any write: the video must exist; the content-right must exist, belong to
that video, and pass the existing `ContentRightsService.is_active` (rejects
revoked, expired, not-yet-granted, inactive, or otherwise unauthorized records).
Nothing is auto-authorized. The Goal 07 composite FK is the DB-level backstop.

### Status lifecycle & atomicity

The existing `status` column is reused — **no migration and no new status
columns**. `processing` is set transiently when the asset is created;
`completed` is set before the single commit and is the only status ever
persisted on success; `failed` is an **outcome label, not a persisted row** —
any failure (before or inside the transaction) leaves the database completely
unchanged (zero assets, zero segments), which is the strongest "no partially
persisted transcript data" guarantee. One `transaction()` wraps asset creation
+ all segment inserts + the status update, so a mid-batch DB error rolls back
the asset too.

Repeated ingestion is allowed and deterministic: with no checksum / dedup key
in the schema, each call creates a new independent asset (no silent dedup).
Checksum-based dedup is a future concern requiring schema support.

### Limitations

No live PostgreSQL concurrency was tested (no isolated instance; production
Supabase was not used). Real-session rollback behavior is verified against an
isolated in-memory SQLite-compatible stand-in schema, plus a control test that
reproduces the `PendingRollbackError` state machine. Original text and optional
`speaker`/`metadata` are not persisted at rest.

---

## Validation performed

### Goal 04 — standalone Python runtime (decoupled from Conda)

- Installed official standalone Python 3.12.10 via
  `winget install --id Python.Python.3.12 -e`.
- `py -3.12` → `C:\Users\USER\AppData\Local\Programs\Python\Python312\python.exe`
  (standalone, no Conda reference).
- Recreated `backend/.venv` from that interpreter:
  - `pyvenv.cfg` home → `...\Python312`, version `3.12.10`
  - `sys.base_prefix` → `C:\Users\USER\AppData\Local\Programs\Python\Python312`
  - `sys.prefix` → `D:\Python Project\Advance-Search-Engine\backend\.venv`
  - Neither references Conda.
- `python -m pip install --upgrade pip` → pip 26.3;
  `pip install -r requirements.txt` → 29 packages resolved.
- `python -m pip check` → `No broken requirements found.`
- `pytest -v` → **4 passed** (Python 3.12.10).
- Live `uvicorn` + `curl /health` → `200 OK`,
  `{"status":"healthy","service":"voxsearch-backend","version":"0.1.0"}`.
- `requirements-lock.txt` generated from the fresh standalone 3.12 venv:
  29 plain `name==version` pins; no local paths, editable installs, or Conda
  references; validated plain UTF-8/ASCII (no BOM).

### Goal 03 — reproducible Python environment (superseded by Goal 04)

- Initially created `backend/.venv` from CPython 3.12.14 (a `conda-forge`
  build in an existing conda env) because `py -3.12` was not registered.
- Pinned the six direct dependencies and validated `pip check` + 4 passing
  tests.
- **Goal 04 replaced this Conda-based venv with a standalone Python 3.12.10
  venv**, which is now the project standard. The old environment is not
  committed.

### Goal 02 — backend foundation

- `pip install -r requirements.txt`
- `pytest -v`
- `uvicorn app.main:app --host 127.0.0.1 --port 8000` + `curl /health`

See the Goal 02–04 final reports in the session history for recorded results.
