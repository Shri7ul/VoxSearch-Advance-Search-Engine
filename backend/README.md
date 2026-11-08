# VoxSearch — Backend

FastAPI backend for VoxSearch.

> **Scope:** FastAPI application initialization, centralized settings/logging
> and `GET /health` (Goal 02), a reproducible **standalone** Python 3.12
> environment decoupled from Conda (Goals 03–04), a read-only Supabase /
> PostgreSQL connection, the Alembic schema & migration foundation, and the
> repository & service layers (Goals 05–08), the **validated transcript
> ingestion foundation** and its original-text persistence correction
> (Goal 09), **deterministic multilingual sentence segmentation** (Goal 10),
> and **timestamp preservation through word-level forced alignment**
> (Goal 12), and **deterministic, sentence-aware chunking** (Goal 13).
> ASR, embeddings, reranking, and semantic search are **not** implemented.
> Goal 14 (**ACCEPTED**) has prepared and validated an **isolated, CPU-only
> BGE-M3 environment** (installation / smoke-test validation ONLY) in
> `backend/.venv-embedding`; it adds no application code, no database objects,
> and no embeddings (see §11).
> Goals 09–13 stay limited to transcript processing foundations (no search,
> no embeddings). Goal 12 is library-level: it adds no HTTP routes,
> downloads no audio, installs no heavy dependency into the core venv, and
> has been accepted but NOT applied to production (see §9 "Status").

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
├── requirements.txt       # exact direct-dependency pins (9 packages)
├── requirements-lock.txt  # fully resolved environment snapshot (37 packages)
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

`requirements.txt` pins the nine direct dependencies exactly. Transitive
versions are resolved by pip:

| Package | Pin |
| --- | --- |
| `fastapi` | 0.162.0 |
| `uvicorn[standard]` | 0.54.0 |
| `pydantic-settings` | 2.16.2 |
| `python-dotenv` | 1.2.4 |
| `sqlalchemy` | 2.0.57 |
| `psycopg[binary]` | 3.2.13 |
| `alembic` | 1.16.5 |
| `pytest` | 9.2.0 |
| `httpx` | 0.28.1 |

### Reproducible installation (fully resolved lock)

`requirements-lock.txt` is a `pip freeze` snapshot of the exact, validated
environment (37 packages). Use it to reproduce the environment without
re-resolving transitive dependencies:

```powershell
python -m pip install --upgrade pip
pip install -r requirements-lock.txt
```

> `requirements.txt` remains the **direct dependency declaration**; do not
> replace it with the lock file. Regenerate the lock from a clean venv with
> `python -m pip freeze > requirements-lock.txt` whenever dependencies change.

No ML packages, PyTorch, Transformers, or Whisper are part of the core
dependencies; the heavy alignment stack is declared separately in the optional
`requirements-alignment.txt`. Database components ARE part of the core: the
direct dependencies include the `psycopg[binary]` driver, SQLAlchemy, and
Alembic.

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
- SQLAlchemy models, Alembic migrations, and database health (Goals 05–07,
  plus the Goal 09 corrective migration).
- Repository and service layers and transaction ownership (Goal 08).
- Transcript input contract, validation, normalization, and atomic ingestion
  with real-session rollback, and separate original/normalized persistence
  (Goal 09).
- Deterministic multilingual sentence segmentation: English / Bangla / mixed,
  character-offset integrity, and segment-level timestamp provenance (Goal 10).
- Goal 12 timestamp preservation: seconds/milliseconds conversion, audio/video
  timeline offset, strict word/char corroboration, fail-closed timing, language
  gate, two-phase rights/transaction/retry behavior, the additive migration,
  and an opt-in (default-skipped) live WhisperX 3.8.6 integration tier. The
  default suite uses a FAKE provider; no heavy ML dependencies or model
  weights are required.

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
| `app/transcripts/sentence_segmentation.py` | Deterministic multilingual sentence segmentation (Goal 10) |
| `app/services/transcript_ingestion_service.py` | Atomic ingestion orchestration (ONE transaction) |
| `migrations/versions/b5d2e8a14c7f_*` | Additive, DDL-only `original_text` column (Goal 09 corrective) |

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
contract) and is persisted in `original_text`. Optional `speaker` / `metadata`
are accepted and surfaced in the ingestion result but **not persisted** — the
`transcript_segments` table has no columns for them and adding those columns
remains out of scope for this foundation.

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
text. The ingestion service persists **both separately**: the verbatim original
in the nullable `original_text` column and the normalized form in the existing
`"text"` column (mapped to `segment_text`). `IngestionResult` still carries both
values so callers need no re-query.

### Corrective migration (`b5d2e8a14c7f`)

The existing normalized `"text"` column keeps its meaning: still `NOT NULL`,
still guarded by `ck_transcript_segments_text_nonempty`, and still the target of
`search_segments_by_text`. The single additive Alembic revision adds **only**
`original_text TEXT NULL` — DDL-only, with no `DEFAULT`, no `UPDATE`, no
backfill, and no data rewrite. Rows created before it have
`original_text = NULL`, an honest "not captured" marker: their verbatim
original was never stored and cannot be reconstructed, so it is NOT fabricated
by copying the normalized text. New rows always populate both values. Adding a
nullable column with no default is a metadata-only operation on PostgreSQL 11+.
This migration was reviewed as **offline SQL only** and has **not** been applied
to production; doing so requires separate approval.

### Authorization (fail closed)

Before any write: the video must exist; the content-right must exist, belong to
that video, and pass the existing `ContentRightsService.is_active` (rejects
revoked, expired, not-yet-granted, inactive, or otherwise unauthorized records).
Nothing is auto-authorized. The Goal 07 composite FK is the DB-level backstop.

### Status lifecycle & atomicity (Option A)

The existing `status` column is reused — **no new status columns, no separate
jobs table, and no independent commits**. (The only migration in this
corrective review is the additive `original_text` column, unrelated to status.)

* `processing` is a **transient internal execution state**. It is set when the
  asset row is created but is never committed: the single transaction commits
  only after the status has been advanced to `completed`.
* `completed` is the **only status ever persisted**, and only on success.
* `failed` is an **outcome label, not a persisted row**. A failed ingestion
  surfaces as a typed, credential-free exception plus a guaranteed, queryable
  zero-row state (no asset, no segments).

A durable `failed` row is fundamentally incompatible with an all-or-nothing
transaction: a rolled-back row cannot carry a status, and persisting one would
require an independent commit that weakens atomicity. Durable job / failure
tracking therefore belongs to the background-processing job layer (roadmap
Goals 39–42), not to this foundation.

One `transaction()` wraps asset creation + all segment inserts + the status
update, so a mid-batch DB error rolls back the asset too and **no partial
segments survive**. Fail-closed authorization runs **before** any write, so an
authorization failure can never create an unauthorized asset. This design
introduces **no independent commits and does not weaken atomicity**.

Repeated ingestion is allowed and deterministic: with no checksum / dedup key
in the schema, each call creates a new independent asset (no silent dedup).
Checksum-based dedup is a future concern requiring schema support.

### Limitations

No live PostgreSQL concurrency was tested (no isolated instance; production
Supabase was not used). Real-session rollback behavior is verified against an
isolated in-memory SQLite-compatible stand-in schema, plus a control test that
reproduces the `PendingRollbackError` state machine. Original text **is** now
persisted verbatim in `original_text`; rows created before the corrective
migration have `NULL` there (original never captured). Optional
`speaker`/`metadata` input fields are still not persisted at rest (no schema
columns exist). The corrective migration is DDL-only, was reviewed as offline
SQL, and has **not** been applied to production.

---

## 8. Goal 10 — Multilingual sentence segmentation

A deterministic, **library-level** segmentation layer in
`app/transcripts/sentence_segmentation.py`. It splits ONE source transcript
segment's text into ordered sentence units and is intended as the INPUT to the
future sentence-aware chunking / semantic-indexing pipeline (architecture §8,
§23 step 2). It does **not** chunk, embed, align, rerank, search, download, or
run workers. It is a pure function: no DB, no I/O, no credentials, no ML model,
and no external NLP service.

### Input / output contract

`SentenceSegmentationInput` (Pydantic 2, `extra="forbid"`):
`source_text` (verbatim), `source_segment_ref` (stable parent reference),
`start_seconds` / `end_seconds` (float seconds, matching the persisted
columns), `language`. It reuses Goal 09's `_finite_float` (strict numeric
validation, **not duplicated**) and rejects blank text / labels.

`segment_sentences()` returns an ordered tuple of `Sentence` objects carrying:
`text` (verbatim), `ordinal`, `char_start` / `char_end`, `source_segment_ref`,
`source_segment_start_seconds` / `source_segment_end_seconds`, `language`,
`timestamp_precision`, `exact_start_seconds` / `exact_end_seconds` (reserved),
and `normalized_text` (a **separate** derived value produced by the existing
`normalize_text`; it never replaces `text` and never affects offsets). The pure
engine `find_sentence_spans()` is independently testable without the contract.

### Segmentation algorithm & punctuation rules

Sentence-ending punctuation: English `. ? !` and Bangla danda `।` (plus double
danda `॥`). Mixed punctuation in one transcript is handled uniformly.

A boundary is recognised only when punctuation is in a token's **trailing**
run. Periods *inside* a token (`3.14`, `example.com`, `https://x.y`,
`Dr.Rahman`, `v1.2.3`) are never candidates — the deterministic rule that
avoids "split every period blindly". Known abbreviations (`Dr.`, `Prof.`,
`etc.`, `Fig.`, `p.m.`), dotted acronyms (`U.S.`, `e.g.`, `Ph.D.`, `B.Sc.`),
and single-letter initials in any script (`J.`, `ড.`) are not boundaries.
Closing quotes / brackets after a boundary stay with the same sentence
(`He said "Hi." Then left.`). Repeated punctuation (`!!!`, `?!`, `...`) is one
boundary; a standalone punctuation-only token creates no empty sentence.
Text without terminal punctuation becomes one sentence. Blank input is
**rejected** with `EmptySourceTextError` (explicit policy, no silent empty
result). Sentences are never reordered; ordinals are stable and deterministic.

### Timestamp precision strategy

Goal 09 stores **segment-level** timestamps only, so no sentence timestamp is
fabricated from character positions or sentence length. Every sentence keeps
its parent reference and the parent's start/end timestamps **verbatim**, with
`timestamp_precision = TimestampPrecision.SEGMENT_LEVEL`. There are NO
`sentence_start_seconds` / `sentence_end_seconds` fields: the inherited window
is never presented as an exact sentence boundary. Text offsets (`char_*`) and
temporal offsets are separate concepts. `exact_start_seconds` /
`exact_end_seconds` are reserved and ALWAYS `None` here; only a future
word-level alignment goal (architecture §7.5) can derive exact boundaries and
upgrade the precision to the reserved `WORD_ALIGNED` value. Nothing is
interpolated, scaled, or proportionally divided by character count.

### Character-offset integrity

Segmentation operates on the ORIGINAL source string; offsets are Python code
point indices into that untransformed string. `Sentence.text` is exactly
`source_text[char_start:char_end]`. Spans **tile** the source: the first starts
at 0, the last ends at `len(source_text)`, and each span starts where the
previous ended, so `''.join(s.text for s in sentences) == source_text` — no
character is dropped, no sentence silently disappears. Punctuation, Bangla
Unicode, English technical terms, casing, and whitespace are preserved; no
lowercasing, punctuation removal, or rewriting. Whitespace between sentences
attaches to the following sentence; trailing whitespace merges into the last
sentence (no whitespace-only phantom sentence).

### No persistence, no migration

The architecture (§9.1) lists `transcript_segments` as the timestamped
sentence/segment table and `transcript_words` as OPTIONAL word-level alignment;
it defines **no dedicated sentence table**. Sentences are therefore not
persisted, no new table is created, the Goal 09 ingestion transaction is
untouched, and **no Alembic migration** was added (the approved three-revision
history is unchanged). This goal added **no dependencies**.

### Known limitations

Perfect segmentation is not guaranteed with deterministic rules. An
abbreviation that genuinely ends a sentence under-splits; missing whitespace
after a period (`Hello.World`) under-splits; an ellipsis used as a pause
(`Wait... what?`) over-splits; quoted/direct-speech boundaries and
language-specific orthography are heuristic. There is no language detection,
no ML model, and no external NLP service. Only authorized transcript content
may be passed to this layer. Exact sentence timestamps require future
word-level alignment and are explicitly out of scope.

---

## 9. Goal 12 — Timestamp preservation (word-level alignment)

Goal 12 upgrades transcript timestamps from **segment-level** to **exact,
word-derived** timing using WhisperX 3.8.6 wav2vec2 forced alignment. It is a
library-level, **fail-closed** foundation: no fabricated, interpolated,
proportional, or character-offset-derived timestamps are ever persisted. It
adds **no HTTP routes, no ASR, no audio downloading, no chunking**, and keeps
the core API runtime free of `torch` / `whisperx`.

> **Status:** **ACCEPTED** and committed on `main` in this change set, on the
> accepted Goal 10 baseline. It has **NOT** been applied to production
> Supabase, and the live real-weights WhisperX integration is an **outstanding,
> separately-approved verification** (see "Known limitations"). The default
> test suite uses a FAKE provider. Migration verification is offline DDL
> rendering only; live PostgreSQL runtime enforcement remains unverified.

### Architecture and data flow

Four layers, with a strict pure/boundary split:

| File | Responsibility |
| --- | --- |
| `app/transcripts/word_contract.py` | PURE primitives: unit conversion, language allowlist, deterministic tokenizer, `AlignedWord`, status/reason vocabulary |
| `app/transcripts/word_alignment.py` | PURE preparation, request building, strict fail-closed validation, deterministic char reconstruction |
| `app/transcripts/sentence_timestamps.py` | PURE `AlignedSentence` derivation (Goal 13 contract) |
| `app/services/word_alignment_service.py` | Two-phase orchestration + atomic persistence (owns the transactions) |
| `app/repositories/{source_audio_asset,transcript_alignment_run,transcript_word}_repository.py` | `flush()`-only data access for the three new tables |
| `alignment_worker/worker.py` | ISOLATED process entry point; lazy `whisperx` imports; pinned adapter flags |
| `alignment_worker/client.py` | `SubprocessAlignmentProvider`: explicit-argv subprocess launcher, controlled env, bounded JSON |
| `migrations/versions/a7c2e9f4b1d8_*` | One additive, DDL-only Alembic migration (three new tables + ownership constraints) |
| `requirements-alignment.txt` | OPTIONAL, isolated provider environment (heavy deps) |

Data flow (two-phase lifecycle, retry + concurrency safe):

```
Phase A (DB txn)  video/asset/audio + SAME active right + language gate
        -> create + COMMIT a `running` alignment run (durable/auditable)
Pure preparation  original_text -> tokens, compute audio/video windows,
        skip out-of-window / NULL-original segments (fail closed, no guessing)
Provider (OUTSIDE any DB txn)  bounded AlignmentRequest -> isolated worker
        (whisperx align, interpolate_method="ignore",
         return_char_alignments=True) -> raw word/char JSON
Pure validation    strict 1:1 ordinal mapping, finite timings, char
        corroboration, window + monotonicity checks -> AlignedWord[]
Phase B (DB txn)  recheck authorization fresh -> on success, in ONE txn:
        supersede old current success + delete its words + insert new words
        + set final status; on any failure, finalize THIS run as failed and
        leave the previous success untouched
```

The worker receives **audio-timeline** seconds and the **verbatim** original
text; all persisted word timestamps are converted to **video-timeline** integer
milliseconds. A `ProviderCallable` protocol (`Callable[[dict], Mapping]`) keeps
the core runtime decoupled: the default tests inject a fake, and the real
provider runs in a separate interpreter.

### Audio asset and word contracts

**`SourceAudioAssetInput`** (Pydantic 2, `extra="forbid"`) registers an
AUTHORIZED audio clip. Fields: `video_id`, `content_right_id`, an **opaque,
non-secret** `storage_uri` (NEVER a URL / signed URL — those could leak into
logs or the worker boundary), `checksum_sha256` (64 lowercase hex, verified by
a DB CHECK), `mime_type`, `rights_basis`, optional `sample_rate_hz` (NULL =
unknown, never guessed), `duration_ms` (the AVAILABLE audio window
`[0, duration_ms]`, required positive on the authorized provisioning path),
and `timeline_offset_ms` (the audio→video offset).

**Persistence tables** (all in the `voxsearch` schema, UUID PKs, composite
ownership FKs, additive):

- `source_audio_assets` — authorized clip: storage/provenance columns above,
  `timeline_offset_ms`, nullable `duration_ms` / `sample_rate_hz`.
- `transcript_alignment_runs` — one attempt: `attempt_number`, `provider`
  (`whisperx`), `provider_version` (`3.8.6`), `align_model_name`, `language`,
  `status`, `segment_count` / `aligned_segment_count` /
  `skipped_segment_count` / `word_count`, `audio_duration_ms`,
  `failure_reason`, `started_at` / `finished_at`.
- `transcript_words` — one validated word: `word_index` (0-based ordinal),
  `char_start` / `char_end` (verbatim original-text offsets), `word` (verbatim
  token), `start_ms` / `end_ms` (integer ms, VIDEO timeline), with
  `CHECK (start_ms >= 0)`, `CHECK (end_ms > start_ms)`, and
  `UNIQUE (segment_id, word_index)`.

An **`AlignedWord`** is only produced after strict validation: it carries
`segment_id`, `ordinal`, `text`, `char_start`, `char_end`, and VIDEO-timeline
`start_ms` / `end_ms`. The provider's word list must map **1:1, in order, with
exact token text** onto the original-text token stream; any count or text
mismatch skips the **ENTIRE** segment (zero words), never a partial or
re-indexed subset. Because WhisperX 3.8.6 final char dicts omit `word-idx`,
`reconstruct_word_chars()` deterministically rebuilds per-word char evidence
from the `chars` array (space separators ignored); missing/NaN char timings
fail closed with no backfill.

### Seconds / milliseconds conversion

Existing `transcript_segments` timestamps stay **float SECONDS** (contracts
unchanged). New Goal 12 word timestamps are **INTEGER MILLISECONDS**
(`BIGINT`). The two units meet only at explicit conversion boundaries in
`word_contract.py`:

- `seconds_to_ms(seconds)` uses `Decimal(str(seconds)) * 1000` and
  `ROUND_HALF_UP` (NOT `round()`'s banker's rounding), rejecting bool / str /
  `None` / NaN / ±Inf / negatives.
- `ms_to_seconds(ms)` accepts only a real non-negative `int`.

Integer milliseconds are a **representation choice, NOT a claim of 1 ms
measurement accuracy** — converting a seconds float quantizes it.

### Audio / video timeline offset

An authorized audio asset may be a clip offset from the video timeline:

```
video_timestamp_ms = audio_timestamp_ms + timeline_offset_ms
audio_segment_start_ms = video_segment_start_ms - timeline_offset_ms
audio_segment_end_ms   = video_segment_end_ms   - timeline_offset_ms
```

WhisperX receives AUDIO-timeline timestamps; persisted words are shifted back
to the VIDEO timeline. `prepare_segments()` skips (with a closed reason) any
segment whose audio window `[video_ms - offset, ...]` falls outside the
available audio window `[0, duration_ms]`; it **never shifts, clamps, or
guesses** a timestamp, and fails closed if audio/video synchronization cannot
be established (NULL / empty duration).

### Timestamp precision semantics

Goal 10's `Sentence` contract is **UNCHANGED**: `exact_start_seconds` /
`exact_end_seconds` remain `None` and are NEVER populated. Goal 12 introduces a
separate `AlignedSentence` (in `sentence_timestamps.py`) with integer-ms
`exact_start_ms` / `exact_end_ms` and a `precision`:

- `SEGMENT_LEVEL` — inherited parent window only (Bengali / unaligned / any
  partially covered sentence stays here, with both exact values `None`).
- `WORD_ALIGNED` — set ONLY when **every** original token in the sentence span
  has a valid aligned word; then `exact_start_ms = min(word.start_ms)`,
  `exact_end_ms = max(word.end_ms)`.

A word belongs to a sentence when its original-text `char_start` falls in that
sentence's half-open `[char_start, char_end)` span. No interpolation,
proportional division, or segment-window fallback is ever presented as an
exact boundary.

### Supported and unsupported languages

WhisperX 3.8.6 wav2vec2 forced alignment is the single primary provider. The
supported allowlist is transcribed from its exact `DEFAULT_ALIGN_MODELS_TORCH`
(`en, fr, de, es, it`) + `DEFAULT_ALIGN_MODELS_HF` (`ja, zh, nl, uk, pt, ar,
cs, ru, pl, hu, fi, fa, el, tr, da, he, vi, ko, ur, te, hi, ca, ml, no, nn,
sk, sl, hr, ro, eu, gl, ka, lv, tl, sv, id`) maps. **Bengali** (`bn`, `ben`,
`bn-BD`, `bn-IN`) and any unknown / unsupported / region-tagged code are
rejected by the language gate with **ZERO provider calls, ZERO runs, and ZERO
word rows**; Bengali remains `SEGMENT_LEVEL` and is deferred (with its
evaluation dataset) to a future goal.

### Rights checks and retry behavior

Authorization is **fail closed and re-checked twice**:

- **Phase A gate (read-only, no writes):** the video must exist; the transcript
  asset and audio asset must exist, belong to the SAME video, and be
  authorized by the SAME active `content_right_id` (the exact lineage is also
  enforced at the DB level by composite FKs); the right must be active /
  unexpired / unrevoked (`ContentRightsService.is_active`). An initial rights
  or language rejection creates **NO alignment run**.
- **Phase B recheck:** because the session uses `expire_on_commit=False`, a
  **fresh** read (`populate_existing=True`) re-reads the right after the
  out-of-transaction provider execution. If it was revoked meanwhile, the run
  is finalized as `failed` / `rights_revoked` with ZERO new word rows and the
  previous success preserved.

Statuses (closed set): `pending`, `running`, `completed`, `partial`, `failed`,
`unsupported_language`, `superseded` (`rejected_rights` is a reserved DB label
never emitted for an initial gate rejection). `completed` = no skipped
segments, `partial` = some skipped. A **partial UNIQUE index** permits at most
one *current successful* (`completed` / `partial`) run per
`(transcript_asset_id, source_audio_asset_id)`. On a **zero-word** result, the
attempt fails and the previous success is left untouched. Only a **successful
replacement** (in ONE transaction) may supersede the old run, delete its
words, insert the new word set, and set the new final status. A concurrent
winner (UNIQUE violation) is detected by re-querying and the losing run is
finalized as `failed` / `superseded_by_concurrent_run`; genuine schema errors
are re-raised, never hidden.

### Optional alignment environment installation

Heavy dependencies are **NEVER** added to `requirements.txt` /
`requirements-lock.txt` or installed into `backend/.venv`. The core API,
service, client, and worker modules import NO `torch` / `whisperx` at module
load (verified by a clean-subprocess test). `requirements-alignment.txt` pins
`whisperx==3.8.6` plus that release's declared direct requirements
(`torch~=2.8.0`, `torchaudio`, `ctranslate2`, `transformers`, etc.) for a
**separate, isolated interpreter**:

```powershell
python -m venv .venv-alignment
.venv-alignment\Scripts\python.exe -m pip install -U pip
.venv-alignment\Scripts\python.exe -m pip install -r requirements-alignment.txt
```

`SubprocessAlignmentProvider(worker_python=..., audio_path=...)` then launches
`python -m alignment_worker.worker` with explicit argv, `shell=False`, a
controlled env allowlist (no DB credentials / secrets forwarded), bounded JSON
stdin/stdout, and a hard timeout. The worker re-validates the pinned flags
(`interpolate_method="ignore"`, `return_char_alignments=True`) and rejects a
URL / signed-URL `audio_path`. This setup is **illustrative** — no heavy
dependency, model weight, or real audio is installed/downloaded by this goal's
test suite.

### Migration instructions

One additive Alembic revision `a7c2e9f4b1d8` with
`down_revision = "b5d2e8a14c7f"` adds the three tables, two additive UNIQUE
constraints on existing tables (the referenced tuples required for the
composite FKs), CHECK constraints, indexes, and the partial UNIQUE index. It
is **DDL-only**: no INSERT / UPDATE / DELETE, no backfill, no column
alteration, no rename; existing seconds columns, `original_text`, and all
pre-existing rows are untouched. All objects live in the dedicated `voxsearch`
schema. `downgrade()` reverses only what `upgrade()` created (child-first).

There are two distinct review paths; they differ in whether `.env` is loaded.

**Dedicated offline migration tests (do NOT load `.env`).**
`tests/test_word_alignment_migration.py` configures a direct
`MigrationContext(as_sql=True)` + `Operations` and calls the real
`upgrade()` / `downgrade()` op functions, rendering PostgreSQL DDL to a
buffer. It deliberately does NOT execute `migrations/env.py`, so it opens no
DB connection, needs no credentials, and never calls `load_dotenv()` — `.env`
is not read:

```powershell
.venv\Scripts\python.exe -m pytest tests/test_word_alignment_migration.py -q
```

**Standard Alembic CLI (executes `env.py`, which loads `.env`).**
`alembic upgrade head --sql` (offline) and `alembic upgrade head` (online)
run through `migrations/env.py`, which calls `load_dotenv()` at import time.
These commands therefore DO read `.env`. Use them only in an explicitly
approved environment, and only after verifying that the intended
`DATABASE_URL` is already set and correct for that environment. Never print,
log, or inspect `.env` contents.

The committed verification is **offline DDL rendering**: the dedicated
migration tests call the real `upgrade()` / `downgrade()` operations through a
`MigrationContext(as_sql=True)` and assert the upgrade / downgrade /
re-upgrade sequence plus every ownership composite FK, additive UNIQUE, CHECK,
index and partial-index definition. They open **no database connection**.

**Live PostgreSQL runtime enforcement is NOT verified.** No isolated disposable
PostgreSQL run is committed as evidence in this repository, and none is
available in the current test environment (`psql` / `pg_ctl` / `docker` absent;
no test-database URL configured). SQLite stand-ins used in service tests are a
behavioural simulation and MUST NOT be substituted for PostgreSQL-specific
constraint verification. Runtime enforcement of the composite FKs, additive
uniques, CHECKs and the partial UNIQUE index therefore remains an outstanding
verification. **`b5d2e8a14c7f` and `a7c2e9f4b1d8` MUST NOT be applied to
production Supabase** without separate, explicit approval; the production
database was NOT touched by this goal.

### Known limitations

- **Live WhisperX 3.8.6 / real-model-weights execution is OUTSTANDING.** The
  `test_alignment_provider_integration.py` default tier launches the REAL
  isolated worker against a controlled pure-Python `whisperx` double (no heavy
  deps / weights / real audio). The opt-in `@pytest.mark.live_whisperx` tier
  (real release + real authorized audio + cached weights) is **skipped by
  default** and is NOT claimed to have run. The exact runtime shape of 3.8.6
  char dicts can only be confirmed live; the pure reconstruction fails closed
  if the assumption does not hold.
- Integer milliseconds are a representation choice, not 1 ms accuracy.
- Bengali alignment is unsupported in this goal.
- A process crash between the Phase A commit and provider completion leaves a
  stale `running` run; reaping is a future goal. Such rows carry no word rows
  and never block a later successful attempt (excluded from the partial index).
- `requirements-alignment.txt` is a dependency DECLARATION, not a resolved
  transitive lock; producing a real lock requires installing the heavy tree.
- Optional `speaker` / `metadata` transcript fields remain unpersisted.
- Goal 12 is committed on `main` but has NOT been applied to production
  Supabase; the two pending migrations require separate, explicit approval.
- PostgreSQL runtime enforcement of the composite FKs / partial UNIQUE index /
  CHECK constraints remains unverified (offline DDL assertions only).

### Goal 13 integration contract

Goal 13 (context-aware chunking) consumes Goal 12 through a **read-only**
composition, `WordAlignmentService.get_aligned_sentences(alignment_run_id)`:
it loads a successful run's word rows and the asset's segments, re-segments
each segment's **verbatim** text with Goal 10 `segment_sentences`, tokenizes
the original text, and calls `derive_aligned_sentences`. It returns
`AlignedSentence` values (Goal 10 `Sentence` + `precision` +
`exact_start_ms` / `exact_end_ms` in integer ms + `word_count`), upgrading to
`WORD_ALIGNED` ONLY for fully covered sentences. It writes nothing, performs
no interpolation, and NEVER populates Goal 10's `exact_start_seconds` /
`exact_end_seconds`. Chunking must treat `SEGMENT_LEVEL` as an inherited
window rather than an exact boundary. Goal 12 itself implements **no**
chunking; Goal 13 consumes this contract as a separate goal.

---

## 10. Goal 13 — Sentence-aware, timestamp-preserving chunking

Goal 13 groups consecutive sentences into deterministic **retrieval chunks**.
It is a **pure, library-level** component (`app/transcripts/chunking.py`): no DB,
no I/O, no clock, no credentials, no ML model, no embeddings, no summarization,
and no rewrite/summary/paraphrase of transcript text. Chunk text is the verbatim
concatenation of its sentences.

> **Status:** **ACCEPTED** and committed on `main` in this change set. Pure
> library-level: **no table, no migration, no embedding column, no dependency
> change, no HTTP route**. It consumes Goal 12 `AlignedSentence` output
> read-only and has NOT been applied to production. During final review one
> diagnostic defect was found and fixed: an oversized first chunk with no
> overlap context no longer reports the false
> `overlap_trimmed_for_max_duration` diagnostic (regression test added).

### Input (Goal 10 + Goal 12 composition)

`chunk_sentences(aligned_sentences, *, parent=None, config=None)` consumes a
sequence of `AlignedSentence` values — exactly the output of
`WordAlignmentService.get_aligned_sentences(alignment_run_id)` described in §9.
It runs **no alignment** and builds no second timestamp contract. Goal 10's
`exact_start_seconds` / `exact_end_seconds` are never populated. For transcripts
with no word alignment, `derive_aligned_sentences(sentences, (), ())` yields
all-`SEGMENT_LEVEL` records that the chunker also accepts. The optional
`parent` is a `ChunkParentContext` (see "Provenance" below); the pure primitive
leaves it optional for backward compatibility, while `build_production_chunks`
requires it.

Word count uses the shared deterministic tokenizer
`word_contract.tokenize_original`; it is NOT the `AlignedSentence.word_count`
(that is the aligned-word count, zero for coarse sentences).

### Configuration (`ChunkingConfig`, frozen Pydantic model)

| Parameter | Default |
|---|---:|
| `target_duration_seconds` | 30.0 |
| `min_duration_seconds` | 10.0 |
| `max_duration_seconds` | 60.0 |
| `overlap_seconds` | 5.0 |
| `target_min_words` | 80 |
| `target_max_words` | 150 |

Validated at construction (finite/positive durations, `min <= target <= max`,
`overlap < max`, `1 <= min_words <= max_words`, no unknown fields, strict int
word counts). All thresholds flow from this single object; no magic numbers are
scattered through the algorithm. Passing a custom config is deterministic.

### Boundary-selection policy (deterministic)

Starting at the next unassigned sentence, sentences are accumulated greedily:

1. Sentences are never split, merged, reordered, or truncated. A single sentence
   exceeding `target_max_words` (or, when fully exact, `max_duration`) stays a
   one-sentence chunk (`single_sentence_exceeds_limits`) with a diagnostic.
2. Cumulative word count `w` and reliable temporal extent `d` are tracked. The
   60s `max_duration` is a hard cap **only for fully `WORD_ALIGNED` groups**.
3. The target is reached at the first boundary where `w >= target_min_words`
   OR (fully exact AND `d >= target_duration`). A **one-step lookahead** may add
   one more sentence only if it keeps an exact group exact, stays within the
   caps, and lowers a combined distance-to-targets score; ties stop earlier.
4. Chunks are finalized at valid sentence boundaries; each evidence sentence
   belongs to exactly one chunk. `boundary_reason` and `diagnostics` record the
   decision and any unavoidable deviation (e.g. `target_min_words_unmet`,
   `coarse_timestamp_precision`, `min_duration_unmet`).

### Overlap policy (deterministic)

Consecutive chunks share ~5s of overlap built from **complete sentences**: a
suffix of the previous chunk's evidence, re-tagged `role="context"`. Context is
never presented as new evidence (`evidence_start` separates them). Under fully
exact precision the suffix closest to 5s is chosen (ties -> shorter); under
coarse precision 5s is unmeasurable, so the honest bounded fallback is **one**
trailing sentence (`overlap_coarse_precision`). Overlap preserves global
sentence indices, IDs, and timestamps, never mutates source records, and is
trimmed (`overlap_trimmed_for_max_duration`) rather than inflating a fully exact
chunk past `max_duration`. That diagnostic is emitted **only when a context
sentence was actually removed**; an oversized single-sentence chunk has no
overlap context and never reports it.

### Timestamp precision (no fabrication)

* A chunk's `timestamp_precision` is the **minimum** precision across all its
  sentences: any `SEGMENT_LEVEL` sentence downgrades the whole chunk.
* `start_ms` / `end_ms` are the **union of the reliable source intervals**
  (`[min start, max end]`), in integer ms via `seconds_to_ms`. There is NO
  proportional split, interpolation, or character-offset timing. Per-sentence
  precision and verbatim parent-seconds windows are preserved on each ref.
* For a coarse/mixed chunk, duration thresholds (30/10/60/5s) **cannot be
  enforced accurately** (an inherited parent window does not shrink as more of
  its sentences are added). The algorithm falls back to word-count grouping and
  records `coarse_timestamp_precision`; duration targets are best-effort. This
  is documented, not silently weakened.

### Data contract (`Chunk`, `ChunkSentence` — frozen dataclasses)

`Chunk` carries `index`, verbatim `text`, ordered `sentences`, `start_ms` /
`end_ms` / `duration_ms` (+ `*_seconds` properties), `timestamp_precision`,
`word_count`, `evidence_start`, de-duplicated ordered `source_segment_refs`
(evidence only), `boundary_reason`, `diagnostics`, `overlap_sentence_indices`,
and the real parent provenance fields `video_id` / `transcript_asset_id` /
`content_right_id` (from `ChunkParentContext`; `None` only on the pure
optional-parent path). Each `ChunkSentence` preserves the global `index`,
`role`, verbatim `text`, `language`, Goal 10 `ordinal` / `char_start` /
`char_end`, `source_segment_ref`, per-sentence `timestamp_precision`, verbatim
parent-seconds window, exact ms (only when word-aligned), and `word_count`, plus
a derived `sentence_ref` property (below). Ordered `source_segment_refs` and
per-sentence `source_segment_ref` preserve multi-segment provenance; context
and evidence stay separated by `role` / `evidence_start`.

### Provenance — real parent IDs, composed stable references

A frozen, strictly validated `ChunkParentContext` carries the **real**
`video_id` and `transcript_asset_id` (strict `uuid.UUID`, no coercion) from the
existing Goal 09 `IngestionResult` / Goal 12 `TranscriptAlignmentRun` parent
context; `content_right_id` is propagated only when already present as a real
value (no new rights lookup). `from_ingestion_result()` / `from_alignment_run()`
read those real fields verbatim — no ID is fabricated or derived from an
unrelated field.

- `chunk_sentences(..., parent=None, ...)` (pure primitive) accepts `parent`
  optionally for backward compatibility.
- `build_production_chunks(..., parent=<required>, ...)` is the production
  composition wrapper: `parent` is a required keyword argument, must be a valid
  `ChunkParentContext`, and the wrapper **fails closed** if any produced chunk
  lacks the exact parent IDs. Production chunks can never silently omit
  `video_id` / `transcript_asset_id`.

`ChunkSentence.sentence_ref` is a deterministic **derived property** composed
from immutable `source_segment_ref` + `ordinal` + `char_start` + `char_end`
(format `source_segment_ref:ordinal:char_start:char_end`). **No sentence UUID is
invented.** Because overlap/context occurrences are built from the same
immutable sentence data, a sentence has the identical `sentence_ref` whether it
appears as evidence or as overlap context in another chunk.

There is no speaker identity and no persisted chunk identity (`chunk_id` /
database id): `Chunk` has a 0-based `index` for order only. Empty input returns
`()`.

### Persistence decision

Goal 13's acceptance criterion is "Context-aware chunks", a library-level
deterministic grouping. Architecture §9.1's future `video_chunks` table is NOT
implemented here: **no table, no migration, no embedding column, no new
dependency** (core `requirements.txt` / `requirements-lock.txt` unchanged). The
chunk dataclasses are pure value objects; a future indexing goal can persist
them via its own additive migration. Invalid input is rejected by
`validate_aligned_sentences` (no silent repair).

### Tests

`backend/tests/test_chunking.py` covers config validation, empty/single/
repeated/Unicode-Bangla input, English abbreviations/URLs/decimals via Goal 10
regression, Goal 10+12 word-aligned integration, coarse/mixed/exact precision,
one-segment-multiple-sentences, multiple/duplicate segment refs, max-size
handling, 30s target / 60s cap, complete-sentence overlap (5s, closest-valid,
trimming), evidence/context separation, determinism, and non-monotonic/invalid
timestamp rejection — plus real parent-ID propagation, strict
`ChunkParentContext` validation, production-path rejection of missing/invalid
parent, deterministic `sentence_ref`, identical refs for context/evidence
overlap, multi-segment provenance, pure-primitive backward compatibility, and
algorithm invariance when parent context is supplied.
`test_oversized_first_chunk_reports_no_overlap_trimmed_diagnostic` is the
final-review regression test for the spurious-diagnostic defect described
above.

### Known limitations / diagnostic semantics

- A chunk-level `start_ms` / `end_ms` / `duration_ms` / `word_count` is the
  union over **context + evidence** sentences, not evidence-only. When overlap
  exists, `start_ms` can precede the first evidence sentence. Per-sentence
  exact intervals and verbatim parent-seconds windows remain available on each
  `ChunkSentence`; nothing is interpolated.
- The dedicated single-oversized-sentence return path emits
  `single_sentence_exceeds_max_duration` / `single_sentence_exceeds_max_words`
  instead of the generic `coarse_timestamp_precision` diagnostic. A coarse
  oversized one-sentence chunk is still explicitly `SEGMENT_LEVEL`
  (`timestamp_precision` is the authoritative signal).
- Goal 13 adds no persistence; a future indexing goal owns the `video_chunks`
  table, the embedding column, and its own additive migration.

---

## 11. Goal 14 — BGE-M3 installation & environment validation

> **Status: ACCEPTED** (implementation and validation reviewed and accepted;
> recorded in `progress.md`).
> This goal is **installation and environment validation ONLY**. It adds no
> application code, no HTTP routes, no database objects, and no embeddings.
> Embedding generation (Goal 15), vector persistence (Goal 16), similarity
> search (Goal 17), hybrid retrieval / RRF / reranking / search API / frontend,
> and any pgvector schema or migration are explicitly **out of scope** and
> remain unimplemented. The heavy ML stack lives in a **separate, isolated
> interpreter**; the core `backend/.venv` is untouched.

### Hardware / environment assessment (measured, not assumed)

| Item | Value |
| --- | --- |
| OS | Windows 11 Pro, 64-bit (AMD64 / WindowsPE), Build 26200 |
| CPU | AMD Ryzen 9 9900X, 12 cores / 24 logical processors, x86-64 |
| GPU | AMD Radeon(TM) Graphics (integrated); **no NVIDIA GPU** |
| `nvidia-smi` / `nvcc` | not found |
| `CUDA_VISIBLE_DEVICES` | unset |
| Python | 3.12.10 (standalone CPython312), AMD64 |
| Physical RAM | 15.06 GB total; free ~0.57 GB pre-install, 1.87 GB pre-smoke, 2.39 GB post-smoke |
| Virtual memory | 40.21 GB total |
| Disk | D: ~9.2 GB free before install, ~8.0 GB after; C: ~2.3 GB free |

**Conclusion:** a **CPU-only** PyTorch (`+cpu`) wheel is required. No CUDA
wheel, CUDA toolkit, or NVIDIA driver is installed or assumed. No `nvidia-*` /
`triton` / `cuda*` wheels are present in the isolated env (verified via
`pip list`), and `torch.cuda.is_available()` is `False`.

### Dependency selection (why these pins, not "latest")

The target model's own cached `config.json` declares
`"transformers_version": "4.33.0"`, and `BGEM3FlagModel` is built on the
transformers 4.x API. At selection time PyPI latest `transformers` is `5.22.2`,
`sentence-transformers` 6.x requires `transformers>=5.0.0` /
`huggingface-hub>=1.3.0`, and `datasets` 5.x requires
`huggingface-hub>=1.31`. Rather than make an untested major-version jump for a
4.x-era model, this goal pins the **last transformers 4.x line** with a
mutually compatible 4.x-capable stack. `datasets==3.6.0` is used because 5.x is
incompatible with the `huggingface-hub<1.0` pin that transformers 4.x requires.

`FlagEmbedding==1.4.2` declares `transformers<6.0.0,>=4.44.2`; the chosen
`transformers==4.57.6` satisfies that. Verified direct pins (see
`requirements-embedding.txt`):

```
FlagEmbedding==1.4.2
torch==2.9.1+cpu
transformers==4.57.6
tokenizers==0.22.2
huggingface-hub==0.36.0
sentence-transformers==5.2.0
sentencepiece==0.2.3
safetensors==0.6.2
numpy==2.2.6
datasets==3.6.0
accelerate==1.10.0
peft==0.17.1
ir-datasets==0.6.3
protobuf==6.33.0
```

`torch==2.9.1+cpu` is the newest CPU-only wheel for `cp312/win_amd64` on
`https://download.pytorch.org/whl/cpu` at selection time. It is **not** on
PyPI, so installation **requires** `--extra-index-url
https://download.pytorch.org/whl/cpu`. No CUDA-specific wheels were selected.
The fully resolved environment (57 packages) is captured in
`requirements-embedding-lock.txt` (`pip freeze`); `python -m pip check` reports
`No broken requirements found.`

### Isolation

- Core `requirements.txt`, `requirements-lock.txt`, and
  `requirements-alignment.txt` (Goal 12) are **preserved unchanged**.
- New `requirements-embedding.txt` (direct pins) +
  `requirements-embedding-lock.txt` (resolved, 57 packages).
- New isolated interpreter `backend/.venv-embedding` (`pyvenv.cfg`: Python
  3.12.10, `include-system-site-packages = false`).
- `.gitignore` gained `.venv-*/`; `git check-ignore -v
  backend/.venv-embedding/pyvenv.cfg` → `.gitignore:231:.venv-*/`. The env is
  **not** tracked.
- Verified: `backend/.venv` (core) still has **no** `torch`, `transformers`,
  `FlagEmbedding`, `numpy`, `sentencepiece`, or `tokenizers`; `python -m pip
  check` there still reports `No broken requirements found.`
- No global Python / PATH changes, no Conda env changes, no Git remote changes.

### Model identity & configuration (official cached artifacts only)

Source: local Hugging Face cache
`~/.cache/huggingface/hub/models--BAAI--bge-m3` (official model card +
configs). No values are invented.

| Property | Value | Source |
| --- | --- | --- |
| Model ID | `BAAI/bge-m3` | model card |
| Dense dimension | **1024** | `config.json` `hidden_size`; `1_Pooling/config.json` `word_embedding_dimension`; model-card table |
| Architecture | `XLMRobertaModel` (`xlm-roberta`), 24 layers, 16 heads | `config.json` |
| `vocab_size` | 250002 | `config.json` |
| `max_position_embeddings` | 8194 | `config.json` |
| Max sequence length | **8192 tokens** | `sentence_bert_config.json` `max_seq_length`; model card |
| `torch_dtype` | `float32` | `config.json` |
| declared `transformers_version` | 4.33.0 | `config.json` |
| Pooling | CLS token, then L2 normalization | `1_Pooling/config.json`, `modules.json` (`2_Normalize`) |
| Multi-functionality | dense + multi-vector (ColBERT) + sparse retrieval | model card |
| Multi-linguality | > 100 working languages | model card |
| Multi-granularity | short sentences up to 8192-token documents | model card |

### Model cache & smoke-test decision

Cache inspection: `MODEL CACHE FOUND`, total **4.3 GB**, two snapshots:

- `snapshots/5617a9f6…/` — complete sentence-transformers layout:
  `config.json`, `pytorch_model.bin` (2,271,145,830 bytes), `tokenizer.json`,
  `sentencepiece.bpe.model`, `special_tokens_map.json`,
  `tokenizer_config.json`, `1_Pooling/config.json`,
  `sentence_bert_config.json`, `modules.json`, and the official `README.md`.
- `snapshots/9a0624b8…/` — `model.safetensors` only (2,271,064,456 bytes).

**Decision:** weights are present locally, so **no large download** was needed
and no approval for an expensive download was required. A minimal **offline**
inference smoke test was run against the complete snapshot with
`HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`, dense-only, `max_length=32`,
`batch_size=1`, `use_fp16=False`, `devices=["cpu"]`, `OMP_NUM_THREADS=4`.

### Smoke-test results (fresh run in this change set)

| # | Check | Result |
| --- | --- | --- |
| 1 | `BGEM3FlagModel` imports | OK |
| 2 | Model loads (CPU, fp32) | OK (5.4 s) |
| 3 | Tokenizer loads | OK (vocab 250002) |
| 4 | Inference succeeds | OK (0.54 s) |
| 5 | Dense shape exactly 1024 | `(1024,)` |
| 6 | All finite — no NaN/Inf | True (min -0.147266 / max 0.211798) |
| 7 | Repeated inference numerically consistent | max\|diff\| = **0** |

Additional observations: L2 norm = `1.000000` (normalized dense output),
process RSS = 2127 MB after load + 2 encodes, second encode 0.18 s, total wall
~6.1 s. Hard assertions on shape / finiteness / determinism all passed →
**SMOKE TEST: PASS (7/7)**.

### Resource & performance report

- Pre-smoke: physical RAM free 1.87 GB, virtual free 4.42 GB, D: 8.06 GB.
- Post-smoke: physical RAM free 2.39 GB, virtual free 4.09 GB, D: 8.04 GB.
  No disk growth and no resident leak; the smoke-test process exited cleanly.
- The float32 CPU load fits in ~2.1 GB RSS with `OMP_NUM_THREADS=4`. This is a
  **short-sequence smoke test**, NOT a throughput / latency benchmark. CPU
  inference on 8192-token inputs, batch pipelines, or GPU acceleration are
  unmeasured and out of scope. Available physical RAM is modest (~2 GB free at
  rest), so sustained heavy inference may be memory-constrained.

### Reproduction

```powershell
# from backend/
python -m venv .venv-embedding
.venv-embedding\Scripts\python.exe -m pip install -U pip
.venv-embedding\Scripts\python.exe -m pip install -r requirements-embedding.txt --extra-index-url https://download.pytorch.org/whl/cpu
.venv-embedding\Scripts\python.exe -m pip check
.venv-embedding\Scripts\python.exe -m pip freeze > requirements-embedding-lock.txt
```

### Known limitations / not verified / explicitly not done

- **No** production embedding generation, batch embedding pipeline, or
  persistence.
- **No** DB embedding columns, **no** `video_chunks` table, **no** pgvector /
  HNSW / GIN / FTS objects, **no** Alembic migration, and **no** production
  migration applied.
- **No** cosine similarity search, hybrid retrieval, RRF, reranker, search
  API, or frontend integration.
- Goal 15 is **not started**.
- `.env` was never read or exposed; no credentials or connection URLs are in
  the change set; the Git remote was not modified; nothing is committed or
  pushed for Goal 14 (all changes remain in the working tree for review).
- Live long-sequence / 8192-token inference, GPU inference, and CPU throughput
  benchmarking are not performed. Only the minimal dense-only CPU smoke test
  above is claimed.

---

## Validation performed

### Goal 14 — BGE-M3 installation & environment validation

Fresh validation in this change set. Isolated interpreter
`backend/.venv-embedding/Scripts/python.exe` (Python 3.12.10).

- `python -m pip check` → `No broken requirements found.`
- Exact versions confirmed: `torch 2.9.1+cpu` (`torch.version.cuda is None`,
  `torch.cuda.is_available() == False`), `transformers 4.57.6`,
  `tokenizers 0.22.2`, `huggingface-hub 0.36.0`, `sentence-transformers
  5.2.0`, `sentencepiece 0.2.3`, `safetensors 0.6.2`, `numpy 2.2.6`,
  `datasets 3.6.0`, `accelerate 1.10.0`, `peft 0.17.1`, `FlagEmbedding
  1.4.2`; `from FlagEmbedding import BGEM3FlagModel` → OK.
- `pip list` → **no** `nvidia-*` / `triton` / `cuda*` wheels (CPU-only stack
  confirmed).
- Offline smoke test (`HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`, dense-only,
  `max_length=32`): model loads, tokenizer loads, inference succeeds, dense
  shape `(1024,)`, all finite (no NaN/Inf), repeated inference max\|diff\| =
  `0` — **SMOKE TEST: PASS (7/7)**. Full detail in §11.
- Core `backend/.venv` regression: `python -m pip check` → clean, no heavy ML
  packages leaked in; full backend suite `754 passed, 1 skipped, 1 warning in
  4.83s` (unchanged from the Goal 12/13 acceptance baseline). Goal 14 adds no
  application code, so no test changes were required.
- `git diff --check` → clean; `git check-ignore` confirms `.venv-embedding` is
  ignored; `.env` not in change set; HEAD and `origin/main` unchanged
  (`33f2126…`), nothing staged, nothing committed or pushed.

**Not claimed:** any embedding generation, vector persistence, retrieval,
migration, or production deployment; long-sequence / GPU / throughput
benchmarking; live PostgreSQL runtime verification.

### Goal 12 + Goal 13 — finalization validation

All commands use the standalone `backend/.venv/Scripts/python.exe` (Python
3.12.10) and run from `backend/`. These are the results of the **fresh final
validation run performed in this change set**, not historical records.

- Goal 12-specific (6 files: `test_word_contract`,
  `test_word_timestamp_validation`, `test_sentence_timestamps`,
  `test_word_alignment_service`, `test_word_alignment_migration`,
  `test_alignment_provider_integration`): **270 passed, 1 skipped**. The single
  skip is the opt-in `@pytest.mark.live_whisperx` real-weights/real-audio tier
  (not executed).
- Goal 13-specific (`test_chunking.py`): **68 passed** (includes the final-review
  regression test `test_oversized_first_chunk_reports_no_overlap_trimmed_diagnostic`).
- Goal 09/10 regression (5 files: `test_transcript_validation`,
  `test_transcript_normalization`, `test_transcript_ingestion`,
  `test_sentence_segmentation`, `test_original_text_migration`): **161 passed**.
- Migration upgrade/downgrade + schema/ownership (4 files:
  `test_word_alignment_migration`, `test_original_text_migration`,
  `test_schema`, `test_ownership_integrity`): **153 passed**. Offline DDL
  rendering only — no live PostgreSQL connection.
- Complete backend test suite: **754 passed, 1 skipped, 1 warning** (the warning
  is the pre-existing `StarletteDeprecationWarning` from `fastapi.testclient`,
  not project code).
- Deterministic chunking subset (`-k determin`): **3 passed, 65 deselected**.
- Provenance subset (`-k "provenance or parent or sentence_ref or production or
  context"`): **21 passed, 47 deselected**.
- Timestamp / mixed-precision subset (`-k "precision or timestamp or mixed"`):
  **114 passed, 641 deselected, 1 warning**.
- Import/export check: `app.transcripts.__all__` (56 names) and
  `chunking.__all__` (23 names) fully resolvable; `chunk_sentences` and
  `build_production_chunks` both exported — **OK**.
- `python -m pip check` → `No broken requirements found.`
- `git diff --check` → clean (LF/CRLF warnings on Windows are informational).

**Not verified / not claimed:** live WhisperX 3.8.6 real-model-weights
execution, live PostgreSQL runtime enforcement of the composite FKs / partial
UNIQUE index / CHECK constraints (offline DDL rendering only; no disposable
PostgreSQL is available in this environment), and any production Supabase
migration. No coverage numbers are claimed.

### Goal 10 — multilingual sentence segmentation

- Pre-flight: branch `main`, clean tree, `HEAD == origin/main ==
  82b4f69d315d0e21ed10151c52bcc3eb2beeb1eb`, standalone Python 3.12.10 venv.
- New segmentation tests: **79 passed**.
- Full backend suite: **398 passed, 1 warning** (the single warning is the
  pre-existing `StarletteDeprecationWarning` from `fastapi.testclient`, not
  project code).
- Health / DB / schema / ownership / corrective-migration tests: **105 passed**.
- `python -m pip check` → `No broken requirements found.`
- `git diff --check` clean; no `.env`, `.venv`, credentials, or lockfile in the
  change set; **no new Alembic revision** (three approved revisions intact);
  **no dependency changes**; OpenAPI paths unchanged (`/health`, `/health/db`).

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
