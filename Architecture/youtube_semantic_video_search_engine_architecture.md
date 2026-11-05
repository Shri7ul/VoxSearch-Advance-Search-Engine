# YouTube Semantic Video Search Engine

## Full Architecture, Technical Approach, Database Design & Implementation Roadmap

**Document type:** Technical Architecture Specification\
**Project scope:** YouTube-first semantic video search\
**Primary output:** Relevant video segments with precise timestamps\
**Architecture style:** Modular monolith + asynchronous workers\
**Status:** Proposed production architecture

------------------------------------------------------------------------

# 1. Executive Summary

The system is a semantic search engine for spoken content inside videos.

A user submits a natural-language query such as:

> ModernBERT is not efficient for reranking.

The search engine should retrieve videos where a speaker discusses that
idea---even when the phrase does not appear in the video's title,
description, or published captions---and return the relevant transcript
segment with a timestamped YouTube link.

The system searches an **indexed and legally processable corpus**. It
cannot guarantee retrieval from every public YouTube video. Video
discovery, transcript access, rights to process content, and search
indexing are separate concerns.

### Core product capabilities

1.  Discover relevant YouTube videos and channels through permitted
    interfaces.
2.  Acquire transcripts from authorized, licensed, or user-provided
    sources.
3.  Transcribe permitted audio when rights and access allow it.
4.  Preserve sentence- and word-level timestamps.
5.  Chunk transcripts without losing semantic context.
6.  Index chunks for lexical and semantic retrieval.
7.  Combine dense retrieval and keyword retrieval.
8.  Rerank candidates for query relevance.
9.  Return timestamped evidence snippets and direct video links.
10. Support deletion, revocation, provenance, and data-retention
    controls.

### Primary design principle

**Search spoken content, not just published metadata.**

The primary search result is a transcript segment, not an AI-generated
answer.

------------------------------------------------------------------------

# 2. Scope and Non-Goals

## 2.1 In scope

-   YouTube video metadata discovery.
-   Creator-authorized and licensed transcript ingestion.
-   User-provided media/transcript ingestion where the user has rights.
-   ASR for permitted audio.
-   Timestamp alignment.
-   Semantic and lexical retrieval.
-   Reranking and result aggregation.
-   Search API and web interface.
-   Background ingestion and indexing.
-   Monitoring, evaluation, and compliance workflows.

## 2.2 Explicit non-goals for the first release

-   Claiming to search all public YouTube videos.
-   Downloading or scraping arbitrary YouTube videos.
-   Circumventing caption, authentication, or access restrictions.
-   Real-time indexing of every newly uploaded YouTube video.
-   AI-generated summaries or conversational answers.
-   Kafka, Kubernetes, and multiple distributed databases before scale
    requires them.
-   Multi-platform ingestion in the first release.

## 2.3 Coverage definition

"Broad coverage" means continuously expanding a searchable corpus
through permitted video discovery and authorized content acquisition. It
does **not** mean unrestricted access to every public YouTube
transcript.

A video may be discoverable as metadata but not eligible for transcript
acquisition or indexing. The system must represent these states
separately.

------------------------------------------------------------------------

# 3. Critical YouTube Access and Compliance Constraints

This is a foundational architecture constraint, not a later legal
checklist.

The YouTube Data API can discover videos and retrieve metadata. The
captions API has important restrictions:

-   `captions.list` returns caption-track metadata; it does not return
    the transcript text.
-   `captions.download` requires authorization with sufficient
    permission for the video.
-   An API key alone does not grant permission to download arbitrary
    public-video captions.
-   YouTube API policies restrict unauthorized downloading, scraping,
    and storage of audiovisual content and API data.

Therefore, do not design the product around downloading every public
video with `yt-dlp` and transcribing it with Whisper. That is not a
defensible production ingestion strategy.

## 3.1 Supported acquisition paths

  -----------------------------------------------------------------------
  Source                  Intended use            Required control
  ----------------------- ----------------------- -----------------------
  Creator OAuth           Creator-authorized      OAuth scopes,
                          channel/video content   authorization record,
                                                  revocation

  Licensed transcript     Third-party transcript  Contract, scope,
  provider                data                    retention and
                                                  processing rights

  User-provided           Transcript supplied by  User declaration and
  transcript              a user with rights      provenance

  User-provided           ASR on supplied media   Rights declaration and
  audio/video                                     retention policy

  Official YouTube API    Discovery and permitted API quota and YouTube
                          metadata                policy compliance
  -----------------------------------------------------------------------

A creator authorization flow must not be assumed to grant rights beyond
the actual permissions and applicable terms. Store the authorization
scope and evidence.

## 3.2 Rights gate

Every content item must pass an authorization check before transcript
processing.

Required permission dimensions:

-   May the system transcribe this content?
-   May it store the transcript?
-   May it generate and store embeddings?
-   May it expose transcript snippets in search results?
-   May it retain derived data, and for how long?
-   What happens when authorization expires or is revoked?

If a video is discoverable but not authorized for transcript processing,
retain only metadata that the applicable API terms permit. Do not create
transcript embeddings.

## 3.3 Compliance worker

A dedicated worker should:

1.  Detect expired or revoked authorization.
2.  Mark affected content as unavailable for search.
3.  Remove transcript segments and embeddings when required.
4.  Invalidate search caches.
5.  Remove derived artifacts where required.
6.  Record an auditable deletion event.
7.  Refresh API-derived metadata within the applicable policy
    requirements.

Obtain a legal and YouTube API policy review before public or commercial
launch.

------------------------------------------------------------------------

# 4. High-Level System Architecture

``` mermaid
flowchart TB
    U[User] --> FE[Next.js Search UI]
    FE --> API[FastAPI API Gateway]

    API --> QS[Query Understanding]
    QS --> DR[Dense Retrieval]
    QS --> LR[Lexical Retrieval]

    DR --> RRF[Reciprocal Rank Fusion]
    LR --> RRF
    RRF --> RR[Cross-Encoder Reranker]
    RR --> AGG[Segment Aggregation]
    AGG --> API
    API --> FE

    DISC[YouTube Discovery Service] --> META[Video Metadata]
    META --> RIGHTS[Rights and Authorization Gate]
    RIGHTS -->|Authorized source| ING[Transcript Acquisition]
    RIGHTS -->|Not authorized| METAONLY[Metadata-only state]

    ING --> NORM[Transcript Normalization]
    NORM --> ALIGN[Timestamp Alignment]
    ALIGN --> CHUNK[Sentence-aware Chunking]
    CHUNK --> EMB[Embedding Worker]
    EMB --> INDEX[PostgreSQL + pgvector + FTS]

    DISC --> Q[Redis / Celery Queue]
    ING --> Q
    CHUNK --> Q
    EMB --> Q

    INDEX --> DR
    INDEX --> LR
```

## 4.1 Two independent pipelines

### Ingestion pipeline

`Discover → Authorize → Acquire → Normalize → Align → Chunk → Embed → Index`

### Search pipeline

`Query → Normalize → Dense + Lexical Retrieval → RRF → Rerank → Aggregate → Timestamped Results`

The search request must never synchronously transcribe or index a large
set of videos.

------------------------------------------------------------------------

# 5. Technology Stack

  -----------------------------------------------------------------------
  Layer                   Selected technology     Responsibility
  ----------------------- ----------------------- -----------------------
  Frontend                Next.js + TypeScript    Search UI and result
                                                  presentation

  Backend                 Python + FastAPI        REST API and
                                                  application services

  Relational database     PostgreSQL              Metadata, rights, jobs,
                                                  transcripts

  Vector search           pgvector                Dense semantic
                                                  retrieval

  Lexical search          PostgreSQL full-text    Exact terms and
                          search                  phrase-oriented
                                                  retrieval

  Embeddings              BGE-M3                  Multilingual
                                                  dense/sparse
                                                  representations

  Reranking               BGE-Reranker-v2-M3      Candidate relevance
                                                  reranking

  ASR                     faster-whisper          Transcription of
                                                  permitted audio

  Alignment               WhisperX or another     Word-level timestamp
                          validated aligner       alignment

  Background queue        Celery + Redis          Asynchronous jobs

  Cache                   Redis                   Query and metadata
                                                  cache

  Object storage          S3-compatible storage   Authorized source
                                                  assets, if retention is
                                                  permitted

  Containers              Docker                  Reproducible deployment

  Metrics                 Prometheus + Grafana    System monitoring

  Error tracking          Sentry or equivalent    Application error
                                                  reporting
  -----------------------------------------------------------------------

## 5.1 Why this stack

### PostgreSQL + pgvector

Start with one database for relational metadata, full-text search, and
vector search. This reduces operational complexity while the corpus is
small or medium-sized.

Use a separate search engine or vector database only when measured query
latency, index size, concurrency, or operational constraints justify it.

### BGE-M3

BGE-M3 is a multilingual embedding model that supports dense and sparse
retrieval approaches. Its dense vectors use 1024 dimensions. For the
initial dense index, use:

``` sql
embedding vector(1024)
```

Do not assume that one embedding model is optimal for every language or
technical domain. Benchmark it on a labeled query set.

### Reranker

Use a cross-encoder reranker on a limited candidate set rather than
applying it to the entire corpus. Reranking is more expensive than
first-stage retrieval.

------------------------------------------------------------------------

# 6. Video Discovery Architecture

## 6.1 Discovery sources

Use permitted YouTube Data API operations for video and channel
discovery.

Potential discovery strategies:

-   Curated channel registry.
-   Channel uploads playlist enumeration.
-   Topic and keyword-based discovery.
-   Scheduled metadata refresh.
-   New-video polling for registered channels.
-   Duplicate video ID elimination.

## 6.2 Discovery workflow

1.  Select a discovery source.
2.  Call the relevant official API endpoint.
3.  Respect quota allocation and pagination.
4.  Normalize the returned metadata.
5.  Upsert the channel and video records.
6.  Record the source and discovery timestamp.
7.  Enqueue an authorization check.
8.  Do not enqueue transcript processing until the rights gate passes.

## 6.3 Discovery limitations

-   Search results are not an exhaustive index of YouTube.
-   API quota constrains discovery throughput.
-   Search ranking and indexing coverage can change.
-   Discoverability does not imply transcript access.
-   Metadata retention and refresh must follow applicable API policies.

Use a quota budget, backoff, and scheduled discovery rather than
uncontrolled polling.

------------------------------------------------------------------------

# 7. Transcript Acquisition and Processing

## 7.1 Transcript source priority

For an authorized video, use this source priority:

1.  Creator-authorized caption/transcript source.
2.  Licensed transcript provider.
3.  User-provided transcript.
4.  ASR on audio that the system is authorized to process.

Record the exact source for every transcript asset.

## 7.2 Normalized transcript format

Use integer milliseconds for timestamps.

``` json
{
  "video_id": "VIDEO_ID",
  "language": "en",
  "source_type": "authorized_asr",
  "segments": [
    {
      "start_ms": 120000,
      "end_ms": 124500,
      "text": "ModernBERT struggles with certain reranking workloads.",
      "speaker_id": "speaker_1"
    },
    {
      "start_ms": 124500,
      "end_ms": 129000,
      "text": "Especially when computational resources are limited.",
      "speaker_id": "speaker_1"
    }
  ]
}
```

## 7.3 Transcript normalization

Processing steps:

-   Validate timestamp order.
-   Remove empty or invalid segments.
-   Normalize whitespace and punctuation.
-   Preserve original transcript text.
-   Detect language.
-   Preserve speaker labels where available.
-   Record source and model version.
-   Detect duplicate transcript imports.
-   Keep a checksum for integrity and deduplication.

Do not overwrite source text with normalized text. Preserve both where
practical.

## 7.4 ASR

Use faster-whisper for authorized audio.

Store:

-   ASR model identifier and version.
-   Language and language-detection confidence.
-   Segment timestamps.
-   Word timestamps if available.
-   Transcription configuration.
-   Processing timestamp.
-   Quality flags.

ASR output is not guaranteed to be correct. Technical names such as
`ModernBERT`, `ColBERT`, and `BGE-M3` are particularly important to
evaluate.

## 7.5 Word-level alignment

Word-level timestamps enable more precise result links than broad
transcript chunks.

Recommended representation:

-   Transcript sentence: semantic and lexical retrieval unit.
-   Transcript chunk: contextual retrieval unit.
-   Word alignment: timestamp precision source.

The search engine can retrieve a chunk, identify the matching sentence,
and return the sentence's timestamp rather than the entire chunk
boundary.

------------------------------------------------------------------------

# 8. Chunking Strategy

Avoid blindly splitting transcripts into fixed token counts. A chunk
boundary should preserve a complete thought whenever possible.

## 8.1 Initial chunking parameters

These are starting values for experiments, not universal optima.

  Parameter             Initial value
  ------------------ ----------------
  Target duration          30 seconds
  Minimum duration         10 seconds
  Maximum duration         60 seconds
  Overlap                   5 seconds
  Target text size      80--150 words
  Boundary             Sentence-aware

Example:

``` text
Chunk 1: 00:10–00:40
Chunk 2: 00:35–01:05
Chunk 3: 01:00–01:30
```

## 8.2 Chunking algorithm

1.  Start from sentence boundaries.
2.  Accumulate sentences until the target duration or text size is
    reached.
3.  Do not split a sentence unless it exceeds the maximum limit.
4.  Add a small overlap with the preceding chunk.
5.  Preserve the exact start and end timestamp.
6.  Store the source segment IDs associated with each chunk.

## 8.3 Context preservation

For each indexed chunk, retain links to:

-   Previous transcript sentence.
-   Current transcript sentence(s).
-   Next transcript sentence.
-   Speaker ID.
-   Original timestamps.

The embedding text can include a small amount of contextual text, while
the displayed evidence should identify the actual matching sentence.

------------------------------------------------------------------------

# 9. Database Architecture

## 9.1 Logical tables

  Table                   Responsibility
  ----------------------- ------------------------------------------
  `channels`              Channel identity and authorization state
  `videos`                Video metadata and indexing state
  `content_rights`        Permission scope and evidence
  `transcript_assets`     Transcript source and version
  `transcript_segments`   Timestamped sentences/segments
  `transcript_words`      Optional word-level alignment
  `video_chunks`          Chunk text and embeddings
  `ingestion_jobs`        Job state and retry metadata
  `deletion_requests`     Revocation and deletion workflow
  `search_events`         Privacy-conscious search analytics

## 9.2 Extensions

``` sql
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
```

## 9.3 Channels

``` sql
CREATE TABLE channels (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    youtube_channel_id TEXT UNIQUE NOT NULL,
    channel_title TEXT,

    authorization_status TEXT NOT NULL
        DEFAULT 'pending',

    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);
```

## 9.4 Videos

``` sql
CREATE TABLE videos (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    youtube_video_id TEXT UNIQUE NOT NULL,
    channel_id UUID REFERENCES channels(id),

    title TEXT,
    description TEXT,

    duration_seconds INTEGER,
    language TEXT,

    published_at TIMESTAMPTZ,

    indexing_status TEXT NOT NULL
        DEFAULT 'discovered',

    metadata_checked_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
```

## 9.5 Content rights

``` sql
CREATE TABLE content_rights (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    video_id UUID REFERENCES videos(id) NOT NULL,

    rights_basis TEXT NOT NULL,
    granted_by TEXT NOT NULL,

    can_transcribe BOOLEAN DEFAULT FALSE,
    can_store_transcript BOOLEAN DEFAULT FALSE,
    can_embed BOOLEAN DEFAULT FALSE,
    can_search BOOLEAN DEFAULT FALSE,

    valid_from TIMESTAMPTZ,
    valid_until TIMESTAMPTZ,

    revoked_at TIMESTAMPTZ,
    evidence_reference TEXT,

    created_at TIMESTAMPTZ DEFAULT NOW()
);
```

The application must validate the active permission record before every
ingestion or re-indexing operation. A Boolean stored in a row is not
sufficient if the authorization has expired or been revoked.

## 9.6 Transcript assets

``` sql
CREATE TABLE transcript_assets (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    video_id UUID REFERENCES videos(id) NOT NULL,

    source_type TEXT NOT NULL,
    language TEXT NOT NULL,

    transcript_version INTEGER DEFAULT 1,
    word_timestamps_available BOOLEAN DEFAULT FALSE,

    checksum TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
```

## 9.7 Transcript segments

``` sql
CREATE TABLE transcript_segments (
    id BIGSERIAL PRIMARY KEY,

    asset_id UUID REFERENCES transcript_assets(id)
        ON DELETE CASCADE,

    start_ms BIGINT NOT NULL,
    end_ms BIGINT NOT NULL,

    text TEXT NOT NULL,
    speaker_id TEXT,

    CHECK (start_ms >= 0),
    CHECK (end_ms > start_ms)
);

CREATE INDEX idx_segments_asset_time
ON transcript_segments(asset_id, start_ms);
```

## 9.8 Video chunks

``` sql
CREATE TABLE video_chunks (
    id BIGSERIAL PRIMARY KEY,

    video_id UUID REFERENCES videos(id)
        ON DELETE CASCADE,

    asset_id UUID REFERENCES transcript_assets(id)
        ON DELETE CASCADE,

    chunk_index INTEGER NOT NULL,

    start_ms BIGINT NOT NULL,
    end_ms BIGINT NOT NULL,

    chunk_text TEXT NOT NULL,
    language TEXT,

    embedding vector(1024),

    search_vector TSVECTOR GENERATED ALWAYS AS (
        to_tsvector('english', chunk_text)
    ) STORED,

    created_at TIMESTAMPTZ DEFAULT NOW(),

    UNIQUE(asset_id, chunk_index),

    CHECK (end_ms > start_ms)
);
```

The generated `search_vector` above is an English-first example. A
multilingual production implementation should select a language-aware
text-search configuration or maintain language-specific lexical indexes.

## 9.9 Optional word alignment table

``` sql
CREATE TABLE transcript_words (
    id BIGSERIAL PRIMARY KEY,

    segment_id BIGINT REFERENCES transcript_segments(id)
        ON DELETE CASCADE,

    word_index INTEGER NOT NULL,
    start_ms BIGINT NOT NULL,
    end_ms BIGINT NOT NULL,

    word TEXT NOT NULL,

    CHECK (start_ms >= 0),
    CHECK (end_ms > start_ms),

    UNIQUE(segment_id, word_index)
);

CREATE INDEX idx_words_segment_time
ON transcript_words(segment_id, start_ms);
```

## 9.10 Search indexes

``` sql
CREATE INDEX idx_chunks_embedding
ON video_chunks
USING hnsw (embedding vector_cosine_ops);

CREATE INDEX idx_chunks_fts
ON video_chunks
USING gin (search_vector);

CREATE INDEX idx_chunks_video
ON video_chunks(video_id);
```

For large datasets, benchmark HNSW parameters, index build time, memory,
recall, and query latency. Approximate nearest-neighbor search trades
some recall for speed.

## 9.11 Database design rules

-   Keep video metadata separate from transcript content.
-   Keep transcript provenance and version history.
-   Use foreign keys and cascading deletion where appropriate.
-   Keep rights evidence and authorization lifecycle auditable.
-   Store timestamps as integer milliseconds.
-   Track embedding model and version.
-   Do not mix content with different rights or retention rules without
    provenance.
-   Use database migrations rather than manually changing production
    schema.

------------------------------------------------------------------------

# 10. Retrieval Architecture

The recommended retrieval pipeline is:

`Query normalization → Dense retrieval + Lexical retrieval → RRF → Reranking → Segment aggregation → Final results`

## 10.1 Query understanding

Perform lightweight processing:

-   Whitespace and punctuation normalization.
-   Language detection.
-   Technical entity extraction.
-   Optional spelling correction.
-   Query embedding generation.

Do not rewrite the query so aggressively that technical terms or
negation are lost.

Example:

``` text
Original:
ModernBERT is not efficient for reranking

Entities:
ModernBERT, reranking

Intent:
Negative performance claim

Language:
English
```

## 10.2 Dense retrieval

Generate a query embedding with the same model/version used for the
indexed vectors.

``` python
query_embedding = embedding_model.encode(
    query,
    normalize_embeddings=True
)
```

Example SQL:

``` sql
SELECT
    id,
    video_id,
    chunk_text,
    start_ms,
    end_ms,
    embedding <=> :query_embedding AS distance
FROM video_chunks
WHERE language = 'en'
  AND embedding IS NOT NULL
ORDER BY embedding <=> :query_embedding
LIMIT 100;
```

Cosine distance is lower for more similar vectors.

## 10.3 Lexical retrieval

``` sql
SELECT
    id,
    video_id,
    chunk_text,
    start_ms,
    end_ms,
    ts_rank_cd(
        search_vector,
        websearch_to_tsquery('english', :query)
    ) AS lexical_score
FROM video_chunks
WHERE search_vector @@ websearch_to_tsquery(
    'english',
    :query
)
ORDER BY lexical_score DESC
LIMIT 100;
```

Lexical retrieval is especially useful for exact technical terms,
identifiers, acronyms, and product names.

## 10.4 Reciprocal Rank Fusion

Do not directly add raw vector distances and full-text scores. They have
different scales.

Use RRF:

\[ RRF(d) = `\sum`{=tex}\_{r `\in `{=tex}R}
`\frac{1}{k + rank_r(d)}`{=tex} \]

Where:

-   (d): candidate document.
-   (R): retrieval result lists.
-   (rank_r(d)): rank of document (d) in list (r).
-   (k): commonly 60 as an initial value.

Python:

``` python
def reciprocal_rank_fusion(result_lists, k=60):
    scores = {}

    for results in result_lists:
        for rank, item in enumerate(results, start=1):
            doc_id = item["id"]
            scores[doc_id] = (
                scores.get(doc_id, 0.0)
                + 1.0 / (k + rank)
            )

    return sorted(
        scores.items(),
        key=lambda item: item[1],
        reverse=True
    )
```

## 10.5 Cross-encoder reranking

Rerank a bounded candidate set, for example the top 50 candidates after
RRF.

``` python
pairs = [
    (query, candidate["context_text"])
    for candidate in candidates
]

scores = reranker.predict(pairs)

ranked_results = sorted(
    zip(candidates, scores),
    key=lambda item: item[1],
    reverse=True
)
```

The reranker score is not automatically a calibrated probability. Do not
label a raw score of `0.91` as "91% confidence."

## 10.6 Context-aware reranking

For each candidate, provide:

-   Matching chunk.
-   Previous sentence or chunk context.
-   Following sentence or chunk context.
-   Speaker context when available.

This helps distinguish:

-   Positive from negative claims.
-   A speaker's claim from a quoted claim.
-   A conclusion from a hypothetical example.
-   A statement from a correction or qualification.

## 10.7 Result aggregation

If several adjacent chunks from one video match:

1.  Deduplicate identical chunk IDs.
2.  Merge overlapping time ranges.
3.  Combine adjacent relevant sentences when appropriate.
4.  Preserve the best evidence sentence.
5.  Use the sentence's timestamp for the primary deep link.
6.  Keep a context window for the displayed snippet.

Do not collapse separate discussions from different parts of a video
into one result.

------------------------------------------------------------------------

# 11. Search API Design

Use versioned REST endpoints.

  --------------------------------------------------------------------------------
  Method                  Endpoint                         Purpose
  ----------------------- -------------------------------- -----------------------
  POST                    `/api/v1/search`                 Search transcript
                                                           corpus

  GET                     `/api/v1/videos/{id}`            Video metadata

  GET                     `/api/v1/videos/{id}/segments`   Timestamped transcript
                                                           segments

  POST                    `/api/v1/channels/connect`       Creator authorization

  POST                    `/api/v1/ingestion/submit`       Authorized content
                                                           submission

  GET                     `/api/v1/ingestion/jobs/{id}`    Job status

  DELETE                  `/api/v1/content/{id}`           Remove eligible indexed
                                                           content
  --------------------------------------------------------------------------------

## 11.1 Search request

``` json
{
  "query": "ModernBERT is not efficient for reranking",
  "filters": {
    "language": ["en"],
    "duration_min": 60,
    "duration_max": 7200,
    "channel_ids": []
  },
  "retrieval": {
    "mode": "hybrid",
    "rerank": true
  },
  "pagination": {
    "page": 1,
    "limit": 20
  }
}
```

## 11.2 Search response

``` json
{
  "query": "ModernBERT is not efficient for reranking",
  "total": 1,
  "results": [
    {
      "video": {
        "youtube_id": "VIDEO_ID",
        "title": "Modern NLP Architectures",
        "channel": "AI Research Channel"
      },
      "match": {
        "text": "ModernBERT struggles with reranking workloads.",
        "start_ms": 2901000,
        "end_ms": 2919000,
        "match_type": "semantic"
      },
      "relevance": {
        "rank": 1,
        "reranker_score": 0.91
      },
      "url": "https://www.youtube.com/watch?v=VIDEO_ID&t=2901s"
    }
  ]
}
```

The response above is an illustrative example. The score is a model
output, not a calibrated probability.

## 11.3 Timestamp URL helper

``` python
def youtube_timestamp_url(video_id: str, start_ms: int):
    seconds = start_ms // 1000

    return (
        f"https://www.youtube.com/watch"
        f"?v={video_id}&t={seconds}s"
    )
```

## 11.4 API requirements

-   Validate and limit query length.
-   Enforce pagination limits.
-   Apply request rate limits.
-   Use authentication for private or creator-specific operations.
-   Return stable error codes.
-   Keep model and retrieval versions in internal logs.
-   Do not expose internal storage paths or authorization secrets.
-   Make ingestion submission idempotent.

------------------------------------------------------------------------

# 12. Backend Project Structure

``` text
backend/
├── app/
│   ├── main.py
│   ├── api/
│   │   └── v1/
│   │       ├── search.py
│   │       ├── videos.py
│   │       ├── channels.py
│   │       └── ingestion.py
│   ├── core/
│   │   ├── config.py
│   │   ├── security.py
│   │   └── logging.py
│   ├── models/
│   │   ├── video.py
│   │   ├── transcript.py
│   │   └── chunk.py
│   ├── schemas/
│   │   ├── search.py
│   │   └── video.py
│   ├── services/
│   │   ├── discovery.py
│   │   ├── rights.py
│   │   ├── transcript.py
│   │   ├── alignment.py
│   │   ├── chunking.py
│   │   ├── embedding.py
│   │   ├── retrieval.py
│   │   ├── reranking.py
│   │   └── timestamp.py
│   ├── workers/
│   │   ├── discovery_worker.py
│   │   ├── ingestion_worker.py
│   │   ├── embedding_worker.py
│   │   └── compliance_worker.py
│   └── db/
│       ├── session.py
│       └── migrations/
├── tests/
├── requirements.txt
└── Dockerfile
```

------------------------------------------------------------------------

# 13. Frontend Architecture

Use Next.js and TypeScript.

## 13.1 Main search page

The page should contain:

-   Prominent natural-language search bar.
-   Search mode selector: Hybrid, Semantic, Exact.
-   Language filter.
-   Duration filter.
-   Channel filter.
-   Result count and latency.
-   Timestamped transcript snippets.
-   Video thumbnail, title, and channel.
-   Direct "Watch from timestamp" action.
-   Previous/next context expansion.

## 13.2 Result card

Each result should show:

``` text
[Video thumbnail]  Video title
                   Channel · Duration

                   Match: Semantic

                   48:21 – 48:39

                   "ModernBERT struggles with certain
                    reranking workloads..."

                   [Watch from 48:21]
                   [Show surrounding context]
```

## 13.3 Search behavior

-   Debounce query input where appropriate.
-   Cancel obsolete in-flight requests.
-   Preserve filter state in URL query parameters.
-   Use stable pagination.
-   Display loading, empty, and error states.
-   Clearly distinguish indexed corpus coverage from all YouTube
    content.
-   Do not display raw reranker scores as percentages.

------------------------------------------------------------------------

# 14. Background Job Architecture

Use Celery + Redis initially.

## 14.1 Job types

  Job                        Responsibility
  -------------------------- ------------------------------
  `discover_channel`         Discover channel videos
  `refresh_video_metadata`   Refresh permitted metadata
  `check_content_rights`     Validate authorization
  `acquire_transcript`       Obtain permitted transcript
  `transcribe_media`         ASR on authorized media
  `align_transcript`         Word-level alignment
  `chunk_transcript`         Sentence-aware chunking
  `generate_embeddings`      Create embeddings
  `index_chunks`             Update search indexes
  `process_revocation`       Delete affected derived data

## 14.2 Reliability controls

-   Idempotency keys.
-   Exponential retry backoff.
-   Maximum retry count.
-   Dead-letter queue.
-   Job lease or timeout.
-   Structured logs.
-   Job-level provenance.
-   Periodic stuck-job detection.

## 14.3 State machine

``` text
DISCOVERED
    |
    v
AUTHORIZATION_CHECK
    |
    +---- Not authorized ---> METADATA_ONLY
    |
    v
TRANSCRIPT_ACQUISITION
    |
    v
NORMALIZATION
    |
    v
ALIGNMENT
    |
    v
CHUNKING
    |
    v
EMBEDDING
    |
    v
INDEXING
    |
    v
READY
```

Failure states should be explicit:

``` text
FAILED_RETRYABLE
FAILED_PERMANENT
AUTHORIZATION_EXPIRED
REVOKED
DELETION_PENDING
DELETED
```

------------------------------------------------------------------------

# 15. Search Evaluation and Benchmarking

Do not optimize the system based on a handful of manually tested
queries.

Create a labeled evaluation dataset with:

-   Query text.
-   Relevant video IDs.
-   Relevant segment IDs.
-   Expected start and end timestamps.
-   Relevance grade.
-   Query type.
-   Language.
-   Technical entities.
-   Negation/contrast label where applicable.

## 15.1 Evaluation metrics

  Metric            Purpose
  ----------------- ------------------------------------
  Recall@10         Relevant result coverage in top 10
  Precision@10      Relevance of top 10
  MRR@10            Rank of first relevant result
  nDCG@10           Graded ranking quality
  Timestamp MAE     Timestamp error
  WER               ASR transcription error
  P50/P95 latency   Search responsiveness

## 15.2 Initial evaluation targets

These are proposed targets, not achieved results.

  Metric               Initial target
  ------------------ ----------------
  Recall@10                     ≥ 85%
  Precision@10                  ≥ 70%
  MRR@10                       ≥ 0.75
  Timestamp MAE           ≤ 3 seconds
  Search P95              ≤ 2 seconds
  API availability            ≥ 99.5%

Measure metrics separately by query type and language. A good aggregate
score can hide poor performance on technical names, negation, or
multilingual queries.

## 15.3 Evaluation query categories

Start with at least 200 manually labeled queries:

  Category                            Query count
  --------------------------------- -------------
  Exact technical terms                        50
  Semantic paraphrases                         50
  Negative and contrastive claims              50
  Typo-heavy queries                           25
  Bangla-English mixed queries                 25
  **Total**                               **200**

For each query, label the actual relevant segment and acceptable
timestamp range.

------------------------------------------------------------------------

# 16. Production Deployment

## 16.1 Initial deployment topology

``` text
Cloudflare / CDN
       |
Next.js Frontend
       |
FastAPI Application
       |
       +---- PostgreSQL + pgvector
       |
       +---- Redis
                  |
              Celery Queue
                  |
          +-------+-------+
          |               |
      CPU Workers     GPU Workers
      Discovery       ASR
      Chunking        Embedding
      Metadata        Reranking
```

## 16.2 Recommended deployment allocation

  Component        Deployment
  ---------------- -----------------------
  Frontend         Vercel or equivalent
  API              Docker-based VPS
  PostgreSQL       Managed PostgreSQL
  Redis            Managed Redis
  ASR/embedding    Separate GPU worker
  Object storage   S3-compatible service
  Monitoring       Prometheus + Grafana

## 16.3 Scaling strategy

### Early stage

-   One FastAPI service.
-   One PostgreSQL instance.
-   One Redis instance.
-   A small number of workers.
-   CPU inference for limited experiments if performance is acceptable.

### Growth stage

-   Separate API and worker deployments.
-   Dedicated GPU inference workers.
-   Connection pooling.
-   Query and embedding caching.
-   PostgreSQL read replicas if justified.
-   Partitioning or index optimization based on measured workload.

### Large scale

Consider a dedicated vector database or distributed search
infrastructure only after benchmarks demonstrate that PostgreSQL cannot
meet the required throughput, recall, or latency.

------------------------------------------------------------------------

# 17. Security, Privacy, and Compliance

Production requirements:

1.  OAuth tokens encrypted at rest.
2.  Secrets stored in a secret manager.
3.  No API keys in frontend bundles.
4.  Per-user and per-channel authorization.
5.  Input validation and query rate limiting.
6.  Audit logs for authorization and deletion events.
7.  Rights-aware transcript access.
8.  Data retention and deletion policies.
9.  Backup encryption.
10. Monitoring for failed deletion jobs.
11. Search analytics designed to minimize personal data.
12. No indexing of unauthorized content.

Revocation must invalidate both the original content and derived search
artifacts where required.

------------------------------------------------------------------------

# 18. Monitoring and Observability

Track system metrics by component.

## Ingestion

-   Videos discovered per hour.
-   Authorization pass/fail rate.
-   Transcript acquisition success rate.
-   ASR throughput.
-   Average processing duration.
-   Failed jobs and retry count.
-   Indexing lag.
-   Revocation/deletion backlog.

## Retrieval

-   Search requests per minute.
-   P50/P95/P99 latency.
-   Dense retrieval latency.
-   Lexical retrieval latency.
-   Reranking latency.
-   Candidate count.
-   Empty-result rate.
-   Cache hit rate.
-   Search errors.

## Search quality

-   Recall@10.
-   Precision@10.
-   MRR@10.
-   nDCG@10.
-   Timestamp MAE.
-   Query category performance.

------------------------------------------------------------------------

# 19. Ten-Week Implementation Roadmap

  -----------------------------------------------------------------------
  Week                    Workstream              Deliverable
  ----------------------- ----------------------- -----------------------
  1                       FastAPI, Next.js,       Running infrastructure
                          PostgreSQL, pgvector,   
                          Docker                  

  2                       YouTube metadata        Video catalog
                          discovery and quota     
                          handling                

  3                       Creator authorization   Authorized transcript
                          and licensed transcript corpus
                          ingestion               

  4                       ASR and timestamp       Timestamped transcripts
                          alignment               

  5                       Chunking, BGE-M3, FTS,  Searchable corpus
                          vector indexing         

  6                       Dense retrieval,        Hybrid search API
                          lexical retrieval, RRF, 
                          reranking               

  7                       Search UI and YouTube   Functional product
                          timestamp player        

  8                       Evaluation dataset and  Search quality report
                          retrieval optimization  

  9                       Worker scaling,         Production candidate
                          monitoring, backup,     
                          security                

  10                      Compliance review, load MVP release
                          test, documentation,    
                          release                 
  -----------------------------------------------------------------------

The timeline is an engineering estimate. Creator onboarding, licensing,
and compliance approval can extend the critical path.

------------------------------------------------------------------------

# 20. Development Milestones and Acceptance Criteria

## Milestone 1: Search foundation

Deliver:

-   Dockerized PostgreSQL and pgvector.
-   FastAPI application.
-   Database migrations.
-   Transcript ingestion endpoint.
-   Basic Next.js search page.

Acceptance:

-   A sample authorized transcript can be inserted.
-   Data survives container restart.
-   Database migrations run reproducibly.

## Milestone 2: Semantic retrieval

Deliver:

-   Transcript normalization.
-   Sentence-aware chunking.
-   BGE-M3 embeddings.
-   Vector and lexical indexes.
-   Dense and lexical retrieval.
-   RRF implementation.

Acceptance:

-   A paraphrased query retrieves its labeled segment.
-   Exact technical terms are retrievable.
-   Results include correct source timestamps.

## Milestone 3: Reranking and context

Deliver:

-   Cross-encoder reranker.
-   Context-aware candidate construction.
-   Duplicate and overlapping result aggregation.
-   Search response schema.

Acceptance:

-   Reranking improves or maintains measured ranking metrics.
-   Adjacent chunks do not produce unnecessary duplicate cards.
-   Negation and contrast test cases are evaluated.

## Milestone 4: YouTube integration

Deliver:

-   Official API discovery.
-   Creator authorization flow.
-   Transcript source adapters.
-   Rights checks.
-   Revocation and deletion workflow.

Acceptance:

-   Unauthorized content cannot enter transcript indexing.
-   Authorized content can be processed end-to-end.
-   Revoked content is removed from searchable derived data as required.

## Milestone 5: Production readiness

Deliver:

-   Monitoring and alerting.
-   Backup and recovery.
-   Load testing.
-   Security review.
-   Policy review.
-   Deployment documentation.

Acceptance:

-   Search and ingestion can fail independently.
-   Jobs are retryable and idempotent.
-   Search latency and relevance are measured.
-   Deletion workflows are tested.

------------------------------------------------------------------------

# 21. Repository Structure

Recommended monorepo:

``` text
voxsearch/
│
├── frontend/
│   ├── app/
│   ├── components/
│   ├── hooks/
│   └── lib/
│
├── backend/
│   ├── app/
│   ├── tests/
│   └── requirements.txt
│
├── ml/
│   ├── embeddings/
│   ├── reranker/
│   ├── asr/
│   ├── alignment/
│   ├── evaluation/
│   └── benchmarks/
│
├── workers/
│   ├── discovery/
│   ├── ingestion/
│   ├── indexing/
│   └── compliance/
│
├── database/
│   ├── migrations/
│   ├── schema.sql
│   └── indexes.sql
│
├── infrastructure/
│   ├── docker/
│   ├── monitoring/
│   └── deployment/
│
├── docs/
│   ├── architecture.md
│   ├── api.md
│   ├── database.md
│   ├── compliance.md
│   └── evaluation.md
│
├── docker-compose.yml
├── .env.example
└── README.md
```

------------------------------------------------------------------------

# 22. Final Engineering Decisions

  -----------------------------------------------------------------------
  Area                                Decision
  ----------------------------------- -----------------------------------
  Product                             YouTube semantic spoken-content
                                      search

  Search unit                         Timestamped transcript segment

  Architecture                        Modular monolith + async workers

  Backend                             FastAPI

  Frontend                            Next.js + TypeScript

  Database                            PostgreSQL

  Vector search                       pgvector

  Lexical search                      PostgreSQL FTS

  Embedding                           BGE-M3

  Reranking                           BGE-Reranker-v2-M3

  ASR                                 faster-whisper

  Alignment                           Word-level alignment where
                                      available

  Chunking                            Sentence-aware with contextual
                                      overlap

  Retrieval                           Dense + lexical + RRF +
                                      cross-encoder

  Queue                               Celery + Redis

  Deployment                          Docker

  Initial corpus                      Creator-authorized, licensed, or
                                      user-provided content

  Primary output                      Video segments with exact
                                      timestamps
  -----------------------------------------------------------------------

## What should not be built initially

-   Kafka.
-   Kubernetes.
-   Separate Elasticsearch cluster.
-   Dedicated vector database cluster.
-   Multi-agent orchestration.
-   LLM-generated summaries.
-   Real-time transcription of arbitrary YouTube streams.
-   Multi-platform ingestion.

Add infrastructure only when measured requirements justify it.

------------------------------------------------------------------------

# 23. Immediate Implementation Plan

Start with the Search Engine Core, not broad YouTube crawling.

### Step 1 --- Create a controlled test corpus

Use authorized transcripts or user-provided test data. Start with
100--1,000 videos or transcript samples.

### Step 2 --- Build transcript ingestion

Implement:

-   Transcript schema.
-   Timestamp validation.
-   Sentence segmentation.
-   Chunk creation.
-   Provenance tracking.

### Step 3 --- Implement embeddings and retrieval

Implement:

-   BGE-M3 embedding generation.
-   pgvector indexing.
-   PostgreSQL full-text search.
-   RRF.
-   Cross-encoder reranking.

### Step 4 --- Validate search quality

Create labeled queries and measure Recall@10, MRR@10, nDCG@10, and
timestamp error.

### Step 5 --- Build the timestamped result UI

Connect the search API to a Next.js interface and open YouTube videos at
the returned timestamp.

### Step 6 --- Add compliant YouTube discovery and ingestion

Integrate official metadata discovery, creator authorization, licensed
transcript sources, and rights-aware processing.

------------------------------------------------------------------------

# 24. Final Product Principle

The product's value depends on three independent dimensions:

1.  **Coverage:** How much eligible video content has been indexed?
2.  **Relevance:** Can the engine retrieve the correct spoken
    discussion?
3.  **Precision:** Does it return the correct timestamp and evidence?

A large corpus with poor retrieval is not useful. Excellent embeddings
cannot retrieve a video that is absent from the indexed corpus.

The correct initial objective is therefore not indexing one million
videos. It is building a reliable end-to-end search engine on a
controlled, authorized corpus, proving retrieval quality, and then
expanding coverage through permitted discovery and content partnerships.

**Product definition:**

> A YouTube-first semantic search engine that finds relevant
> spoken-content segments across an expanding indexed video corpus and
> takes users directly to the moment that matters.
