Phase 1 — Project Foundation
GOAL 01–04 · আনুমানিক ২–৩ দিন
Goal	কাজ	Achieve condition
01	GitHub repository তৈরি	Repository ready
02	Backend FastAPI setup	/health কাজ করে
03	Frontend Next.js setup	Homepage চালু
04	Docker setup	এক কমান্ডে services চালু

Phase complete
 যখন frontend, backend ও Docker একসঙ্গে কাজ করবে।
Phase 2 — Database Foundation
GOAL 05–08 · আনুমানিক ২ দিন
Goal	কাজ	Achieve condition
05	PostgreSQL setup	Database connected
06	pgvector extension	Vector type কাজ করে
07	Videos table	Video metadata insert/read
08	Transcript tables	Transcript ও timestamps save/read

Phase completion test: একটি video ও তার transcript database-এ save করে পুনরায় retrieve করতে পারবে।
Phase 3 — Transcript Processing
GOAL 09–13 · আনুমানিক ৩–৪ দিন
Goal	কাজ	Achieve condition
09	Sample transcript input	JSON transcript গ্রহণ
10	Transcript normalization	Clean text
11	Sentence segmentation	Sentence boundaries
12	Timestamp preservation	Correct start/end
13	Chunking algorithm	Context-aware chunks

Phase completion test: একটি ১০ মিনিটের transcript থেকে সঠিক timestamps-সহ searchable chunks তৈরি হবে।
Phase 4 — Embedding Engine
GOAL 14–17 · আনুমানিক ৩ দিন
Goal	কাজ	Achieve condition
14	BGE-M3 install	Model loads
15	Embedding generation	1024-dimensional vector
16	Vector database insert	Embedding saved
17	Cosine similarity search	Similar chunks retrieved

Phase completion test: একটি query দিলে তার কাছাকাছি অর্থের transcript chunk পাওয়া যাবে।
Phase 5 — Semantic Search API
GOAL 18–21 · আনুমানিক ৩ দিন
Goal	কাজ	Achieve condition
18	Search endpoint	POST /search
19	Query embedding	Query vector generated
20	Vector retrieval	Top 10 results
21	Search response schema	Structured JSON

Phase completion test: Postman থেকে query পাঠিয়ে relevant transcript result পাওয়া যাবে।
Phase 6 — Hybrid Search
GOAL 22–25 · আনুমানিক ৩–৪ দিন
Goal	কাজ	Achieve condition
22	PostgreSQL FTS	Exact keyword retrieval
23	Dense + lexical retrieval	দুই ধরনের result
24	RRF implementation	Combined ranking
25	Duplicate removal	Clean result list

Phase completion test: Exact technical terms এবং semantic paraphrase—দুটোই retrieve করতে পারবে।
Phase 7 — Reranking Engine
GOAL 26–28 · আনুমানিক ৩ দিন
Goal	কাজ	Achieve condition
26	BGE reranker setup	Model loads
27	Candidate reranking	Reordered results
28	Context-aware matching	Surrounding context included

Phase completion test: Reranking-এর আগে ও পরে Recall/MRR/nDCG পরীক্ষা করে ranking quality উন্নত হয়েছে প্রমাণ করতে হবে।
Phase 8 — Search Frontend
GOAL 29–33 · আনুমানিক ৪ দিন
Goal	কাজ	Achieve condition
29	Search homepage	Search UI
30	API integration	Live results
31	Result cards	Video + transcript
32	Timestamp links	Correct YouTube time
33	Search filters	Language ও duration filter

Phase completion test: Browser থেকে query দিয়ে timestamped results দেখতে পারবে।
Phase 9 — YouTube Integration
GOAL 34–38 · আনুমানিক ৪–৫ দিন
Goal	কাজ	Achieve condition
34	YouTube API setup	API authentication
35	Video discovery	Metadata retrieval
36	Channel registry	Channel records
37	Creator authorization	OAuth workflow
38	Transcript source adapter	Authorized transcript ingestion

Phase completion test: Authorized source থেকে transcript এনে স্বয়ংক্রিয়ভাবে indexing pipeline-এ পাঠাতে পারবে।
Phase 10 — Background Processing
GOAL 39–42 · আনুমানিক ৩–৪ দিন
Goal	কাজ	Achieve condition
39	Redis setup	Queue operational
40	Celery worker	Background job
41	Retry mechanism	Failed jobs retry
42	Indexing automation	End-to-end ingestion

Phase completion test: Transcript submission থেকে indexing পর্যন্ত কোনো manual step ছাড়াই কাজ করবে।
Phase 11 — Evaluation & Optimization
GOAL 43–46 · আনুমানিক ৪ দিন
Goal	কাজ	Achieve condition
43	Test query dataset	২০০টি labeled query
44	Retrieval evaluation	Recall@10, MRR, nDCG
45	Timestamp evaluation	Timestamp MAE
46	Performance optimization	P95 latency benchmark

Phase completion test: Search quality report তৈরি হবে এবং প্রতিটি query category-এর performance জানা থাকবে।
Phase 12 — Production Deployment
GOAL 47–50 · আনুমানিক ৪–৫ দিন
Goal	কাজ	Achieve condition
47	Production Docker	Reproducible deployment
48	Monitoring	Logs + metrics
49	Backup ও security	Recovery tested
50	Compliance review ও launch	Public MVP ready

## Goal status

Goal 12 — Timestamp preservation: ACCEPTED.
Goal 13 — Deterministic sentence-aware chunking: ACCEPTED.
Goal 14 — BGE-M3 installation & environment validation: ACCEPTED.
Goal 15 — Embedding generation (1024-dimensional BGE-M3 dense embeddings): ACCEPTED.
Goal 16 — Vector database insert (PostgreSQL + pgvector persistence for 1024-dimensional BGE-M3 embeddings): ACCEPTED. Live PostgreSQL/pgvector runtime round-trip verification remains pending because no local PostgreSQL/Docker runtime was available. Offline validation, compilation, mocked transaction testing and regression testing passed.
Goal 17 — Cosine similarity search (pgvector `<=>` cosine distance retrieval with strict query vector validation): ACCEPTED. Read-only top-k retrieval over persisted 1024-dimensional BGE-M3 embeddings using parameterized SQLAlchemy queries, pgvector `<=>` cosine distance and `CAST(... AS vector(1024))`; optional `video_id` / `transcript_asset_id` / `model_id` filters applied before ranking and limit; deterministic `ORDER BY distance, chunk_index, id`; repository- and service-level top-k bounds (1–100); strict query-vector validation (dimension, finite values, unit L2 norm, no silent rescaling); filter validation and a clear error taxonomy. Focused tests: 96 passed (35 repository + 61 service); full backend suite: 1061 passed, 4 skipped. Live PostgreSQL/pgvector runtime verification remains pending because no local PostgreSQL/Docker runtime was available; the opt-in live test is skipped. No vector index, no search endpoint (Goal 18) and no migration.

Goal 18 — Search endpoint (`POST /search`): ACCEPTED. HTTP transport layer exposing the Goal 17 retrieval service as `POST /api/v1/search` (plus an unversioned `POST /search` alias hidden from OpenAPI). Strict JSON request contract (precomputed 1024-dimensional query vector, bounded `top_k`, optional UUID/string filters), dependency-injected read-only session and service, domain errors translated to HTTP 422, malformed input translated to HTTP 422 without echoing the submitted query vector, missing database configuration translated to HTTP 503, and a minimal bounded JSON response envelope that never includes raw embeddings. Final review fixed one genuine defect: Pydantic validation errors echoed the entire 1024-float query vector in the 422 body; a sanitizing handler now strips `input`/`ctx` from validation errors. Focused endpoint tests: 32 passed; Goal 17 + Goal 18 focused: 128 passed; Goal 15/16 regressions: 205 passed, 1 skipped; full backend suite: 1093 passed, 4 skipped. Live PostgreSQL/pgvector runtime verification remains pending (no local instance available).

Goal 19 — Query embedding (query vector generated from natural-language text): ACCEPTED. `app/embeddings/query.py` adds `QueryEmbeddingService`, a thin Goal 15 orchestration layer that generates ONE query vector via `generate_embeddings` (exactly 1024 finite, unit-norm floats; invalid input/output rejected, never silently rescaled). Inference stays in the isolated `embedding_worker` subprocess using `VOXSEARCH_EMBEDDING_PYTHON`; no torch / BGE-M3 model is loaded in the core API process, no `.env` file is read, and no credentials are forwarded. The Goal 18 endpoint now accepts either `query` (Goal 19) or `query_vector` (Goal 18), exactly one; invalid query input maps to HTTP 422, unavailable worker / inference failures to HTTP 503, and raw embeddings are never returned. Final review fixed one genuine defect: a falsy injected provider was silently discarded in favour of the subprocess fallback; injection now uses an explicit `is not None` check. Focused tests: 19 query-embedding + 40 endpoint = 59 passed; combined search-focused: 155 passed; Goal 15/16 regressions: 205 passed, 1 skipped; full backend suite: 1120 passed, 4 skipped. Real BGE-M3 inference is NOT VERIFIED (no configured `VOXSEARCH_EMBEDDING_PYTHON` and no local Hugging Face model cache); only controlled subprocess doubles were exercised. Live PostgreSQL/pgvector verification also remains pending. Goal 20 and Goal 21 are NOT implemented.
