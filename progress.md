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