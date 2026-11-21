"""Isolated BGE-M3 embedding-worker process (Goal 15) — OPTIONAL provider boundary.

This package is the ONLY place that touches FlagEmbedding / PyTorch. It is
executed as a SEPARATE process via ``python -m embedding_worker.worker`` and is
NEVER imported by the core API runtime: importing :mod:`app` /
:mod:`app.transcripts` / :mod:`app.embeddings` never loads ``torch`` or
``FlagEmbedding``.

Modules
-------
* :mod:`embedding_worker.worker` — the dense BGE-M3 adapter. Reads ONE JSON
  request from stdin, writes ONE JSON result to stdout. Heavy imports are lazy
  (inside ``_load_model``), so even importing this module for introspection does
  not load the model stack. It is stdlib-only + a lazy ``FlagEmbedding`` import
  and deliberately has NO :mod:`app` import, so it runs in the OPTIONAL isolated
  ``backend/.venv-embedding`` interpreter without the core API dependencies.
* :mod:`embedding_worker.client` — :class:`SubprocessEmbeddingProvider`, a
  torch-free :class:`app.embeddings.contracts.EmbeddingProvider` that launches the
  worker with an explicit argv, no shell, bounded stdin/stdout, a timeout, and a
  MINIMAL controlled environment (no DB credentials, HF offline + CPU thread
  caps). It maps the worker's closed error reasons to the typed
  :class:`~app.embeddings.contracts.EmbeddingError` subclasses.

The two are deliberately split: the client can be imported by the core backend
without pulling in torch; only the subprocess (with the OPTIONAL
``requirements-embedding.txt`` installed in an isolated interpreter) needs the
model stack. The model is loaded ONCE per call, never once per chunk.

Goal 15 is embedding generation ONLY. Vector persistence (Goal 16), cosine
similarity search (Goal 17), and the search API (Goal 18) are later goals.
"""
