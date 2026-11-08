"""Isolated alignment-worker process (Goal 12) — OPTIONAL, provider boundary.

This package is the ONLY place that touches WhisperX / PyTorch. It is executed
as a SEPARATE process via ``python -m alignment_worker.worker`` and is NEVER
imported by the core API runtime: importing :mod:`app` / :mod:`app.transcripts`
/ :mod:`app.services` / :mod:`app.repositories` never loads ``torch`` or
``whisperx``.

Modules
-------
* :mod:`alignment_worker.worker` — the WhisperX 3.8.6 adapter. Reads ONE JSON
  request from stdin, writes ONE JSON result to stdout. Heavy imports are lazy
  (inside ``main()``), so even importing this module for introspection does not
  load the model stack.
* :mod:`alignment_worker.client` — :class:`SubprocessAlignmentProvider`, a
  torch-free callable that launches the worker with an explicit argv, no shell,
  bounded stdin/stdout, a timeout, and a MINIMAL controlled environment (no DB
  credentials). It satisfies the service's ``ProviderCallable`` protocol and
  normalizes the worker's RAW output via the pure
  :func:`app.transcripts.word_alignment.reconstruct_word_chars` (one shared
  implementation; the worker process itself needs no :mod:`app` import).

The two are deliberately split: the client can be imported by an operator /
scheduler without pulling in torch; only the subprocess (with the OPTIONAL
``requirements-alignment.txt`` installed in an isolated interpreter) needs the
model stack. Live provider execution is an OUTSTANDING verification step: this
goal does NOT install WhisperX, download model weights, or fetch real audio.
"""
