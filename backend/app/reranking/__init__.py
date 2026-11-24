"""VoxSearch BGE reranker environment (Goal 26 — setup / model loads).

Only the torch-free contract layer is importable from the core backend. The real
model is loaded exclusively in the isolated ``reranker_worker`` subprocess.
Candidate scoring / reordering (Goal 27) is NOT implemented here.
"""
from app.reranking.context import (
    MAX_CONTEXT_CHARACTERS,
    MAX_RADIUS,
    ContextChunk,
    ContextConfig,
    ContextExpansionService,
    ContextProvider,
    ContextAwareRerankingService,
    ContextualCandidate,
    RepositoryContextProvider,
)
from app.reranking.scoring import (
    MAX_CANDIDATES,
    MAX_QUERY_CHARACTERS,
    CandidateRerankingService,
    InvalidRerankInputError,
    RerankScorer,
    RerankedCandidate,
    RerankingError,
)
from app.reranking.contracts import (
    DEFAULT_RERANKER_MODEL_ID,
    DEFAULT_TIMEOUT_SECONDS,
    RERANKER_PYTHON_ENV,
    InvalidRerankerConfigError,
    RerankerConfig,
    RerankerError,
    RerankerModelLoadError,
    RerankerProvider,
    RerankerSetup,
    RerankerStatus,
    RerankerUnavailableError,
)

__all__ = [
    "MAX_CONTEXT_CHARACTERS",
    "MAX_RADIUS",
    "ContextChunk",
    "ContextConfig",
    "ContextExpansionService",
    "ContextProvider",
    "ContextAwareRerankingService",
    "ContextualCandidate",
    "RepositoryContextProvider",
    "MAX_CANDIDATES",
    "MAX_QUERY_CHARACTERS",
    "CandidateRerankingService",
    "InvalidRerankInputError",
    "RerankScorer",
    "RerankedCandidate",
    "RerankingError",
    "DEFAULT_RERANKER_MODEL_ID",
    "DEFAULT_TIMEOUT_SECONDS",
    "RERANKER_PYTHON_ENV",
    "InvalidRerankerConfigError",
    "RerankerConfig",
    "RerankerError",
    "RerankerModelLoadError",
    "RerankerProvider",
    "RerankerSetup",
    "RerankerStatus",
    "RerankerUnavailableError",
]
