"""VoxSearch evaluation layer (Phase 11)."""
from app.evaluation.performance import (
    LatencyBenchmark,
    LatencyReport,
    PerformanceEvaluationError,
    percentile,
)
from app.evaluation.timestamps import (
    JudgmentError,
    TimestampEvaluation,
    TimestampEvaluationError,
    TimestampJudgment,
    TimestampPrediction,
    evaluate_timestamps,
)
from app.evaluation.retrieval import (
    EvaluationReport,
    QueryMetrics,
    RetrievalEvaluationError,
    evaluate_retrieval,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)
from app.evaluation.dataset import (
    DEFAULT_DATASET_SIZE,
    EvaluationDataset,
    LabeledQuery,
    QueryCategory,
    build_default_dataset,
    validate_dataset,
)

__all__ = [
    "LatencyBenchmark",
    "LatencyReport",
    "PerformanceEvaluationError",
    "percentile",
    "JudgmentError",
    "TimestampEvaluation",
    "TimestampEvaluationError",
    "TimestampJudgment",
    "TimestampPrediction",
    "evaluate_timestamps",
    "EvaluationReport",
    "QueryMetrics",
    "RetrievalEvaluationError",
    "evaluate_retrieval",
    "ndcg_at_k",
    "recall_at_k",
    "reciprocal_rank",
    "DEFAULT_DATASET_SIZE",
    "EvaluationDataset",
    "LabeledQuery",
    "QueryCategory",
    "build_default_dataset",
    "validate_dataset",
]
