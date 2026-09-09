"""Backward-compatible FastAPI module.

Prefer: uvicorn Model_PichiaCLM.interfaces.api:app
"""

from .interfaces.api import (
    AnalyzeCdsBatchRequest,
    AnalyzeCdsBatchResponse,
    AnalyzeCdsRequest,
    AnalyzeCdsResponse,
    CdsRecord,
    PredictBatchRequest,
    PredictBatchResponse,
    PredictRequest,
    PredictResponse,
    SequenceRecord,
    app,
    analyze_cds_batch,
    analyze_cds_endpoint,
    get_predictor,
    health,
    predict,
    predict_batch,
)

__all__ = [
    "AnalyzeCdsBatchRequest",
    "AnalyzeCdsBatchResponse",
    "AnalyzeCdsRequest",
    "AnalyzeCdsResponse",
    "CdsRecord",
    "PredictBatchRequest",
    "PredictBatchResponse",
    "PredictRequest",
    "PredictResponse",
    "SequenceRecord",
    "analyze_cds_batch",
    "analyze_cds_endpoint",
    "app",
    "get_predictor",
    "health",
    "predict",
    "predict_batch",
]
