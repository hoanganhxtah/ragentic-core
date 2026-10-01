"""Retrieval boundary for ``app/``.

``rag_service`` là nguồn retrieval DUY NHẤT (xem Phase 4 trong lịch sử tách
service). Mọi code cần retrieval import từ package này, không import
``app.retrieval.remote`` trực tiếp — giữ một điểm duy nhất để đổi implementation
nếu sau này cần.
"""

from app.retrieval.base import (
    NO_THRESHOLD,
    RetrievalResult,
    RetrievedDoc,
    Retriever,
)
from app.retrieval.factory import build_retriever

__all__ = [
    "NO_THRESHOLD",
    "RetrievalResult",
    "RetrievedDoc",
    "Retriever",
    "build_retriever",
]
