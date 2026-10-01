"""Tạo retriever cho ``app/``.

Sau Phase 4, ``rag_service`` là nguồn retrieval DUY NHẤT — ``app/vectordb/`` và
``app/llm/embeddings/`` đã bị xoá khỏi ``app/``. ``build_retriever()`` luôn trả
``RemoteRetriever``.

Hàm vẫn giữ tên và vị trí cũ (không đổi sang gọi ``RemoteRetriever()`` trực tiếp
ở call site) để nếu sau này cần thêm một implementation khác (ví dụ cache layer,
hoặc retriever thứ hai cho một domain khác), chỉ cần sửa một chỗ.
"""

from __future__ import annotations

import logging

from app.retrieval.base import Retriever

_log = logging.getLogger(__name__)


def build_retriever() -> Retriever:
    """Tạo ``RemoteRetriever`` — HTTP client tới ``rag_service``."""
    from app.retrieval.remote import RemoteRetriever

    _log.info("Retrieval → rag_service (RemoteRetriever)")
    return RemoteRetriever()
