"""Knowledge-base search tool for the RAG sub-agent.

``make_search_vectordb(retriever)`` xây tool ``search_vectordb`` gắn với một
``Retriever``. Trước đây tool là hàm top-level dùng biến global ``_vdb`` tiêm qua
``set_vdb()``; giờ dùng factory + closure để bỏ global mutable state — đó chính
là điểm dính khiến việc đổi backend retrieval phải sửa nhiều nơi.

TÊN TOOL VÀ DOCSTRING GIỮ NGUYÊN. Prompt của ``rag_agent`` và các assertion
trong log đều dựa vào tên ``search_vectordb``, còn docstring là thứ LLM đọc để
quyết định có gọi tool hay không — đổi chúng là đổi hành vi model.

Ranh giới trách nhiệm: ``Retriever`` trả documents có cấu trúc; module này biến
chúng thành text cho LLM. Việc format ở lại ``app/`` (không đẩy sang
``rag_service``) để prompt tuning không cần deploy service khác.
"""

from __future__ import annotations

import logging
from typing import Optional

from langchain_core.tools import BaseTool, tool

from app.retrieval.base import RetrievalResult, Retriever

_log = logging.getLogger(__name__)

# Trả về khi không có document nào đủ liên quan. Câu chữ được giữ nguyên như
# bản cũ: nó hướng dẫn LLM đừng bịa câu trả lời từ tài liệu không liên quan.
NO_RELEVANT_DOCS_MESSAGE = (
    "No sufficiently relevant documents found in the internal "
    "knowledge base for this query. The knowledge base does not "
    "appear to contain information on this specific topic. "
    "Do NOT fabricate an answer from unrelated documents."
)


def format_result(result: RetrievalResult) -> str:
    """Biến ``RetrievalResult`` thành text cho LLM đọc.

    Format giữ nguyên bản cũ::

        [Doc 0] (relevance=0.812) source=..., file_name=...
        <content>

        ---

        [Doc 1] ...
    """
    blocks = []
    for i, doc in enumerate(result.documents):
        meta_str = ", ".join(f"{k}={v}" for k, v in (doc.metadata or {}).items())
        score_str = (
            f"relevance={doc.score:.3f}" if doc.score is not None else "relevance=n/a"
        )
        blocks.append(f"[Doc {i}] ({score_str}) {meta_str}\n{doc.content}")
    return "\n\n---\n\n".join(blocks)


def make_search_vectordb(retriever: Retriever) -> BaseTool:
    """Build the ``search_vectordb`` tool bound to ``retriever``."""

    @tool("search_vectordb")
    def search_vectordb(query: str) -> str:
        """Search the internal knowledge base / company documents. Use this for questions about internal/domain-specific information."""
        _log.info("[tool:search_vectordb] called | query=%r", query)
        try:
            result = retriever.retrieve(query)
        except Exception as error:  # never raise into the agent loop
            _log.exception("[tool:search_vectordb] failed | query=%r", query)
            return f"Knowledge base is currently unavailable: {error}"

        if result.is_empty:
            best = result.best_score
            _log.info(
                "[tool:search_vectordb] filtered out all hits | query=%r | "
                "raw=%d | threshold=%.2f | best_score=%s | elapsed=%.2fs",
                query,
                result.raw_count,
                result.threshold,
                f"{best:.3f}" if best is not None else "n/a",
                result.took_ms / 1000,
            )
            return NO_RELEVANT_DOCS_MESSAGE

        _log.info(
            "[tool:search_vectordb] success | query=%r | kept=%d/%d | "
            "threshold=%.2f | elapsed=%.2fs",
            query,
            len(result.documents),
            result.raw_count,
            result.threshold,
            result.took_ms / 1000,
        )
        return format_result(result)

    return search_vectordb
