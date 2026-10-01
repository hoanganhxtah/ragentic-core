"""HTTP retriever — gọi ``rag_service`` qua REST.

Đây là implementation DUY NHẤT của ``Retriever`` trong ``app/``. ``app/`` không
mở vector store hay load embedding model nào — toàn bộ nằm ở ``rag_service``.

Nguyên tắc xử lý lỗi: lỗi hạ tầng KHÔNG được raise ra ngoài. Retrieval nằm trong
vòng lặp của agent — nếu knowledge base tạm không truy cập được, agent phải nhận
một kết quả rỗng có ngữ cảnh rõ ràng để nó còn có thể trả lời bằng đường khác
(web search) hoặc nói thật là không tra được. Một exception ở đây sẽ làm sập cả
lượt hội thoại. Cách này giống hệt fallback của ``banking_mcp_client``.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

import httpx

from app.config.app_settings import rag_client_settings
from app.retrieval.base import (
    NO_THRESHOLD,
    RetrievalResult,
    RetrievedDoc,
)

_log = logging.getLogger(__name__)


class RemoteRetriever:
    """``Retriever`` gọi HTTP tới ``rag_service``."""

    def __init__(
        self,
        retrieve_url: Optional[str] = None,
        stats_url: Optional[str] = None,
        timeout: Optional[float] = None,
        max_retries: Optional[int] = None,
    ):
        self._retrieve_url = retrieve_url or rag_client_settings.retrieve_url
        self._stats_url = stats_url or rag_client_settings.stats_url
        self._timeout = timeout if timeout is not None else rag_client_settings.TIMEOUT
        self._max_retries = (
            max_retries if max_retries is not None else rag_client_settings.MAX_RETRIES
        )
        # Một client dùng chung để tái sử dụng connection pool — tạo client mới
        # mỗi request sẽ phải bắt tay TCP lại từ đầu cho mọi lượt retrieve.
        #
        # trust_env=False là điểm quan trọng: đây là lời gọi service-to-service
        # nội bộ, KHÔNG được đi qua proxy. Trên máy có system proxy (rất phổ
        # biến trong mạng doanh nghiệp), httpx áp proxy cho cả 127.0.0.1 và
        # proxy trả 403 — service hoàn toàn bình thường nhưng mọi lượt tra cứu
        # đều thất bại. Tắt trust_env cũng đồng thời bỏ qua NO_PROXY, nên không
        # phụ thuộc vào việc từng máy có cấu hình NO_PROXY đúng hay không.
        self._client = httpx.Client(timeout=self._timeout, trust_env=False)
        _log.info(
            "RemoteRetriever ready (url=%s | timeout=%.1fs | retries=%d)",
            self._retrieve_url,
            self._timeout,
            self._max_retries,
        )

    def _post_retrieve(self, payload: dict) -> dict:
        """POST kèm retry. Raise nếu hết số lần thử."""
        last_error: Optional[Exception] = None
        for attempt in range(self._max_retries + 1):
            try:
                response = self._client.post(self._retrieve_url, json=payload)
                response.raise_for_status()
                return response.json()
            except httpx.HTTPStatusError as exc:
                # 4xx là lỗi của chính request (query rỗng, tham số sai) —
                # retry sẽ cho ra đúng kết quả đó. Chỉ retry 5xx.
                if exc.response.status_code < 500:
                    raise
                last_error = exc
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_error = exc

            if attempt < self._max_retries:
                _log.warning(
                    "RemoteRetriever: lần thử %d thất bại (%s) — thử lại",
                    attempt + 1,
                    last_error,
                )
        raise last_error  # type: ignore[misc]

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
    ) -> RetrievalResult:
        payload: dict[str, Any] = {"query": query}
        if top_k is not None:
            payload["top_k"] = top_k
        if score_threshold is not None:
            # NO_THRESHOLD là -inf, không serialize được sang JSON. Service coi
            # số âm là "không lọc" nên gửi 0.0 cho cùng ý nghĩa.
            payload["score_threshold"] = (
                0.0 if score_threshold == NO_THRESHOLD else score_threshold
            )

        start = time.perf_counter()
        try:
            body = self._post_retrieve(payload)
        except Exception as exc:  # noqa: BLE001
            took_ms = round((time.perf_counter() - start) * 1000, 2)
            _log.warning(
                "RemoteRetriever: rag_service không truy cập được tại %s (%s) — "
                "trả kết quả rỗng để agent không bị vỡ.",
                self._retrieve_url,
                exc,
            )
            return RetrievalResult(
                query=query,
                documents=[],
                raw_count=0,
                best_score=None,
                threshold=score_threshold if score_threshold is not None else 0.0,
                took_ms=took_ms,
            )

        documents = [
            RetrievedDoc(
                content=item.get("content", ""),
                metadata=item.get("metadata") or {},
                score=item.get("score"),
            )
            for item in body.get("results", [])
        ]
        return RetrievalResult(
            query=body.get("query", query),
            documents=documents,
            raw_count=body.get("raw_count", len(documents)),
            best_score=body.get("best_score"),
            threshold=body.get("threshold", 0.0),
            took_ms=body.get("took_ms", round((time.perf_counter() - start) * 1000, 2)),
        )

    def stats(self) -> dict[str, Any]:
        """Lấy stats từ service. Trả về dict có ``error`` nếu không gọi được."""
        try:
            response = self._client.get(self._stats_url)
            response.raise_for_status()
            return {**response.json(), "mode": "remote"}
        except Exception as exc:  # noqa: BLE001
            _log.warning("RemoteRetriever: không lấy được stats: %s", exc)
            return {
                "mode": "remote",
                "error": str(exc),
                "service_url": self._stats_url,
            }

    def close(self) -> None:
        """Đóng HTTP client. Gọi khi shutdown."""
        self._client.close()
