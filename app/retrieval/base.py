"""Retrieval seam — the single boundary between the agent and the RAG backend.

Đây là ranh giới HTTP của ``rag_service``. Mọi thứ trong ``app/`` chỉ được nói
chuyện với retrieval qua ``Retriever`` protocol, không bao giờ import vector
store hay embedding model trực tiếp — những thứ đó thuộc về ``rag_service``,
chạy như một service riêng (xem ``rag_service/README.md``).

Phân chia trách nhiệm — quan trọng khi tách service:

- ``Retriever``  trả lời "documents nào liên quan" (retrieval concern).
- Caller quyết định "nói với LLM thế nào" (prompt concern, xem
  ``app/agent/tools/vectordb_tool.py``).

Vì vậy ``RetrievalResult`` cố tình mang cả metadata chẩn đoán (``raw_count``,
``best_score``) — caller cần chúng để dựng thông báo "không tìm thấy" và để log,
mà không phải tự chạy lại phần lọc.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, runtime_checkable

# Sentinel cho ``score_threshold``: giữ lại MỌI hit, không lọc.
# Dùng -inf thay vì 0.0 vì relevance score của LangChain về lý thuyết có thể âm,
# nên 0.0 không thực sự nghĩa là "không lọc".
NO_THRESHOLD: float = float("-inf")


@dataclass(frozen=True)
class RetrievedDoc:
    """Một document trả về từ vector store, đã phẳng hoá.

    Attributes
    ----------
    content:
        Nội dung chunk.
    metadata:
        Metadata của chunk (``source``, ``file_name``, ``chunk_index``, ``id``...).
    score:
        Relevance score 0..1 (cosine). ``None`` khi backend không cung cấp —
        khi đó hit KHÔNG bị lọc bởi threshold vì không có cơ sở để đánh giá.
    """

    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    score: Optional[float] = None


@dataclass(frozen=True)
class RetrievalResult:
    """Kết quả một lượt retrieve, đã áp threshold.

    Attributes
    ----------
    query:
        Query đã dùng để tìm.
    documents:
        Các hit ĐÃ vượt threshold. Có thể rỗng.
    raw_count:
        Số hit vector store trả về TRƯỚC khi lọc. Cần để phân biệt hai trường
        hợp rất khác nhau: "knowledge base không có gì" (raw_count=0) và
        "có nhưng không đủ liên quan" (raw_count>0, documents rỗng).
    best_score:
        Score cao nhất trong tập raw. ``None`` khi không có hit nào có score.
        Dùng để log tại sao mọi thứ bị lọc.
    threshold:
        Threshold đã áp dụng thực tế.
    took_ms:
        Thời gian retrieve, tính bằng ms.
    """

    query: str
    documents: list[RetrievedDoc]
    raw_count: int
    best_score: Optional[float]
    threshold: float
    took_ms: float

    @property
    def is_empty(self) -> bool:
        return not self.documents


@runtime_checkable
class Retriever(Protocol):
    """Hợp đồng retrieval. Implementation duy nhất hiện tại:

    - ``RemoteRetriever`` — HTTP tới ``rag_service`` (xem ``app/retrieval/remote.py``)

    Implementation phải KHÔNG raise ra ngoài đối với lỗi hạ tầng có thể lường
    trước (service offline, network timeout). Agent loop không được vỡ vì
    knowledge base tạm không truy cập được — xem cách ``vectordb_tool`` xử lý.
    """

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
    ) -> RetrievalResult:
        """Tìm document liên quan tới ``query``.

        Parameters
        ----------
        query:
            Câu truy vấn.
        top_k:
            Số hit lấy ra. ``None`` → dùng mặc định từ config.
        score_threshold:
            Ngưỡng lọc. ``None`` → dùng mặc định từ config.
            ``NO_THRESHOLD`` → không lọc gì.
        """
        ...

    def stats(self) -> dict[str, Any]:
        """Thông tin chẩn đoán về backend đang dùng.

        Tối thiểu nên có: ``backend``, ``collection``, ``doc_count``,
        ``embedding_model``. ``embedding_model`` đặc biệt quan trọng — ingest và
        query dùng model khác nhau sẽ cho kết quả tệ mà KHÔNG báo lỗi, nên phải
        quan sát được từ bên ngoài.
        """
        ...
