import logging
from fastapi import APIRouter, HTTPException
from app.schemas.chat import SearchRequest, SearchResponse, WebSearchRequest, WebSearchResponse
from app.agent.tools.web_search_tool import web_search

_log = logging.getLogger(__name__)
router = APIRouter()

# Đặt bởi lifespan trong main.py — là một app.retrieval.Retriever.
retriever = None


@router.get("/")
def read_root():
    return {"status": "Ez RAG API is running"}


@router.post("/search/vector", response_model=SearchResponse)
def search_vector_endpoint(request: SearchRequest):
    """Search endpoint to query the vector database directly.

    Endpoint này KHÔNG áp ``SCORE_THRESHOLD`` — nó trả về top-k thô để phục vụ
    debug và tuning threshold. Phần lọc là việc của ``search_vectordb`` tool.
    Giữ nguyên hành vi của bản trước khi tách retrieval seam.
    """
    _log.info("Vector search request | query=%r | n_results=%s", request.query, request.n_results)
    try:
        from app.retrieval.base import NO_THRESHOLD

        result = retriever.retrieve(
            request.query,
            top_k=request.n_results,
            score_threshold=NO_THRESHOLD,
        )
        # Giữ shape lồng [[...]] của response cũ để UI/client không phải đổi.
        return SearchResponse(
            documents=[[d.content for d in result.documents]],
            metadatas=[[d.metadata for d in result.documents]],
            scores=[[d.score for d in result.documents if d.score is not None]],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search/web", response_model=WebSearchResponse)
def search_web_endpoint(request: WebSearchRequest):
    """Search the public web via Tavily and return raw results."""
    _log.info("Web search request | query=%r", request.query)
    try:
        result = web_search.invoke(input=request.query)
        return WebSearchResponse(query=request.query, result=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
