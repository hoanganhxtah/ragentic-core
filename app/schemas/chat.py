from pydantic import BaseModel
from typing import List, Dict, Any, Optional


class SearchRequest(BaseModel):
    query: str
    n_results: int = 5


class SearchResponse(BaseModel):
    documents: List[List[str]]
    metadatas: List[List[Dict[str, Any]]]
    scores: List[List[float]]


class WebSearchRequest(BaseModel):
    query: str


class WebSearchResponse(BaseModel):
    query: str
    result: str


