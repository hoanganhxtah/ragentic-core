"""Per-agent tool registry for the supervisor multi-agent system.

``build_agent_tools(retriever)``       — sync, không load MCP (dùng khi không có event loop).
``build_agent_tools_async(retriever)`` — async, load MCP tools đúng cách từ lifespan.

Banking MCP tools được load từ ``MCP_SERVER_URL`` (default: http://localhost:8001/mcp).
Nếu MCP server offline, rag_agent tiếp tục với search_vectordb only — không crash.

``retriever`` là một ``app.retrieval.Retriever`` (local hoặc remote). Registry
không biết phía sau là in-process VectorDB hay HTTP call tới ``rag_service``.
"""

import os
from typing import Dict, List

from langchain_core.tools import BaseTool

from app.agent.tools.vectordb_tool import make_search_vectordb
from app.agent.tools.web_search_tool import web_search
from app.agent.tools.handoff import handoff_to_rag_agent, handoff_to_web_agent
from app.agent.tools.banking_mcp_client import load_banking_tools
from app.retrieval.base import Retriever


def build_agent_tools(retriever: Retriever) -> Dict[str, List[BaseTool]]:
    """Sync version — dùng khi KHÔNG có event loop (script, test).

    Không load MCP tools để tránh asyncio conflict.
    Dùng ``build_agent_tools_async()`` từ FastAPI lifespan.
    """
    return {
        "supervisor": [handoff_to_rag_agent, handoff_to_web_agent],
        "rag_agent": [make_search_vectordb(retriever)],
        "web_agent": [web_search],
    }


async def build_agent_tools_async(retriever: Retriever) -> Dict[str, List[BaseTool]]:
    """Async version — dùng từ FastAPI lifespan để load MCP tools đúng cách.

    Await ``load_banking_tools()`` trực tiếp, không dùng asyncio.run().
    Graceful fallback: nếu MCP offline, rag_agent chạy với search_vectordb only.
    """
    mcp_server_url = os.getenv("MCP_SERVER_URL", "http://localhost:8003/mcp")
    banking_tools = await load_banking_tools(mcp_server_url)

    # Datetime tool — cấp cho web_agent để nó resolve được "hôm nay", "ngày mai"
    # trước khi tạo query tìm kiếm. Dùng tên MCP tool thực tế.
    _DATETIME_TOOL_NAMES = {"get_datetime_api_v1_get_datetime_get", "get_datetime"}
    datetime_tools = [t for t in banking_tools if t.name in _DATETIME_TOOL_NAMES]

    return {
        "supervisor": [handoff_to_rag_agent, handoff_to_web_agent],
        "rag_agent": [make_search_vectordb(retriever)] + banking_tools,
        "web_agent": [web_search] + datetime_tools,
    }
