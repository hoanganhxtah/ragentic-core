"""MCP Client wrapper — kết nối Banking MCP Server và load tools.

Dùng ``langchain-mcp-adapters`` để biến MCP tools thành LangChain BaseTool.

Graceful fallback: cả hai hàm đều trả ``[]`` nếu server offline — không raise.

NOTE về event loop:
- ``load_banking_tools()``      — async, dùng từ FastAPI lifespan
- ``load_banking_tools_sync()`` — sync, dùng ngoài event loop (script, pytest)
"""

from __future__ import annotations

import asyncio
import logging

import httpx
from langchain_core.tools import BaseTool

_log = logging.getLogger(__name__)


def _no_proxy_client_factory(
    headers: dict[str, str] | None = None,
    timeout: httpx.Timeout | None = None,
    auth: httpx.Auth | None = None,
) -> httpx.AsyncClient:
    """Tạo httpx client KHÔNG đi qua system proxy.

    MCP server là service nội bộ (localhost hoặc mạng nội bộ), không được đi qua
    proxy. Trên máy có system proxy — rất phổ biến trong mạng doanh nghiệp —
    httpx áp proxy cho cả ``127.0.0.1`` và proxy trả 403. Kết quả: MCP server
    chạy hoàn toàn bình thường nhưng agent vẫn im lặng mất toàn bộ banking tools,
    chỉ có một dòng warning "MCP server not available".

    ``trust_env=False`` cũng bỏ qua ``NO_PROXY``, nên không phụ thuộc vào việc
    từng máy có cấu hình ``NO_PROXY`` đúng hay không.
    """
    return httpx.AsyncClient(
        headers=headers,
        timeout=timeout,
        auth=auth,
        trust_env=False,
        follow_redirects=True,
    )


async def load_banking_tools(server_url: str) -> list[BaseTool]:
    """Async: kết nối MCP server, load tools. Trả [] nếu lỗi."""
    try:
        from langchain_mcp_adapters.client import MultiServerMCPClient

        client = MultiServerMCPClient(
            {
                "banking": {
                    "url": server_url,
                    "transport": "streamable_http",
                    "httpx_client_factory": _no_proxy_client_factory,
                }
            }
        )
        tools: list[BaseTool] = await client.get_tools()
        _log.info(
            "[banking_mcp_client] Loaded %d tools from %s | %s",
            len(tools),
            server_url,
            [t.name for t in tools],
        )
        return tools
    except Exception as exc:  # noqa: BLE001
        _log.warning(
            "[banking_mcp_client] MCP server not available at %s: %s. "
            "rag_agent will run without banking tools.",
            server_url,
            exc,
        )
        return []


# def load_banking_tools_sync(server_url: str) -> list[BaseTool]:
#     """Sync wrapper — chỉ dùng ngoài event loop (script, pytest).

#     Từ FastAPI lifespan dùng ``await load_banking_tools()`` thay thế.
#     """
#     try:
#         return asyncio.run(load_banking_tools(server_url))
#     except RuntimeError as exc:
#         _log.warning(
#             "[banking_mcp_client] asyncio.run() failed (%s). "
#             "Use 'await load_banking_tools()' from async context. Returning [].",
#             exc,
#         )
#         return []
