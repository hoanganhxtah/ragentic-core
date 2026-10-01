"""Web search tool (Tavily) for the ReAct agent.

A single top-level ``@tool`` named ``web_search`` backed by Tavily. The Tavily
client is created once (lazily) from ``web_search_settings`` and reused.

The tool logs every invocation (which tool ran, the query, elapsed time, and
outcome) and retries on transient connection errors (e.g. ``ConnectionReset``
/ "connection forcibly closed"). Any final error is caught and returned as a
readable string so it never raises into the agent loop.

Proxy support: set ``WEB_SEARCH_HTTP_PROXY`` / ``WEB_SEARCH_HTTPS_PROXY`` in
``.env`` (e.g. ``http://127.0.0.1:3128``). They are applied to ``os.environ``
before the Tavily client is constructed so the underlying ``requests`` / httpx
sessions pick them up automatically.
"""

import logging
import os
import time
from typing import Union

from langchain_core.tools import tool
from langchain_tavily import TavilySearch

from app.config.app_settings import web_search_settings, agent_settings

_log = logging.getLogger(__name__)

_tavily = None

# Exception types / messages that indicate a transient network failure worth
# retrying rather than surfacing immediately to the agent.
_TRANSIENT_HINTS = (
    "connection aborted",
    "connection reset",
    "forcibly closed",
    "timed out",
    "timeout",
    "temporarily unavailable",
    "max retries exceeded",
)


def _is_transient(error: Exception) -> bool:
    """Return True when the error looks like a retryable network hiccup."""
    if isinstance(error, (ConnectionError, TimeoutError)):
        return True
    text = str(error).lower()
    return any(hint in text for hint in _TRANSIENT_HINTS)


def _format_tavily_results(raw: Union[dict, list, str], threshold: float) -> str:
    """Parse Tavily response, filter by relevance score, and format for the agent.

    Args:
        raw: Tavily response — either dict ``{"results": [...]}`` or list of
             result dicts. Falls back to ``str(raw)`` if format is unexpected.
        threshold: Minimum score (0..1) to keep a result. Results with no
                   ``score`` field (None) are always kept per Req 4.4.

    Returns:
        Formatted string with one block per kept result, or a "no results"
        message if all were filtered out.
    """
    # Fallback: if raw is already a string (format changed / error path)
    if isinstance(raw, str):
        return raw

    # Normalise to a flat list of result dicts
    if isinstance(raw, list):
        results = raw
    elif isinstance(raw, dict):
        results = raw.get("results") or []
    else:
        return str(raw)

    total = len(results)

    # Filter: keep result if score is missing (None) or >= threshold
    kept = [
        r for r in results
        if r.get("score") is None or r.get("score", 0) >= threshold
    ]

    if not kept:
        scores = [r.get("score") for r in results if r.get("score") is not None]
        best = max(scores) if scores else None
        _log.info(
            "[tool:web_search] all %d results filtered out | threshold=%.3f | best_score=%s",
            total,
            threshold,
            f"{best:.3f}" if best is not None else "n/a",
        )
        return "No sufficiently relevant web results found for this query."

    _log.info(
        "[tool:web_search] relevance filter: kept %d/%d | threshold=%.3f",
        len(kept),
        total,
        threshold,
    )

    blocks = []
    for i, r in enumerate(kept):
        s = r.get("score")
        score_str = f"score={s:.3f}" if s is not None else "score=n/a"
        url = r.get("url", "")
        title = r.get("title", "")
        content = r.get("content", "")
        blocks.append(f"[Web {i}] ({score_str}) {url}\n{title}\n{content}")

    return "\n\n---\n\n".join(blocks)


def _client() -> TavilySearch:
    """Lazily build and cache the Tavily search client.

    Also applies HTTP/HTTPS proxy env vars (if configured) before constructing
    the client so the underlying HTTP session honours them.
    """
    global _tavily
    if _tavily is None:
        if not web_search_settings.TAVILY_API_KEY:
            raise ValueError("TAVILY_API_KEY is required for web search.")

        # Apply proxy settings to os.environ so requests / httpx picks them up.
        # if web_search_settings.HTTP_PROXY:
        #     os.environ.setdefault("HTTP_PROXY", web_search_settings.HTTP_PROXY)
        #     _log.info("[tool:web_search] HTTP_PROXY set to %s", web_search_settings.HTTP_PROXY)
        # if web_search_settings.HTTPS_PROXY:
        #     os.environ.setdefault("HTTPS_PROXY", web_search_settings.HTTPS_PROXY)
        #     _log.info("[tool:web_search] HTTPS_PROXY set to %s", web_search_settings.HTTPS_PROXY)

        _tavily = TavilySearch(
            max_results=web_search_settings.MAX_RESULTS,
            tavily_api_key=web_search_settings.TAVILY_API_KEY,
        )
    return _tavily


@tool
def web_search(query: str) -> str:
    """Search the public web for external/general information not in the internal knowledge base."""
    max_retries = max(0, web_search_settings.MAX_RETRIES)
    backoff = max(0.0, web_search_settings.RETRY_BACKOFF_SECONDS)

    _log.info("[tool:web_search] called | query=%r", query)
    start = time.perf_counter()

    last_error: Exception | None = None
    for attempt in range(1, max_retries + 2):  # 1 initial try + max_retries
        try:
            result = _client().invoke(input=query)
            elapsed = time.perf_counter() - start
            _log.info(
                "[tool:web_search] success | query=%r | attempt=%d | elapsed=%.2fs",
                query,
                attempt,
                elapsed,
            )
            return _format_tavily_results(result, agent_settings.WEB_SCORE_THRESHOLD)
        except Exception as error:  # noqa: BLE001 - classified below
            last_error = error
            transient = _is_transient(error)
            _log.warning(
                "[tool:web_search] attempt %d/%d failed | query=%r | transient=%s | error=%s",
                attempt,
                max_retries + 1,
                query,
                transient,
                error,
            )
            # Stop early on non-transient errors (e.g. bad API key, bad input).
            if not transient or attempt > max_retries:
                break
            time.sleep(backoff * attempt)

    elapsed = time.perf_counter() - start
    _log.error(
        "[tool:web_search] giving up | query=%r | elapsed=%.2fs | error=%s",
        query,
        elapsed,
        last_error,
    )
    return f"Web search failed after {max_retries + 1} attempt(s): {last_error}"
