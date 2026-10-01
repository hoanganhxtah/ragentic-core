"""Handoff tools for the supervisor.

The supervisor is a ``create_agent`` whose ONLY tools are these handoffs. When
the supervisor decides to consult a sub-agent, it calls the matching tool, which
returns a ``Command`` that jumps to the sibling node in the PARENT graph (the
supervisor graph in ``app/agent/graph.py``).

Two subtleties make this work correctly:

1. ``graph=Command.PARENT`` — the supervisor runs as a subgraph node, so the
   goto must target the parent graph to reach the ``rag_agent`` / ``web_agent``
   nodes. A plain ``goto`` would only look inside the supervisor's own subgraph
   and loop forever.
2. We re-emit ``state["messages"]`` (read via ``InjectedState``) plus the new
   ``ToolMessage``. If we returned only the ToolMessage, the supervisor's
   ``AIMessage(tool_calls=...)`` would be dropped from the shared history,
   leaving an orphan tool result that breaks Anthropic/Bedrock on the next turn.
"""

from __future__ import annotations

from typing import Annotated

from langchain_core.messages import ToolMessage
from langchain_core.tools import tool, InjectedToolCallId
from langgraph.prebuilt import InjectedState
from langgraph.types import Command


def _handoff(target: str, tool_call_id: str, state: dict) -> Command:
    """Build a Command that routes to ``target`` in the parent graph."""
    tool_msg = ToolMessage(
        content=f"Handing off to {target}.",
        name=f"handoff_to_{target}",
        tool_call_id=tool_call_id,
    )
    return Command(
        goto=target,
        graph=Command.PARENT,
        update={"messages": state["messages"] + [tool_msg]},
    )


@tool("handoff_to_rag_agent")
def handoff_to_rag_agent(
    tool_call_id: Annotated[str, InjectedToolCallId],
    state: Annotated[dict, InjectedState],
) -> Command:
    """Hand off to the RAG agent to look up the INTERNAL knowledge base
    (company documents, policies, products, internal procedures)."""
    return _handoff("rag_agent", tool_call_id, state)


@tool("handoff_to_web_agent")
def handoff_to_web_agent(
    tool_call_id: Annotated[str, InjectedToolCallId],
    state: Annotated[dict, InjectedState],
) -> Command:
    """Hand off to the Web agent to search the PUBLIC web for external,
    up-to-date, or general information not in the internal knowledge base."""
    return _handoff("web_agent", tool_call_id, state)
