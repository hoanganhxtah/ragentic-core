"""RAG sub-agent factory.

A specialized agent equipped ONLY with the ``search_vectordb`` tool. Built with
LangGraph's ``create_agent`` (same constructor the single-agent engine used) and
wrapped as a graph node in ``app/agent/graph.py``.
"""

from __future__ import annotations

from langchain.agents import create_agent

from app.llm.llm_factory import get_chat_model
from app.agent.prompts import RAG_AGENT_PROMPT


def create_rag_agent(tools):
    """Build the rag_agent compiled graph bound to its (vectordb) tools."""
    return create_agent(
        model=get_chat_model(),
        tools=tools,
        system_prompt=RAG_AGENT_PROMPT,
        name="rag_agent",
    )
