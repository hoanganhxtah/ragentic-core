"""Web sub-agent factory.

A specialized agent equipped ONLY with the ``web_search`` tool. Built with
LangGraph's ``create_agent`` and wrapped as a graph node in ``app/agent/graph.py``.
"""

from __future__ import annotations

from langchain.agents import create_agent

from app.llm.llm_factory import get_chat_model
from app.agent.prompts import WEB_AGENT_PROMPT


def create_web_agent(tools):
    """Build the web_agent compiled graph bound to its (web search) tools."""
    return create_agent(
        model=get_chat_model(),
        tools=tools,
        system_prompt=WEB_AGENT_PROMPT,
        name="web_agent",
    )
