"""Supervisor agent factory.

The supervisor is built with the SAME ``create_agent`` constructor as the
``rag_agent`` / ``web_agent`` sub-agents, so all three share one format. Its
only tools are the handoffs (``handoff_to_rag_agent`` / ``handoff_to_web_agent``):

- When the supervisor wants to consult a source, it calls the matching handoff
  tool, which routes to that sub-agent in the parent graph (see
  ``app/agent/tools/handoff.py``).
- When it has enough information, it simply writes the final answer as a plain
  message (no tool call). The graph then ends — there is no separate finalize
  node.

The supervisor reads the conversation history (agent names + ``[STATUS: ...]``
markers produced by sub-agents) to decide whether to consult another source or
finish, following ``SUPERVISOR_PROMPT``.
"""

from __future__ import annotations

from langchain.agents import create_agent

from app.llm.llm_factory import get_chat_model
from app.agent.prompts import SUPERVISOR_PROMPT


def create_supervisor(tools):
    """Build the supervisor compiled graph bound to its handoff tools."""
    return create_agent(
        model=get_chat_model(),
        tools=tools,
        system_prompt=SUPERVISOR_PROMPT,
        name="supervisor",
    )
