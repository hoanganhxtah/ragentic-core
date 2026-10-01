"""Validator agent factory.

The validator is an independent grader that evaluates the relevance and
completeness of a sub-agent's result against the original question.

It uses the same ``create_agent`` constructor as rag_agent / web_agent for
consistency, but is bound to NO tools — it only reasons over its input
(original question + sub-agent result) and produces a grade + guidance.

Grade output is published as a ``[VALIDATOR]`` HumanMessage so that:
1. The supervisor reads it as guidance (not a final answer).
2. The role alternation (Human→AI) is preserved for Anthropic/Bedrock.
"""

from __future__ import annotations

from langchain.agents import create_agent

from app.llm.llm_factory import get_chat_model
from app.agent.prompts import VALIDATOR_PROMPT


def create_validator_agent():
    """Build the validator compiled graph. No tools — evaluation only."""
    return create_agent(
        model=get_chat_model(),
        tools=[],
        system_prompt=VALIDATOR_PROMPT,
        name="validator",
    )
