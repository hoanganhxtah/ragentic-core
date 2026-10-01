"""Explicit LangGraph graph state for the supervisor multi-agent system.

``AgentState`` is the state shared between the supervisor node and the sub-agent
nodes. It is a ``TypedDict`` (LangGraph internal state, NOT a Pydantic model).

This is distinct from the API-layer Pydantic schemas in ``app/schemas/agent.py``
(``AgentRequest`` / ``AgentResponse`` / ``AgentStep``), which define the HTTP
request/response contract at the FastAPI boundary.

Routing is driven entirely by the supervisor's handoff *tool calls* (see
``app/agent/tools/handoff.py``), so the only extra field beyond ``messages`` is
``agent_hops`` — a handoff counter used to bound the loop so the graph always
terminates (see ``MAX_HOPS`` in ``app/agent/graph.py``).

Additional structured state fields (``last_grade``, ``retry_count``,
``rewritten_query``) support the independent Validator node that evaluates
sub-agent results and drives deterministic routing without text-marker parsing.
"""

from typing import Annotated, Literal, Optional, TypedDict

from langgraph.graph.message import add_messages
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Grade type — result of Validator evaluation
# ---------------------------------------------------------------------------

Grade = Literal["PASS", "FAIL", "AMBIGUOUS"]

# ---------------------------------------------------------------------------
# RouteDecision and ValidatorOutput — used by routing logic and validator node
# ---------------------------------------------------------------------------

RouteDecision = Literal["proceed", "retry", "synthesize_best_effort"]


class ValidatorOutput(BaseModel):
    """Structured output produced by the Validator node."""

    grade: Grade
    rewritten_query: Optional[str] = None
    reason: str = ""


# ---------------------------------------------------------------------------
# AgentState — shared LangGraph state TypedDict
# ---------------------------------------------------------------------------


class AgentState(TypedDict):
    # Conversation messages — the reducer appends new messages each step.
    # This is the SHARED channel between the supervisor and sub-agent nodes.
    messages: Annotated[list, add_messages]

    # Number of supervisor -> sub-agent handoffs so far. Used to bound the loop
    # (see MAX_HOPS in app/agent/graph.py) so the graph always terminates.
    agent_hops: int

    # --- Validator state fields (Requirements 8.1-8.3) ---

    # Grade assigned by the DBt recent Validator evaluation.
    # None before the first validation in a turn.
    last_grade: Optional[Grade]

    # Number of retry attempts consumed in the current turn.
    # Incremented by the Validator node only when routing decision is "retry".
    retry_count: int

    # Advisory rewritten query proposed by Validator on FAIL/AMBIGUOUS grade.
    # None when last_grade is PASS or before first validation.
    rewritten_query: Optional[str]
