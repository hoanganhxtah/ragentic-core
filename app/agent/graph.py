"""Supervisor multi-agent graph built with plain LangGraph primitives.

All three agents (supervisor, rag_agent, web_agent) are ``create_agent`` graphs
wrapped as nodes. Routing is driven by the supervisor's handoff tool calls:

    START -> supervisor
    supervisor --(handoff_to_rag_agent)--> rag_agent   (via Command.PARENT)
    supervisor --(handoff_to_web_agent)--> web_agent   (via Command.PARENT)
    supervisor --(no tool call = final answer)--> END
    rag_agent / web_agent --> supervisor   (or END if MAX_HOPS reached)

The supervisor writes the final answer itself (a plain message with no tool
call), so there is no separate finalize node. Sub-agent nodes run on the shared
``messages`` channel and return only the NEW messages they produced so the
``add_messages`` reducer appends them.
"""

from __future__ import annotations

import copy
import logging
from typing import Optional

from langgraph.graph import StateGraph, START, END

from app.agent.state import AgentState, Grade, RouteDecision
from app.agent.agents.rag_agent import create_rag_agent
from app.agent.agents.web_agent import create_web_agent
from app.agent.agents.supervisor import create_supervisor

_log = logging.getLogger(__name__)

MAX_HOPS = 6  # số lần bàn giao tối đa trước khi buộc kết thúc


# ---------------------------------------------------------------------------
# Pure routing functions — no LLM calls, no message parsing (Req 3.3, 6.1-6.4)
# ---------------------------------------------------------------------------


def decide_routing(
    grade: Grade,
    retry_count: int,
    retry_budget: int,
) -> RouteDecision:
    """Deterministic routing decision based solely on structured state fields.

    Rules:
    - PASS                                       → "proceed"
    - (FAIL | AMBIGUOUS) ∧ retry_count < budget  → "retry"
    - (FAIL | AMBIGUOUS) ∧ retry_count ≥ budget  → "synthesize_best_effort"

    This is a pure function — it never reads ``messages`` or calls an LLM.
    """
    if grade == "PASS":
        return "proceed"
    # grade is FAIL or AMBIGUOUS
    if retry_count < retry_budget:
        return "retry"
    return "synthesize_best_effort"


def _normalize_rewritten_query(
    grade: Grade,
    rewritten_query: Optional[str],
    fallback_query: str,
) -> Optional[str]:
    """Normalise the rewritten_query field written into AgentState.

    Rules (Property 4):
    - PASS             → None  (no rewrite needed)
    - FAIL | AMBIGUOUS → rewritten_query if provided, else fallback_query
                         (guarantee: result is never None when grade != PASS)
    """
    if grade == "PASS":
        return None
    return rewritten_query if rewritten_query else fallback_query


def _extract_text(content) -> str:
    """Normalize message content thành string thuần.

    Bedrock Converse trả content dạng list of dicts:
        [{'type': 'text', 'text': '...', 'index': 0}]
    Các provider khác trả string trực tiếp.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    return str(content)


def _make_sub_agent_node(compiled_agent, agent_name: str):
    """Wrap a compiled sub-agent into a graph node.

    The sub-agent runs on the LAST HumanMessage only (avoids reading a previous
    agent's result and skipping its own tool, and avoids mixing old questions
    into the new one). We return only the NEW messages it produced so the shared
    ``add_messages`` reducer appends them, and we bump ``agent_hops``.
    """

    async def node(state: dict) -> dict:
        from langchain_core.messages import HumanMessage

        incoming = state.get("messages", [])

        last_human = None
        for m in reversed(incoming):
            if isinstance(m, HumanMessage):
                last_human = m
                break
        agent_input = [last_human] if last_human else incoming

        result = await compiled_agent.ainvoke({"messages": agent_input})
        produced = result["messages"][len(agent_input):]

        # Lọc bỏ thinking blocks của Bedrock (type="thinking") khỏi messages
        # trước khi thêm vào shared state — chúng chỉ valid trong context
        # riêng của sub-agent và sẽ gây lỗi khi supervisor đọc lại history.
        # LƯU Ý: GIỮ NGUYÊN [STATUS:] markers — supervisor cần chúng để route.
        from langchain_core.messages import AIMessage as _AIMsg
        cleaned = []
        for msg in produced:
            if isinstance(msg, _AIMsg) and isinstance(msg.content, list):
                text_parts = [p for p in msg.content
                              if isinstance(p, dict) and p.get("type") == "text"]
                if text_parts:
                    # Giữ lại chỉ text parts, bỏ thinking/tool_use blocks
                    msg = copy.copy(msg)
                    msg.content = text_parts if len(text_parts) > 1 else text_parts[0].get("text", "")
            cleaned.append(msg)
        produced = cleaned

        # ── Logging: STATUS marker + chi tiết để debug ──────────────────────
        status = "UNKNOWN"
        last_content = ""
        for msg in reversed(produced):
            if getattr(msg, "content", None):
                last_content = _extract_text(msg.content)
                upper = last_content.upper()
                if "[STATUS: COMPLETE]" in upper:
                    status = "COMPLETE"
                elif "[STATUS: PARTIAL]" in upper:
                    status = "PARTIAL"
                elif "[STATUS: NOT_FOUND]" in upper:
                    status = "NOT_FOUND"
                break

        _log.info(
            "[%s] inference done | status=%s | new_msgs=%d | hops=%d | tail=%.150s",
            agent_name, status, len(produced), state.get("agent_hops", 0) + 1,
            last_content[-150:].replace("\n", " "),
        )

        return {"messages": produced, "agent_hops": state.get("agent_hops", 0) + 1}

    return node


def _route_after_sub_agent(state: dict) -> str:
    """Sau khi sub-agent chạy: quay lại supervisor, trừ khi đã chạm MAX_HOPS."""
    if state.get("agent_hops", 0) >= MAX_HOPS:
        _log.warning("MAX_HOPS=%d reached → forcing END", MAX_HOPS)
        return END
    return "supervisor"


# ---------------------------------------------------------------------------
# Validator node — wraps compiled validator agent with input building + parsing
# ---------------------------------------------------------------------------

_VALIDATOR_GUIDANCE_PREFIX = "[VALIDATOR]"
# Regex for parsing the structured guidance line emitted by the validator LLM.
import re as _re
_GUIDANCE_RE = _re.compile(
    r"\[VALIDATOR\]\s*Grade=(?P<grade>PASS|FAIL|AMBIGUOUS)"
    r"(?:\s+rewritten_query=(?P<rq>\S+))?"
    r"(?:\s+reason=(?P<reason>.+))?",
    _re.IGNORECASE,
)


def _parse_validator_guidance(text: str) -> tuple[str, Optional[str], str]:
    """Extract (grade, rewritten_query, reason) from validator LLM output.

    Returns ("AMBIGUOUS", None, "") on any parse failure — safe fallback.
    """
    m = _GUIDANCE_RE.search(text)
    if not m:
        return "AMBIGUOUS", None, ""
    grade = m.group("grade").upper()
    rq = m.group("rq")
    if rq and rq.lower() in ("none", "null", ""):
        rq = None
    reason = (m.group("reason") or "").strip()
    return grade, rq, reason


def _make_validator_node(compiled_validator, retry_budget: int):
    """Wrap the compiled validator agent into a LangGraph node.

    Input building differs from _make_sub_agent_node:
    - rag/web nodes only need the last HumanMessage (the question).
    - The validator needs BOTH the original question AND the sub-agent result
      so it can compare them and produce a grade.

    The node:
    1. Extracts the last real HumanMessage (not a [VALIDATOR] guidance line).
    2. Extracts the DBt recent AIMessage from rag_agent or web_agent.
    3. Calls the compiled validator with [question, sub_agent_result].
    4. Parses the [VALIDATOR] guidance line from the output.
    5. Calls decide_routing + _normalize_rewritten_query (pure functions).
    6. Increments retry_count only when decision == "retry".
    7. Adds a budget-exhausted hint to the guidance when synthesize_best_effort.
    8. Returns updated state fields.
    """
    from app.config.app_settings import agent_settings as _agent_settings

    async def node(state: dict) -> dict:
        from langchain_core.messages import HumanMessage, AIMessage

        messages = state.get("messages", [])

        # ── 1. Câu hỏi gốc: last HumanMessage not starting with [VALIDATOR] ─
        original_question: Optional[HumanMessage] = None
        for m in reversed(messages):
            if isinstance(m, HumanMessage):
                content = _extract_text(m.content)
                if not content.startswith(_VALIDATOR_GUIDANCE_PREFIX):
                    original_question = m
                    break

        # ── 2. Kết quả sub-agent: last AIMessage from rag_agent or web_agent ─
        sub_agent_result: Optional[AIMessage] = None
        for m in reversed(messages):
            if isinstance(m, AIMessage) and getattr(m, "name", "") in ("rag_agent", "web_agent"):
                sub_agent_result = m
                break

        # Build validator input
        validator_input: list = []
        if original_question:
            validator_input.append(original_question)
        if sub_agent_result:
            validator_input.append(sub_agent_result)
        if not validator_input:
            validator_input = messages  # fallback

        # ── 3. Call validator LLM ─────────────────────────────────────────────
        grade: str = "AMBIGUOUS"
        rewritten_query_raw: Optional[str] = None
        reason: str = ""
        produced: list = []

        try:
            result = await compiled_validator.ainvoke({"messages": validator_input})
            produced = result["messages"][len(validator_input):]

            # ── Strip Bedrock thinking blocks (same as _make_sub_agent_node) ─
            cleaned: list = []
            for msg in produced:
                if isinstance(msg, AIMessage) and isinstance(msg.content, list):
                    text_parts = [p for p in msg.content
                                  if isinstance(p, dict) and p.get("type") == "text"]
                    if text_parts:
                        msg = copy.copy(msg)
                        msg.content = (
                            text_parts if len(text_parts) > 1
                            else text_parts[0].get("text", "")
                        )
                cleaned.append(msg)
            produced = cleaned

            # ── 4. Parse guidance from last AIMessage ────────────────────────
            for msg in reversed(produced):
                content_str = _extract_text(getattr(msg, "content", ""))
                if _VALIDATOR_GUIDANCE_PREFIX in content_str:
                    grade, rewritten_query_raw, reason = _parse_validator_guidance(content_str)
                    break

        except Exception as exc:  # noqa: BLE001
            grade = "AMBIGUOUS"
            reason = f"validator error: {exc}"
            _log.warning("[validator] LLM call failed → AMBIGUOUS | error=%s", exc)

        # ── 5. Routing decision (pure function) ──────────────────────────────
        current_retry = state.get("retry_count", 0)
        decision = decide_routing(grade, current_retry, retry_budget)

        # ── 6. retry_count increment ─────────────────────────────────────────
        new_retry_count = current_retry + 1 if decision == "retry" else current_retry

        # ── 7. Normalise rewritten_query ─────────────────────────────────────
        fallback = _extract_text(getattr(original_question, "content", "")) if original_question else ""
        normalized_query = _normalize_rewritten_query(grade, rewritten_query_raw, fallback)

        # ── 8. Build guidance HumanMessage ───────────────────────────────────
        budget_note = ""
        if decision == "synthesize_best_effort":
            budget_note = " [đã hết số lần thử — hãy tổng hợp best-effort và nêu rõ phần thiếu]"

        guidance_content = (
            f"[VALIDATOR] Grade={grade}"
            f" rewritten_query={normalized_query or 'None'}"
            f" reason={reason}{budget_note}"
        )
        guidance_msg = HumanMessage(content=guidance_content)

        # Replace any [VALIDATOR] AIMessage from validator with our HumanMessage.
        # Keep non-validator produced messages (e.g. tool_call artifacts).
        validator_ai_messages = [
            m for m in produced
            if isinstance(m, AIMessage) and _VALIDATOR_GUIDANCE_PREFIX in _extract_text(getattr(m, "content", ""))
        ]
        other_produced = [m for m in produced if m not in validator_ai_messages]
        final_produced = other_produced + [guidance_msg]

        _log.info(
            "[validator] grade=%s | decision=%s | retry=%d→%d | rq=%r | reason=%s",
            grade, decision, current_retry, new_retry_count,
            normalized_query, reason,
        )

        return {
            "messages": final_produced,
            "last_grade": grade,
            "retry_count": new_retry_count,
            "rewritten_query": normalized_query,
        }

    return node


def _route_after_validator(state: dict) -> str:
    """After validator: route to supervisor unless MAX_HOPS reached."""
    if state.get("agent_hops", 0) >= MAX_HOPS:
        _log.warning("MAX_HOPS=%d reached after validator → forcing END", MAX_HOPS)
        return END
    return "supervisor"


def build_supervisor_graph(agent_tools: dict, checkpointer=None):
    """Build and compile the supervisor multi-agent graph.

    Graph topology (post-validator upgrade):

        START → supervisor
        supervisor --(handoff_to_rag_agent)--> rag_agent
        supervisor --(handoff_to_web_agent)--> web_agent
        supervisor --(no handoff = final answer)--> END
        rag_agent → validator
        web_agent → validator
        validator → supervisor  (or END if MAX_HOPS reached)

    Args:
        agent_tools: mapping of agent name -> its toolset, as returned by
            ``build_agent_tools(vdb)``.
        checkpointer: optional LangGraph checkpointer for persisting conversation
            memory across turns. When ``None`` the graph runs stateless.
    """
    from app.agent.agents.validator import create_validator_agent
    from app.config.app_settings import agent_settings

    supervisor = create_supervisor(agent_tools["supervisor"])
    rag_agent = create_rag_agent(agent_tools["rag_agent"])
    web_agent = create_web_agent(agent_tools["web_agent"])
    validator = create_validator_agent()

    builder = StateGraph(AgentState)

    # Nodes
    builder.add_node("supervisor", supervisor)
    builder.add_node("rag_agent", _make_sub_agent_node(rag_agent, "rag_agent"))
    builder.add_node("web_agent", _make_sub_agent_node(web_agent, "web_agent"))
    builder.add_node("validator", _make_validator_node(validator, agent_settings.RETRY_BUDGET))

    # Edges
    builder.add_edge(START, "supervisor")
    # Supervisor routes to sub-agents via handoff tools (Command.PARENT). When it
    # produces a final answer (no handoff), control falls through to END.
    builder.add_edge("supervisor", END)
    # Sub-agents now always go to the validator (not directly to supervisor).
    builder.add_edge("rag_agent", "validator")
    builder.add_edge("web_agent", "validator")
    # Validator routes back to supervisor or END based on agent_hops.
    builder.add_conditional_edges("validator", _route_after_validator, ["supervisor", END])

    graph = builder.compile(checkpointer=checkpointer)
    _log.info(
        "Supervisor multi-agent graph compiled "
        "(nodes: supervisor, rag_agent, web_agent, validator | memory=%s)",
        "checkpointer" if checkpointer else "none",
    )
    return graph
