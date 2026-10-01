"""Supervisor multi-agent engine.

Dùng ``MemorySaver`` cho conversation memory trong phiên (in-memory, per-process).
Lịch sử chat được lưu vào bảng ``chat_messages`` (PostgreSQL) sau mỗi turn bởi
endpoint — engine trả về ``(answer, api_steps, all_steps, time_response)``.

Thinking / STATUS xử lý theo từng path:
- stream()  → phân tách thành event ``thinking`` / ``token`` riêng; UI tự render
- run()     → strip toàn bộ thinking + [STATUS:] trước khi trả JSON answer
"""

from __future__ import annotations

import json
import logging
import re as _re
import time
from typing import AsyncGenerator, Optional

from langgraph.checkpoint.memory import MemorySaver

from app.config.app_settings import llm_settings
from app.agent.tools.registry import build_agent_tools, build_agent_tools_async
from app.agent.graph import build_supervisor_graph, _extract_text
from app.retrieval.base import Retriever

_log = logging.getLogger(__name__)

# Chỉ dùng trong run() — stream() để UI xử lý presentation
_THINKING_RE = _re.compile(r"<thinking>.*?</thinking>", _re.DOTALL | _re.IGNORECASE)
_STATUS_RE   = _re.compile(r"\[STATUS:\s*\w+\]", _re.IGNORECASE)
_BR_RE       = _re.compile(r"<br\s*/?>", _re.IGNORECASE)


def _strip_thinking(text: str) -> str:
    """Dùng cho run() — strip thinking blocks, [STATUS:], và <br> tags khỏi final answer JSON."""
    if not text:
        return ""
    text = _THINKING_RE.sub("", text)
    text = _STATUS_RE.sub("", text)
    text = _BR_RE.sub("\n", text)
    return text.strip()


def _is_trivial(text: str) -> bool:
    """Trả True nếu text quá ngắn/vô nghĩa để làm câu trả lời cuối.

    Supervisor đôi khi chỉ emit dấu chấm hoặc một từ đơn sau khi sub-agent
    đã trả lời đầy đủ. Những message đó không nên được dùng làm final answer.
    Ngưỡng 10 ký tự bao gồm cả trường hợp "OK.", "Xong.", "." v.v.
    """
    return len(text.strip()) < 10


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _default_retriever() -> Retriever:
    """Fallback retriever khi caller không tiêm sẵn (script, debug, test).

    Production path là ``app/main.py`` lifespan: nó tạo retriever một lần và
    chia sẻ cho cả search API và agent engine.
    """
    from app.retrieval import build_retriever

    return build_retriever()


def _build_steps_for_turn(all_msgs: list) -> list:
    """Xây dựng danh sách steps chỉ cho TURN HIỆN TẠI.

    - Chỉ lấy messages từ HumanMessage cuối cùng trở đi (bỏ qua lịch sử cũ của MemorySaver).
    - ToolMessage được gán cho agent đã gọi tool (AIMessage trước nó),
      không dùng msg.name vì ToolMessage.name = tên tool, không phải tên agent.
    - HumanMessage bắt đầu bằng [VALIDATOR] → step type "validate" (không bỏ qua).
    """
    from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

    # Tìm vị trí bắt đầu turn hiện tại = HumanMessage cuối cùng
    turn_start = 0
    for i, msg in enumerate(all_msgs):
        if isinstance(msg, HumanMessage):
            content = _extract_text(getattr(msg, "content", ""))
            if not content.startswith("[VALIDATOR]"):
                turn_start = i

    steps: list = []
    last_ai_agent = ""   # agent gọi tool gần nhất — dùng để gán cho ToolMessage

    for msg in all_msgs[turn_start:]:
        if isinstance(msg, HumanMessage):
            # [VALIDATOR] guidance lines → emit as validate step (Req 8.8)
            content = _extract_text(getattr(msg, "content", ""))
            if content.startswith("[VALIDATOR]"):
                steps.append({
                    "type": "validate",
                    "agent": "validator",
                    "tool": "",
                    "content": content,
                })
            # All other HumanMessages (the user's question) are skipped.
            continue

        agent_name = getattr(msg, "name", "") or ""

        if isinstance(msg, AIMessage) and msg.tool_calls:
            last_ai_agent = agent_name
            for call in msg.tool_calls:
                steps.append({
                    "type": "tool_call",
                    "tool": call.get("name", ""),
                    "agent": agent_name,
                    "content": str(call.get("args", "")),
                })
        elif isinstance(msg, ToolMessage):
            # ToolMessage.name = tên tool (không phải agent) — dùng last_ai_agent
            steps.append({
                "type": "observation",
                "tool": getattr(msg, "name", "") or "",
                "agent": last_ai_agent,
                "content": str(msg.content),
            })
        elif isinstance(msg, AIMessage):
            last_ai_agent = agent_name
            steps.append({
                "type": "reason",
                "tool": "",
                "agent": agent_name,
                "content": str(msg.content),
            })

    return steps


class _ThinkingSplitter:
    """Tách stream token thành các đoạn (channel, text) khi gặp <thinking> tag.

    channel = "think"  → nội dung bên trong <thinking>...</thinking>
    channel = "answer" → nội dung bên ngoài tag

    Xử lý đúng trường hợp tag bị cắt qua nhiều token: giữ lại phần đuôi có thể
    là partial tag, chờ token kế tiếp — không bao giờ emit tag ra ngoài.
    """

    OPEN = "<thinking>"
    CLOSE = "</thinking>"

    def __init__(self):
        self.buf = ""
        self.in_think = False

    @staticmethod
    def _safe_len(buf: str, tag: str) -> int:
        """Độ dài an toàn có thể emit mà không cắt phải 1 tag tiềm năng ở cuối."""
        max_keep = min(len(tag) - 1, len(buf))
        for k in range(max_keep, 0, -1):
            if buf[-k:].lower() == tag[:k].lower():
                return len(buf) - k
        return len(buf)

    def feed(self, text: str):
        """Nhận thêm text, trả về list[(channel, text)] có thể emit ngay."""
        self.buf += text
        out = []
        while self.buf:
            if self.in_think:
                idx = self.buf.find(self.CLOSE)
                if idx == -1:
                    safe = self._safe_len(self.buf, self.CLOSE)
                    if safe > 0:
                        out.append(("think", self.buf[:safe]))
                        self.buf = self.buf[safe:]
                    break
                if idx > 0:
                    out.append(("think", self.buf[:idx]))
                self.buf = self.buf[idx + len(self.CLOSE):]
                self.in_think = False
            else:
                idx = self.buf.lower().find(self.OPEN)
                if idx == -1:
                    safe = self._safe_len(self.buf, self.OPEN)
                    if safe > 0:
                        out.append(("answer", self.buf[:safe]))
                        self.buf = self.buf[safe:]
                    break
                if idx > 0:
                    out.append(("answer", self.buf[:idx]))
                self.buf = self.buf[idx + len(self.OPEN):]
                self.in_think = True
        return out

    def flush(self):
        """Đẩy nốt buffer còn lại khi stream kết thúc."""
        if not self.buf:
            return []
        ch = "think" if self.in_think else "answer"
        out = [(ch, self.buf)]
        self.buf = ""
        self.in_think = False
        return out


class ReActAgentEngine:
    """Supervisor multi-agent engine với MemorySaver cho trong-phiên memory.

    Khởi tạo:
    - Sync  : ``ReActAgentEngine(retriever)``         — không load MCP tools
    - Async : ``await ReActAgentEngine.create(retriever)`` — load MCP tools đúng cách
      Dùng ``create()`` từ FastAPI lifespan để có đầy đủ banking tools.

    Engine chỉ biết ``Retriever`` protocol, không biết vector store nào phía sau.
    """

    def __init__(self, retriever: Optional[Retriever] = None, agent_tools=None):
        self.retriever = retriever if retriever is not None else _default_retriever()
        # Nếu agent_tools được truyền vào (từ create()), dùng luôn.
        # Nếu không, build sync (không có MCP tools).
        self.agent_tools = (
            agent_tools if agent_tools is not None else build_agent_tools(self.retriever)
        )
        checkpointer = MemorySaver()
        self.graph = build_supervisor_graph(self.agent_tools, checkpointer=checkpointer)

        _log.info(
            "Agent engine initialized (provider=%s | model=%s | memory=MemorySaver | tools=%s)",
            llm_settings.PROVIDER,
            llm_settings.MODEL,
            [t.name for tools in self.agent_tools.values() for t in tools],
        )

    @classmethod
    async def create(cls, retriever: Optional[Retriever] = None) -> "ReActAgentEngine":
        """Async factory — dùng từ FastAPI lifespan để load MCP tools đúng cách.

        Ví dụ:
            agent_engine = await ReActAgentEngine.create(retriever=retriever)
        """
        if retriever is None:
            retriever = _default_retriever()

        agent_tools = await build_agent_tools_async(retriever)
        return cls(retriever=retriever, agent_tools=agent_tools)

    # ─────────────────────────────────────────────────────────────────────────
    # run() — non-stream, trả (answer, steps, time_response)
    # Backend strip thinking + status vì UI chỉ nhận string cuối.
    # ─────────────────────────────────────────────────────────────────────────
    async def run(
        self,
        question: str,
        thread_id: str = "default",
        include_steps: bool = False,
    ) -> tuple[str, list, list, float]:
        """Run a single turn and return (answer, api_steps, all_steps, time_response).

        Returns:
            answer:        Final answer string (thinking and status markers stripped).
            api_steps:     Steps list returned to API caller (only when include_steps=True).
            all_steps:     Full steps list (always built, used for DB persistence).
            time_response: Elapsed time in seconds (float, 3 decimals).
        """
        if not question or not question.strip():
            raise ValueError("question must be non-empty")

        from langchain_core.messages import HumanMessage, AIMessage

        config = {"configurable": {"thread_id": thread_id}}
        t0 = time.perf_counter()
        result = await self.graph.ainvoke(
            {
                "messages": [HumanMessage(content=question)],
                "agent_hops": 0,
                "last_grade": None,
                "retry_count": 0,
                "rewritten_query": None,
            },
            config=config,
        )
        time_response = round(time.perf_counter() - t0, 3)
        msgs = result["messages"]

        answer = ""
        # Pass 1: supervisor message ưu tiên — nhưng bỏ qua nếu trivial (dấu chấm, v.v.)
        for msg in reversed(msgs):
            if (isinstance(msg, AIMessage)
                    and not msg.tool_calls
                    and getattr(msg, "name", "") in ("supervisor", "", None)):
                candidate = _strip_thinking(_extract_text(msg.content))
                if candidate and not _is_trivial(candidate):
                    answer = candidate
                    break
        # Pass 2: fallback sub-agent — lấy message dài nhất có nội dung thực
        if not answer:
            for msg in reversed(msgs):
                if isinstance(msg, AIMessage) and not msg.tool_calls:
                    candidate = _strip_thinking(_extract_text(msg.content))
                    if candidate and not _is_trivial(candidate):
                        answer = candidate
                        break
        # Pass 3: last resort — lấy bất kỳ supervisor message nào dù trivial
        if not answer:
            for msg in reversed(msgs):
                if (isinstance(msg, AIMessage)
                        and not msg.tool_calls
                        and getattr(msg, "name", "") in ("supervisor", "", None)):
                    candidate = _strip_thinking(_extract_text(msg.content))
                    if candidate:
                        answer = candidate
                        break

        # Build steps cho turn hiện tại (lọc khỏi history cũ của MemorySaver)
        all_steps = _build_steps_for_turn(msgs)

        # api_steps: trả cho client (chỉ khi include_steps=True)
        # all_steps: luôn được build — dùng để lưu DB
        api_steps = all_steps if include_steps else []
        return (answer, api_steps, all_steps, time_response)

    # ─────────────────────────────────────────────────────────────────────────
    # stream() — SSE.
    # Kiến trúc (xác nhận qua log): supervisor KHÔNG stream câu trả lời cuối —
    # nó chỉ reasoning + handoff. Câu trả lời thật là message cuối cùng trong
    # graph state (giống run()). Vì vậy:
    #   - Mọi text model stream → event "thinking" (hiển thị reasoning live)
    #   - Kết thúc → trích câu trả lời cuối từ state → event "token" + "done"
    # ─────────────────────────────────────────────────────────────────────────
    async def stream(
        self,
        question: str,
        thread_id: str = "default",
        include_steps: bool = False,
    ) -> AsyncGenerator[str, None]:
        """Stream graph events dưới dạng SSE.

        Event types:
        - ``thinking``  : reasoning live (supervisor + sub-agents)
        - ``token``     : câu trả lời cuối (trích từ state khi kết thúc)
        - ``step``      : tool call / result (khi include_steps=True)
        - ``done``      : {"answer": str, "time_response": float}
        - ``error``     : {"detail": str}
        """
        from langchain_core.messages import HumanMessage, AIMessage

        if not question or not question.strip():
            yield _sse({"type": "error", "detail": "question must be non-empty"})
            return

        config = {"configurable": {"thread_id": thread_id}}
        t0 = time.perf_counter()

        # Splitter cho từng context — chỉ để LỌC BỎ literal <thinking> tag khỏi
        # phần reasoning hiển thị (không leak tag ra UI). Cả 2 channel đều coi
        # là reasoning vì câu trả lời cuối lấy riêng từ state.
        splitters: dict[str, _ThinkingSplitter] = {}

        try:
            async for event in self.graph.astream_events(
                {
                    "messages": [HumanMessage(content=question)],
                    "agent_hops": 0,
                    "last_grade": None,
                    "retry_count": 0,
                    "rewritten_query": None,
                },
                config=config,
                version="v2",
            ):
                kind          = event.get("event", "")
                name          = event.get("name", "")
                metadata      = event.get("metadata", {})
                checkpoint_ns = metadata.get("checkpoint_ns", "")
                node          = metadata.get("langgraph_node", "")

                is_supervisor = checkpoint_ns.startswith("supervisor:")

                # ── Step trace (tool calls) — chỉ của sub-agents ──────────
                if kind == "on_tool_start" and include_steps and not is_supervisor:
                    tool_input = event.get("data", {}).get("input", {})
                    yield _sse({
                        "type": "step",
                        "status": "tool_start",
                        "agent": name,
                        "tool": name,
                        "content": json.dumps(tool_input, ensure_ascii=False),
                    })

                elif kind == "on_tool_end" and include_steps and not is_supervisor:
                    output = event.get("data", {}).get("output", "")
                    yield _sse({
                        "type": "step",
                        "status": "tool_end",
                        "agent": name,
                        "tool": name,
                        "content": str(output)[:500] if output else "",
                    })

                # ── Reasoning streaming → thinking ────────────────────────
                elif kind == "on_chat_model_stream":
                    chunk = event.get("data", {}).get("chunk")
                    if not (chunk and hasattr(chunk, "content")):
                        continue
                    if getattr(chunk, "tool_calls", None) or getattr(chunk, "tool_call_chunks", None):
                        continue

                    text = _extract_text(chunk.content)
                    if not text:
                        continue

                    sp = splitters.setdefault(checkpoint_ns, _ThinkingSplitter())
                    # Cả answer + think channel đều là reasoning live — gộp lại,
                    # chỉ loại bỏ literal tag + [STATUS:] để không leak ra UI.
                    for _channel, seg in sp.feed(text):
                        clean = _STATUS_RE.sub("", seg)
                        if clean.strip():
                            agent = "supervisor" if is_supervisor else (node or "agent")
                            yield _sse({"type": "thinking", "content": clean, "agent": agent})

            # Flush splitter còn sót
            for sp in splitters.values():
                for _channel, seg in sp.flush():
                    clean = _STATUS_RE.sub("", seg)
                    if clean.strip():
                        yield _sse({"type": "thinking", "content": clean, "agent": "agent"})

            # ── Trích câu trả lời cuối từ graph state (giống run()) ───────
            # Ưu tiên lấy AIMessage của supervisor (name="supervisor").
            # Chỉ fallback sang sub-agent message nếu supervisor không có content.
            final_answer = ""
            state = self.graph.get_state(config)
            all_msgs = state.values.get("messages", [])

            # Pass 1: tìm supervisor message cuối có nội dung không trivial
            for msg in reversed(all_msgs):
                if (isinstance(msg, AIMessage)
                        and not msg.tool_calls
                        and getattr(msg, "name", "") in ("supervisor", "", None)):
                    candidate = _strip_thinking(_extract_text(msg.content))
                    if candidate and not _is_trivial(candidate):
                        final_answer = candidate
                        break

            # Pass 2: nếu supervisor message trivial/rỗng, lấy sub-agent message cuối
            if not final_answer:
                for msg in reversed(all_msgs):
                    if isinstance(msg, AIMessage) and not msg.tool_calls:
                        candidate = _strip_thinking(_extract_text(msg.content))
                        if candidate and not _is_trivial(candidate):
                            final_answer = candidate
                            break

            # Pass 3: last resort — lấy bất kỳ supervisor message nào dù trivial
            if not final_answer:
                for msg in reversed(all_msgs):
                    if (isinstance(msg, AIMessage)
                            and not msg.tool_calls
                            and getattr(msg, "name", "") in ("supervisor", "", None)):
                        candidate = _strip_thinking(_extract_text(msg.content))
                        if candidate:
                            final_answer = candidate
                            break

            if final_answer:
                yield _sse({"type": "token", "content": final_answer, "agent": "supervisor"})

            # Build steps chỉ cho turn hiện tại (độc lập với include_steps)
            all_steps = _build_steps_for_turn(all_msgs)

            yield _sse({
                "type": "done",
                "answer": final_answer,
                "time_response": round(time.perf_counter() - t0, 3),
                "_all_steps": all_steps,   # internal field — API layer dùng để lưu DB
            })

        except Exception as exc:
            _log.exception("stream() error | thread_id=%s", thread_id)
            yield _sse({"type": "error", "detail": str(exc)})
