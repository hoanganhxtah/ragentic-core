from pydantic import BaseModel
from typing import List, Literal, Optional


class AgentStep(BaseModel):
    type: str
    tool: str = ""
    agent: str = ""
    content: str


class AgentRequest(BaseModel):
    question: str
    thread_id: str = "default"   # session ID — LangGraph dùng để load/save checkpoint
    include_steps: bool = False


class AgentResponse(BaseModel):
    answer: str
    steps: List[AgentStep] = []


# ── SSE streaming events ──────────────────────────────────────────────────────

class StreamTokenEvent(BaseModel):
    """Một token LLM được sinh ra (supervisor hoặc sub-agent đang viết)."""
    type: Literal["token"] = "token"
    content: str
    agent: str = ""           # tên node đang chạy: supervisor | rag_agent | web_agent


class StreamStepEvent(BaseModel):
    """Tool call hoặc kết thúc node — dùng để hiển thị bước trung gian."""
    type: Literal["step"] = "step"
    status: Literal["tool_start", "tool_end", "node_end"]
    agent: str = ""
    tool: str = ""
    content: str = ""


class StreamDoneEvent(BaseModel):
    """Sự kiện cuối cùng — mang toàn bộ câu trả lời và trạng thái hoàn tất."""
    type: Literal["done"] = "done"
    answer: str
    time_response: float


class StreamErrorEvent(BaseModel):
    """Lỗi xảy ra trong quá trình stream."""
    type: Literal["error"] = "error"
    detail: str
