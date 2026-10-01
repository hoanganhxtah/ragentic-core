import logging
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from app.schemas.agent import AgentRequest, AgentResponse, AgentStep

_log = logging.getLogger(__name__)
router = APIRouter()
agent_engine = None  # set by lifespan in main.py


async def _save_turn_full(
    thread_id: str,
    question: str,
    answer: str,
    all_steps: list,
    time_response: float,
) -> None:
    """Lưu 1 lượt hội thoại đầy đủ (user + agent_steps + assistant) vào PostgreSQL.

    Lỗi DB được nuốt và chỉ log warning để không làm hỏng response trả về client.
    """
    try:
        from app.repository.chat_history_repository import ChatMessageRepository
        repo = ChatMessageRepository()
        await repo.save_turn_with_steps(
            thread_id=thread_id,
            question=question,
            answer=answer,
            steps=all_steps,
            time_response=time_response,
        )
    except Exception as db_err:
        _log.warning("Failed to save chat history | thread_id=%s: %s", thread_id, db_err)


async def _save_turn_simple(
    thread_id: str,
    question: str,
    answer: str,
    time_response: float,
) -> None:
    """Lưu lượt hội thoại đơn giản (user + assistant) — dùng cho stream endpoint
    vì stream không có steps sẵn.
    """
    try:
        from app.repository.chat_history_repository import ChatMessageRepository
        repo = ChatMessageRepository()
        await repo.save_turn_with_steps(
            thread_id=thread_id,
            question=question,
            answer=answer,
            steps=[],
            time_response=time_response,
        )
    except Exception as db_err:
        _log.warning("Failed to save chat history | thread_id=%s: %s", thread_id, db_err)


@router.post("/agent", response_model=AgentResponse)
async def agent_endpoint(request: AgentRequest):
    """Multi-agent supervisor endpoint với conversation memory."""
    _log.info("Agent request | thread_id=%s | include_steps=%s", request.thread_id, request.include_steps)

    if agent_engine is None:
        raise HTTPException(status_code=503, detail="Agent engine not initialized")
    if not request.question or not request.question.strip():
        raise HTTPException(status_code=400, detail="question must be non-empty")

    try:
        # engine.run() trả 4-tuple: (answer, api_steps, all_steps, time_response)
        # api_steps: chỉ có nội dung khi include_steps=True — dùng cho API response
        # all_steps: luôn có — dùng để lưu DB đầy đủ
        answer, api_steps, all_steps, time_response = await agent_engine.run(
            request.question,
            request.thread_id,
            request.include_steps,
        )

        # Lưu lịch sử đầy đủ vào PostgreSQL (best-effort)
        await _save_turn_full(request.thread_id, request.question, answer, all_steps, time_response)

        return AgentResponse(answer=answer, steps=[AgentStep(**s) for s in api_steps])

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        _log.exception("Agent error | thread_id=%s", request.thread_id)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/agent-stream")
async def agent_stream_endpoint(request: AgentRequest):
    """Multi-agent endpoint với SSE streaming — trả token ngay khi LLM sinh ra.

    Response là ``text/event-stream``. Mỗi event là một JSON line:
    - ``{"type":"token","content":"...","agent":"supervisor|rag_agent|web_agent"}``
    - ``{"type":"step","status":"tool_start|tool_end","agent":"...","tool":"...","content":"..."}``
    - ``{"type":"done","answer":"...","time_response":1.23}``
    - ``{"type":"error","detail":"..."}``

    ``step`` events chỉ được emit khi ``include_steps=true`` trong request.
    """
    _log.info(
        "Agent stream request | thread_id=%s | include_steps=%s",
        request.thread_id, request.include_steps,
    )

    if agent_engine is None:
        raise HTTPException(status_code=503, detail="Agent engine not initialized")
    if not request.question or not request.question.strip():
        raise HTTPException(status_code=400, detail="question must be non-empty")

    async def _generate():
        full_answer = ""
        time_response = 0.0
        all_steps: list = []
        async for chunk in agent_engine.stream(
            request.question,
            request.thread_id,
            request.include_steps,
        ):
            # Theo dõi done event để lưu DB sau khi stream xong
            if '"type": "done"' in chunk or '"type":"done"' in chunk:
                import json as _json
                try:
                    payload = _json.loads(chunk.removeprefix("data: ").strip())
                    full_answer = payload.get("answer", "")
                    time_response = payload.get("time_response", 0.0)
                    all_steps = payload.get("_all_steps", [])  # steps từ engine.stream()
                    # Xóa _all_steps trước khi gửi xuống client (internal field)
                    payload.pop("_all_steps", None)
                    chunk = f"data: {_json.dumps(payload, ensure_ascii=False)}\n\n"
                except Exception:
                    pass
            yield chunk

        # Lưu lịch sử đầy đủ sau khi stream hoàn tất (best-effort)
        if full_answer:
            await _save_turn_full(request.thread_id, request.question, full_answer, all_steps, time_response)

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # tắt nginx buffer để token đến ngay lập tức
        },
    )
