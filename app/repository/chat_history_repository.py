"""Repository cho bảng chat_messages — convention theo DB-service."""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.config.env.postgres_config import get_async_session_factory
from app.entity.postgres_models import ChatMessage

_log = logging.getLogger(__name__)


class ChatMessageRepository:

    @staticmethod
    def _serialize_row(row: ChatMessage) -> Dict[str, Any]:
        return {
            col.key: (
                getattr(row, col.key).isoformat()
                if isinstance(getattr(row, col.key), datetime)
                else getattr(row, col.key)
            )
            for col in ChatMessage.__table__.columns
        }

    async def save(
        self,
        thread_id: str,
        role: str,
        content: str,
        agent: Optional[str] = None,
        step_type: Optional[str] = None,
        tool_name: Optional[str] = None,
        time_response: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Lưu 1 tin nhắn vào bảng chat_messages."""
        try:
            session_factory: async_sessionmaker[AsyncSession] = get_async_session_factory()
            async with session_factory() as session:
                now = datetime.now(timezone.utc)
                row = ChatMessage(
                    thread_id=thread_id,
                    role=role,
                    content=content,
                    agent=agent,
                    step_type=step_type,
                    tool_name=tool_name,
                    time_response=time_response,
                    created_at=now,
                    updated_at=now,
                )
                session.add(row)
                await session.commit()
                await session.refresh(row)
                return self._serialize_row(row)
        except Exception as e:
            _log.error("Error saving chat message: %s", e, exc_info=True)
            raise

    async def save_turn_with_steps(
        self,
        thread_id: str,
        question: str,
        answer: str,
        steps: List[Dict[str, Any]],
        time_response: float,
    ) -> None:
        """Lưu toàn bộ 1 turn trong 1 transaction:
        user message + tất cả agent steps (reasoning/tool_call/observation) + final assistant answer.

        Args:
            thread_id:     Session ID.
            question:      Câu hỏi gốc của user.
            answer:        Câu trả lời cuối của supervisor.
            steps:         List steps từ engine.run() — mỗi phần tử có:
                           {type, tool, agent, content}
            time_response: Thời gian xử lý toàn bộ turn (giây).
        """
        try:
            session_factory: async_sessionmaker[AsyncSession] = get_async_session_factory()
            async with session_factory() as session:
                now = datetime.now(timezone.utc)

                # 1. User message
                session.add(ChatMessage(
                    thread_id=thread_id,
                    role="user",
                    content=question,
                    created_at=now,
                    updated_at=now,
                ))

                # 2. Agent steps — reasoning, tool_call, observation
                # step.type: "reason" | "tool_call" | "observation" (từ engine.run())
                _STEP_TYPE_MAP = {
                    "reason":      "reasoning",
                    "tool_call":   "tool_call",
                    "observation": "observation",
                }
                for step in steps:
                    raw_type  = step.get("type", "")
                    tool      = step.get("tool", "") or None
                    content   = step.get("content", "")
                    agent     = step.get("agent", "") or None
                    if not content:
                        continue
                    session.add(ChatMessage(
                        thread_id=thread_id,
                        role="agent_step",
                        content=content,
                        agent=agent,
                        step_type=_STEP_TYPE_MAP.get(raw_type, raw_type) or None,
                        tool_name=tool,
                        created_at=now,
                        updated_at=now,
                    ))

                # 3. Final assistant answer
                session.add(ChatMessage(
                    thread_id=thread_id,
                    role="assistant",
                    content=answer,
                    agent="supervisor",
                    time_response=time_response,
                    created_at=now,
                    updated_at=now,
                ))

                await session.commit()
                _log.debug(
                    "Saved turn | thread_id=%s | steps=%d",
                    thread_id, len(steps),
                )
        except Exception as e:
            _log.error("Error saving turn with steps | thread_id=%s: %s", thread_id, e, exc_info=True)
            raise

    async def find_by_thread(
        self, thread_id: str, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Lấy lịch sử hội thoại theo thread_id, sắp xếp theo thời gian."""
        try:
            session_factory: async_sessionmaker[AsyncSession] = get_async_session_factory()
            async with session_factory() as session:
                stmt = (
                    select(ChatMessage)
                    .where(ChatMessage.thread_id == thread_id)
                    .order_by(ChatMessage.created_at.asc())
                    .limit(limit)
                )
                result = await session.execute(stmt)
                rows = result.scalars().all()
                return [self._serialize_row(r) for r in rows]
        except Exception as e:
            _log.error("Error fetching chat history thread_id=%s: %s", thread_id, e, exc_info=True)
            raise

    async def find_by_thread_roles(
        self,
        thread_id: str,
        roles: Optional[List[str]] = None,
        limit: int = 200,
    ) -> List[Dict[str, Any]]:
        """Lấy messages theo thread_id và lọc theo role.

        Ví dụ:
            # Chỉ lấy user + assistant (bỏ agent_step) — cho UI chat
            repo.find_by_thread_roles(tid, roles=["user", "assistant"])

            # Lấy tất cả kể cả agent_step — cho debug / audit
            repo.find_by_thread_roles(tid)
        """
        try:
            session_factory: async_sessionmaker[AsyncSession] = get_async_session_factory()
            async with session_factory() as session:
                stmt = select(ChatMessage).where(ChatMessage.thread_id == thread_id)
                if roles:
                    stmt = stmt.where(ChatMessage.role.in_(roles))
                stmt = stmt.order_by(ChatMessage.created_at.asc()).limit(limit)
                result = await session.execute(stmt)
                rows = result.scalars().all()
                return [self._serialize_row(r) for r in rows]
        except Exception as e:
            _log.error("Error fetching chat history thread_id=%s: %s", thread_id, e, exc_info=True)
            raise

    async def find_all_threads(self) -> List[str]:
        """Lấy danh sách tất cả thread_id distinct."""
        try:
            session_factory: async_sessionmaker[AsyncSession] = get_async_session_factory()
            async with session_factory() as session:
                from sqlalchemy import distinct
                stmt = select(distinct(ChatMessage.thread_id)).order_by(ChatMessage.thread_id)
                result = await session.execute(stmt)
                return [r[0] for r in result.all()]
        except Exception as e:
            _log.error("Error fetching thread list: %s", e, exc_info=True)
            raise
