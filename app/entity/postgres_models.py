"""SQLAlchemy ORM models cho PostgreSQL — convention theo DB-service."""

from datetime import datetime, timezone
from sqlalchemy import BigInteger, Column, Float, String, Text, DateTime
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id            = Column(BigInteger, primary_key=True, autoincrement=True)
    thread_id     = Column(String(255), nullable=False)
    role          = Column(String(20),  nullable=False)
    # role values:
    #   'user'       — câu hỏi của người dùng
    #   'assistant'  — câu trả lời cuối của supervisor
    #   'agent_step' — bước trung gian: reasoning, tool_call, observation
    content       = Column(Text,        nullable=False)
    agent         = Column(String(150), nullable=True)           # 'supervisor' | 'rag_agent' | 'web_agent'
    step_type     = Column(String(20),  nullable=True)           # 'reasoning' | 'tool_call' | 'observation' | NULL
    tool_name     = Column(String(150), nullable=True)           # tool name, có thể dài (MCP tools)
    time_response = Column(Float,       nullable=True)           # giây, chỉ có ở assistant message
    created_at    = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    updated_at    = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc),
                           onupdate=lambda: datetime.now(timezone.utc))
