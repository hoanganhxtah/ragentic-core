"""app/agent/prompts — tất cả prompts của hệ thống tập trung tại đây.

5 module:
- supervisor  → SUPERVISOR_PROMPT
- rag_agent   → RAG_AGENT_PROMPT
- web_agent   → WEB_AGENT_PROMPT
- validator   → VALIDATOR_PROMPT
- chat        → INTENT_CLASSIFICATION_PROMPT, REFORMULATION_PROMPT,
                RELEVANCE_CHECK_PROMPT, CLARIFICATION_PROMPT, RAG_ANSWER_PROMPT
"""

from app.agent.prompts.supervisor import SUPERVISOR_PROMPT
from app.agent.prompts.rag_agent import RAG_AGENT_PROMPT
from app.agent.prompts.web_agent import WEB_AGENT_PROMPT
from app.agent.prompts.validator import VALIDATOR_PROMPT
from app.agent.prompts.chat import (
    INTENT_CLASSIFICATION_PROMPT,
    REFORMULATION_PROMPT,
    RELEVANCE_CHECK_PROMPT,
    CLARIFICATION_PROMPT,
    RAG_ANSWER_PROMPT,
)

__all__ = [
    "SUPERVISOR_PROMPT",
    "RAG_AGENT_PROMPT",
    "WEB_AGENT_PROMPT",
    "VALIDATOR_PROMPT",
    "INTENT_CLASSIFICATION_PROMPT",
    "REFORMULATION_PROMPT",
    "RELEVANCE_CHECK_PROMPT",
    "CLARIFICATION_PROMPT",
    "RAG_ANSWER_PROMPT",
]
