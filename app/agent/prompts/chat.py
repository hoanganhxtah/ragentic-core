"""Chat pipeline prompts — dùng cho /search và /chat endpoints (legacy RAG pipeline).

Các prompt này phục vụ luồng RAG truyền thống:
intent classification → query reformulation → relevance check → answer generation.
"""

INTENT_CLASSIFICATION_PROMPT = """You are an Assistant with a knowledge database. Classify the user's last query into exactly ONE category.

Conversation: {conversation_text}
User's last query: {last_query}

Categories:
A. Answer directly — ONLY for greetings, thanks, or casual small-talk that needs no information (e.g. "hello", "thank you").
B. Get info from database — ANY question asking for facts, knowledge, definitions, rules, numbers, ranges, procedures, or explanations. This is the DEFAULT for any informational question.
C. Ask for clarification — ONLY when the query is empty, gibberish, or so incomplete that it has no discernible topic at all.

Rules:
- A self-contained factual question (even with no conversation history) is ALWAYS B, never C.
- Do NOT choose C just because the conversation history is empty or because details are missing — try B first.
- When in doubt between B and C, choose B.

Examples:
- "Khoảng điểm từ mấy tới mấy sẽ được quy đổi là C+?" -> B
- "What is the refund policy?" -> B
- "hello there" -> A
- "asdfgh" -> C

Answer with a single letter (A or B or C) only."""

REFORMULATION_PROMPT = """Conversation: {conversation_text}
Last query: {last_query}
Rewrite the query into a clear search question for a knowledge database.
IMPORTANT: Write the search question in the SAME language as the last query. Do NOT translate."""

RELEVANCE_CHECK_PROMPT = """Query: {query}
Context: {context}
Is the context relevant? (A. Yes / B. No)"""

CLARIFICATION_PROMPT = "Given conversation: {conversation_text}\nGenerate a helpful clarification question."

RAG_ANSWER_PROMPT = """Conversation: {conversation_history}
Query: {query}
Context: {context}
Answer based on context."""
