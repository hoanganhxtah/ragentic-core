"""
LangChain chat-model factory (native tool-calling).

This module is SEPARATE from the text-only `app/llm/factory.py::get_llm_client()`.
While `get_llm_client()` returns the project's custom text-only `LLMClient`
(exposing `generate(prompt, system_instruction)`), `get_chat_model()` here returns
a LangChain `BaseChatModel` that supports native function/tool calling
(`bind_tools()`), which the ReAct agent requires.

The provider is selected from the same `.env`-backed `llm_settings.PROVIDER`
settings used by `get_llm_client()`, and the branch structure mirrors that
factory — including LAZY per-branch imports so unused provider integration
packages are not required at import time.
"""

from __future__ import annotations

import logging

from langchain_core.language_models.chat_models import BaseChatModel

from app.config.app_settings import llm_settings, api_key_settings

_log = logging.getLogger(__name__)


def get_chat_model() -> BaseChatModel:
    """
    Factory: build a LangChain chat model (native tool-calling) from .env.

    Separate from `get_llm_client()` which returns the text-only custom client.
    The provider is selected from `llm_settings.PROVIDER`; API keys/credentials
    are resolved from `llm_settings.API_KEY` falling back to the provider-specific
    values in `api_key_settings`.

    Supported providers:
    - gemini/google
    - openai/openai-compatible (incl. self-host via LLM_BASE_URL)
    - anthropic/claude
    - groq
    - bedrock/aws

    Returns:
        A `BaseChatModel` instance that supports `bind_tools()`. Never returns None.

    Raises:
        ValueError: when `llm_settings.PROVIDER` is not a supported provider.
    """
    provider = llm_settings.PROVIDER.lower()
    model = llm_settings.MODEL
    temp = llm_settings.TEMPERATURE
    maxtok = llm_settings.MAX_TOKENS

    _log.debug("Chat model config: provider=%s | model=%s", provider, model)

    if provider in {"gemini", "google"}:
        from langchain_google_genai import ChatGoogleGenerativeAI

        key = llm_settings.API_KEY or api_key_settings.GEMINI_API_KEY
        return ChatGoogleGenerativeAI(
            model=model,
            google_api_key=key,
            temperature=temp,
            max_tokens=maxtok,
        )

    if provider in {"openai", "oai", "openai-compatible", "openai_compatible"}:
        from langchain_openai import ChatOpenAI

        key = llm_settings.API_KEY or api_key_settings.OPENAI_API_KEY
        return ChatOpenAI(
            model=model,
            api_key=key,
            base_url=llm_settings.BASE_URL,
            temperature=temp,
            max_tokens=maxtok,
        )

    if provider in {"anthropic", "claude"}:
        from langchain_anthropic import ChatAnthropic

        key = llm_settings.API_KEY or api_key_settings.ANTHROPIC_API_KEY
        return ChatAnthropic(
            model=model,
            api_key=key,
            temperature=temp,
            max_tokens=maxtok,
        )

    if provider == "groq":
        from langchain_groq import ChatGroq

        key = llm_settings.API_KEY or api_key_settings.GROQ_API_KEY
        return ChatGroq(
            model=model,
            api_key=key,
            temperature=temp,
            max_tokens=maxtok,
        )

    if provider in {"bedrock", "aws", "aws-bedrock"}:
        from langchain_aws import ChatBedrockConverse

        return ChatBedrockConverse(
            model=model,
            region_name=api_key_settings.AWS_REGION,
            temperature=temp,
            max_tokens=maxtok,
            aws_access_key_id=api_key_settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=api_key_settings.AWS_SECRET_ACCESS_KEY,
            aws_session_token=api_key_settings.AWS_SESSION_TOKEN,
        )

    raise ValueError(
        f"Unsupported LLM_PROVIDER for chat model: {provider!r}. "
        "Use: gemini (google) | openai | openai-compatible | anthropic (claude) | "
        "groq | bedrock (aws)"
    )
