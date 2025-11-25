"""LLM 팩토리."""

from __future__ import annotations

import os
from typing import Any, Dict

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI


class LLMConfigurationError(RuntimeError):
    """LLM 설정 오류."""


def create_llm(provider: str, model: str, temperature: float = 0.1, **extra: Any) -> BaseChatModel:
    """Provider별 ChatModel 생성."""

    provider_key = (provider or "").lower()

    if provider_key in {"openai", "azure"}:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise LLMConfigurationError("OPENAI_API_KEY가 설정되지 않았습니다.")
        return ChatOpenAI(model=model, temperature=temperature, api_key=api_key)

    if provider_key == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise LLMConfigurationError("OPENROUTER_API_KEY가 필요합니다.")
        base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            base_url=base_url,
            api_key=api_key,
        )

    if provider_key == "gemini":
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise LLMConfigurationError("GEMINI_API_KEY가 필요합니다.")
        return ChatGoogleGenerativeAI(model=model, temperature=temperature, api_key=api_key)

    if provider_key == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise LLMConfigurationError("ANTHROPIC_API_KEY가 필요합니다.")
        return ChatAnthropic(model=model, temperature=temperature, api_key=api_key)

    raise LLMConfigurationError(f"지원되지 않는 provider: {provider}")

