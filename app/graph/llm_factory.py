"""LLM 팩토리."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI


class LLMConfigurationError(RuntimeError):
    """LLM 설정 오류."""


def _get_llm_api_key(provider: str) -> Optional[str]:
    """DB에서 LLM API 키를 가져오고, 없으면 환경변수에서 폴백합니다."""
    try:
        from app.auth.api_keys import get_llm_key
        key = get_llm_key(provider)
        if key:
            return key
    except Exception:
        pass
    
    # 환경변수 폴백
    env_map = {
        "openai": "OPENAI_API_KEY",
        "azure": "OPENAI_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
        "gemini": "GEMINI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
    }
    env_name = env_map.get(provider.lower())
    if env_name:
        return os.getenv(env_name)
    return None


def create_llm(provider: str, model: str, temperature: float = 0.1, **extra: Any) -> BaseChatModel:
    """Provider별 ChatModel 생성."""

    provider_key = (provider or "").lower()

    if provider_key in {"openai", "azure"}:
        api_key = _get_llm_api_key("openai")
        if not api_key:
            raise LLMConfigurationError("OPENAI_API_KEY가 설정되지 않았습니다.")
        return ChatOpenAI(model=model, temperature=temperature, api_key=api_key)

    if provider_key == "openrouter":
        api_key = _get_llm_api_key("openrouter")
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
        api_key = _get_llm_api_key("gemini")
        if not api_key:
            raise LLMConfigurationError("GEMINI_API_KEY가 필요합니다.")
        return ChatGoogleGenerativeAI(model=model, temperature=temperature, api_key=api_key)

    if provider_key == "anthropic":
        api_key = _get_llm_api_key("anthropic")
        if not api_key:
            raise LLMConfigurationError("ANTHROPIC_API_KEY가 필요합니다.")
        return ChatAnthropic(model=model, temperature=temperature, api_key=api_key)

    raise LLMConfigurationError(f"지원되지 않는 provider: {provider}")

