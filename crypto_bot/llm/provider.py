# pylint: disable=broad-except
# ruff: noqa: E722, BLE001
from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from openai import OpenAI

LOGGER = logging.getLogger(__name__)

Message = Mapping[str, Any]


class AIProvider:
    """OpenAI Responses API wrapper used across the trading workflows."""

    def __init__(self) -> None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY가 필요합니다")
        base_url = os.getenv("OPENAI_BASE_URL") or None
        self._client = OpenAI(api_key=api_key, base_url=base_url)

        self.provider = "openai"
        self._default_model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
        self._temperature = float(os.getenv("OPENAI_TEMPERATURE", "0.2"))
        self._max_output_tokens = int(os.getenv("OPENAI_MAX_OUTPUT_TOKENS", "2048"))

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------
    def decide(self, prompt: str) -> str:
        """Return a plain-text answer for review/journal usage."""
        messages: List[Message] = [
            {"role": "system", "content": "You are a concise trading analyst."},
            {"role": "user", "content": prompt},
        ]
        return self._request_text(messages).strip()

    def decide_json(self, prompt: str) -> Dict[str, Any]:
        """Return a structured trade decision JSON payload."""
        schema = _decision_schema()
        messages: List[Message] = [
            {
                "role": "system",
                "content": (
                    "Return a JSON object describing a single trading decision. "
                    "Do not include any additional text."
                ),
            },
            {"role": "user", "content": prompt},
        ]
        return self.request_json(messages, schema=schema)

    def confirm_trade_json(self, prompt: str) -> Dict[str, Any]:
        """Return confirmation/adjustment data for an order proposal."""
        schema = {
            "type": "object",
            "properties": {
                "confirm": {"type": "boolean"},
                "tp": {"type": ["number", "null"]},
                "sl": {"type": ["number", "null"]},
                "price": {"type": ["number", "null"]},
                "buy_now": {"type": ["boolean", "null"]},
                "leverage": {"type": ["number", "null"]},
                "explain": {"type": ["string", "null"]},
            },
            "required": ["confirm"],
            "additionalProperties": False,
        }
        messages: List[Message] = [
            {
                "role": "system",
                "content": (
                    "Return a JSON object describing whether to confirm the proposed order. "
                    "If confirm is false, include an explain field describing why."
                ),
            },
            {"role": "user", "content": prompt},
        ]
        return self.request_json(messages, schema=schema)

    # ------------------------------------------------------------------
    # Core request methods
    # ------------------------------------------------------------------
    def request_json(
        self,
        messages: Sequence[Message],
        *,
        schema: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        response_format = schema if schema is not None else {"type": "json_object"}
        response = self._client.responses.create(
            model=model or self._default_model,
            input=list(messages),
            temperature=self._resolve_temperature(temperature),
            max_output_tokens=max_output_tokens or self._max_output_tokens,
            response_format=_wrap_schema(response_format),
        )
        text = self._extract_text(response)
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:  # pragma: no cover - defensive logging
            LOGGER.error("OpenAI JSON 파싱 실패: %s", exc)
            LOGGER.debug("OpenAI 응답 텍스트: %s", text)
            raise

    def _request_text(
        self,
        messages: Sequence[Message],
        *,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
    ) -> str:
        response = self._client.responses.create(
            model=model or self._default_model,
            input=list(messages),
            temperature=self._resolve_temperature(temperature),
            max_output_tokens=max_output_tokens or self._max_output_tokens,
        )
        return self._extract_text(response)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _resolve_temperature(self, override: Optional[float]) -> float:
        if override is None:
            return self._temperature
        try:
            return float(override)
        except Exception:  # pragma: no cover - defensive
            return self._temperature

    @staticmethod
    def _extract_text(response: Any) -> str:
        text = getattr(response, "output_text", None)
        if isinstance(text, str) and text:
            return text
        output = getattr(response, "output", None)
        if isinstance(output, Iterable):
            fragments: List[str] = []
            for block in output:
                content = getattr(block, "content", None)
                if not isinstance(content, Iterable):
                    continue
                for item in content:
                    item_text = getattr(item, "text", None)
                    if isinstance(item_text, str):
                        fragments.append(item_text)
            if fragments:
                return "".join(fragments)
        raise RuntimeError("예상치 못한 OpenAI 응답 형식입니다")


def _wrap_schema(schema_or_format: Dict[str, Any]) -> Dict[str, Any]:
    schema_type = (
        schema_or_format.get("type") if isinstance(schema_or_format, dict) else None
    )
    if schema_type in {"json_schema", "json_object"}:
        return schema_or_format
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "structured_output",
            "schema": schema_or_format,
            "strict": True,
        },
    }


def _decision_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "Status": {
                "type": "string",
                "enum": ["hold", "short", "long", "stop"],
            },
            "order_type": {
                "type": "string",
                "enum": ["market", "limit"],
            },
            "price": {"type": ["number", "null"]},
            "tp": {"type": ["number", "null"]},
            "sl": {"type": ["number", "null"]},
            "buy_now": {"type": ["boolean", "null"]},
            "leverage": {"type": ["number", "null"]},
            "close_now": {"type": ["boolean", "null"]},
            "close_percent": {"type": ["number", "null"]},
            "update_existing": {"type": ["boolean", "null"]},
            "explain": {"type": "string"},
        },
        "required": ["Status", "explain"],
        "additionalProperties": False,
    }
