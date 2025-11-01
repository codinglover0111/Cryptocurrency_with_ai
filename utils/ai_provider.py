# pylint: disable=broad-except
# ruff: noqa: E722, BLE001
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from openai import OpenAI  # type: ignore


class AIProvider:
    """OpenAI 호환 챗 컴플리션 클라이언트를 감싼 헬퍼.

    환경변수
      - OPENAI_API_KEY (필수)
      - OPENAI_BASE_URL (선택, 미설정 시 OpenAI 기본 엔드포인트 사용)
      - OPENAI_MODEL (선택, 기본 deepseek-reasoner)
      - OPENAI_TOOLCALL (선택, "1"이면 function call 사용)
    """

    def __init__(self) -> None:
        self.provider = "openai"

        base_url = os.environ.get("OPENAI_BASE_URL")
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY가 필요합니다")

        if base_url:
            self._openai_client = OpenAI(base_url=base_url, api_key=api_key)
        else:
            self._openai_client = OpenAI(api_key=api_key)

    @property
    def client(self) -> OpenAI:
        return self._openai_client

    @staticmethod
    def _current_model() -> str:
        return os.environ.get("OPENAI_MODEL", "deepseek-reasoner")

    def decide(
        self, prompt: str, _images: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """텍스트(및 호환되는 이미지 입력)를 받아 자유형 응답을 반환."""

        # _images 파라미터는 호환성 유지를 위해 남겨두지만 OpenAI 경로에서는 사용하지 않습니다.
        model = self._current_model()
        response = self._openai_client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": prompt}],
        )
        return response.choices[0].message.content or ""

    def decide_json(
        self, prompt: str, _images: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """JSON 구조로 거래 결정을 반환하도록 강제."""

        # _images 파라미터는 인터페이스 호환 목적으로만 유지됩니다.
        model = self._current_model()
        use_tools = os.getenv("OPENAI_TOOLCALL", "0") == "1"

        if use_tools:
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "decide_trade",
                        "description": "Decide trade action and parameters from OHLCV CSV context.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "Status": {
                                    "type": "string",
                                    "enum": ["hold", "short", "long", "stop"],
                                },
                                "tp": {"type": "number"},
                                "sl": {"type": "number"},
                                "price": {"type": "number"},
                                "buy_now": {"type": "boolean"},
                                "stop_order": {"type": "boolean"},
                                "leverage": {"type": "number"},
                                "close_now": {"type": "boolean"},
                                "close_percent": {"type": "number"},
                                "reduce_only": {"type": "boolean"},
                                "explain": {"type": "string"},
                            },
                            "required": ["Status"],
                        },
                    },
                }
            ]
            resp = self._openai_client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": prompt}],
                tools=tools,
                tool_choice="auto",
                parallel_tool_calls=False,
                response_format={"type": "json_object"},
            )
            choice = resp.choices[0]
            tool_calls = (
                getattr(choice.message, "tool_calls", None) or choice.message.tool_calls
                if hasattr(choice.message, "tool_calls")
                else None
            )
            if tool_calls:
                args = tool_calls[0].function.arguments
                return json.loads(args)
            content = choice.message.content
            if content:
                try:
                    return json.loads(content)
                except Exception:
                    pass

        resp = self._openai_client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "Return strictly a JSON object with keys: Status, price, sl, tp, buy_now, leverage, explain. Status must be one of [hold, short, long, stop]",
                },
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
        )
        content = resp.choices[0].message.content
        return json.loads(content or "{}")

    def confirm_trade_json(self, prompt: str) -> Dict[str, Any]:
        """주요 주문 파라미터 확인/수정 여부를 JSON으로 반환."""

        model = self._current_model()
        use_tools = os.getenv("OPENAI_TOOLCALL", "0") == "1"

        if use_tools:
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "confirm_trade",
                        "description": "Confirm or adjust TP/SL and order params.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "confirm": {"type": "boolean"},
                                "tp": {"type": "number"},
                                "sl": {"type": "number"},
                                "price": {"type": "number"},
                                "buy_now": {"type": "boolean"},
                                "leverage": {"type": "number"},
                                "explain": {"type": "string"},
                            },
                            "required": ["confirm"],
                        },
                    },
                }
            ]
            resp = self._openai_client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": prompt}],
                tools=tools,
                tool_choice="auto",
                parallel_tool_calls=False,
                response_format={"type": "json_object"},
            )
            choice = resp.choices[0]
            tool_calls = (
                getattr(choice.message, "tool_calls", None) or choice.message.tool_calls
                if hasattr(choice.message, "tool_calls")
                else None
            )
            if tool_calls:
                args = tool_calls[0].function.arguments
                return json.loads(args)
            content = choice.message.content
            if content:
                try:
                    return json.loads(content)
                except Exception:
                    pass

        resp = self._openai_client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "Return strictly a JSON object with keys: confirm (boolean), tp, sl, price, buy_now, leverage, explain. Only 'confirm' is required.",
                },
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
        )
        content = resp.choices[0].message.content
        return json.loads(content or "{}")
