# pylint: disable=broad-except
# ruff: noqa: E722, BLE001
from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

from openai import OpenAI  # type: ignore


LOGGER = logging.getLogger(__name__)


class AIProvider:
    """OpenAI 호환 SDK 사용 (DeepSeek/Qwen 등).

    환경변수
      - OPENAI_BASE_URL (예: https://api.deepseek.com/v1 혹은 https://dashscope.aliyuncs.com/compatible-mode/v1)
      - OPENAI_API_KEY
      - OPENAI_MODEL (예: deepseek-reasoner, qwen2.5-72b-instruct 등)
      - OPENAI_TOOLCALL (optional, "1" to enable tool calling)
    """

    def __init__(self) -> None:
        base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        api_key = os.environ.get("OPENAI_API_KEY")
        if not base_url or not api_key:
            raise RuntimeError("OPENAI_BASE_URL/OPENAI_API_KEY가 필요합니다")
        self._openai_client = OpenAI(base_url=base_url, api_key=f"{api_key}")

    def decide(self, prompt: str, images: Optional[List[Dict[str, Any]]] = None) -> str:
        """텍스트 입력을 받아 의사결정 응답을 문자열로 반환.

        Args:
            prompt: LLM에 전달할 프롬프트 텍스트
            images: 미사용 (호환성을 위해 유지)

        Returns:
            LLM의 텍스트 응답
        """
        # images 파라미터는 호환성을 위해 유지하지만 현재 사용하지 않음
        _ = images
        model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        response = self._openai_client.chat.completions.create(
            model=model, messages=[{"role": "system", "content": prompt}]
        )
        return response.choices[0].message.content or ""

    def decide_json(
        self, prompt: str, images: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """모델이 직접 JSON을 반환하도록 강제.

        Args:
            prompt: LLM에 전달할 프롬프트 텍스트
            images: 미사용 (호환성을 위해 유지)

        Returns:
            파싱된 JSON 딕셔너리
        """
        # images 파라미터는 호환성을 위해 유지하지만 현재 사용하지 않음
        _ = images
        model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
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
                                "update_existing": {"type": "boolean"},
                                "explain": {"type": "string"},
                            },
                            "required": ["Status", "explain"],
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
            try:
                return json.loads(content)
            except Exception:
                pass

        resp = self._openai_client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "Return strictly a JSON object with keys: Status, price, sl, tp, buy_now, leverage, explain, update_existing. Status must be one of [hold, short, long, stop]",
                },
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
        )
        content = resp.choices[0].message.content
        if not content:
            raise RuntimeError("OpenAI JSON 응답이 비었습니다.")
        return json.loads(content)

    def confirm_trade_json(self, prompt: str) -> Dict[str, Any]:
        """TP/SL, 가격, 레버리지 제안에 대해 확인/수정 여부를 JSON으로 반환."""
        model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
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
        if not content:
            raise RuntimeError("OpenAI JSON 응답이 비었습니다.")
        return json.loads(content)
