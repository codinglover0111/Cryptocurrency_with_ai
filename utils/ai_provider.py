# pylint: disable=broad-except
# ruff: noqa: E722, BLE001
from __future__ import annotations

import base64
import binascii
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

import google.generativeai as genai
from openai import OpenAI  # type: ignore

try:  # pragma: no cover - optional dependency
    from google.api_core.exceptions import ResourceExhausted, TooManyRequests
except ImportError:  # pragma: no cover - the module may not be available at runtime
    ResourceExhausted = None  # type: ignore
    TooManyRequests = None  # type: ignore


LOGGER = logging.getLogger(__name__)


class AIProvider:
    """Gemini(기본) + OpenAI 호환(DeepSeek/Qwen) 라우팅.

    환경변수
      - AI_PROVIDER: gemini | openai
      - GEMINI_API_KEY
      - GEMINI_MODEL (optional)
      - GEMINI_RETRY_WAIT_SECONDS (optional, default 300)
      - GEMINI_MAX_RETRIES (optional, default 3)
      - OPENAI_BASE_URL (예: https://api.deepseek.com/v1 혹은 https://dashscope.aliyuncs.com/compatible-mode/v1)
      - OPENAI_API_KEY
      - OPENAI_MODEL (예: deepseek-reasoner, qwen2.5-72b-instruct 등)
    """

    def __init__(self) -> None:
        raw_provider = os.getenv("AI_PROVIDER", "gemini")
        provider_key = raw_provider.lower()
        if provider_key not in {"gemini", "openai"}:
            provider_key = "openai"
        self.provider = provider_key

        if self.provider == "gemini":
            genai.configure(api_key=os.environ["GEMINI_API_KEY"])  # KeyError 발생 의도
            self._gemini_model_name = os.getenv(
                "GEMINI_MODEL", "gemini-2.0-flash-thinking-exp-01-21"
            )
            self._gemini_retry_wait = max(
                0, int(os.environ.get("GEMINI_RETRY_WAIT_SECONDS", "300"))
            )
            self._gemini_max_retries = max(
                1, int(os.environ.get("GEMINI_MAX_RETRIES", "3"))
            )
        else:
            base_url = os.environ.get("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
            api_key = os.environ.get("OPENAI_API_KEY")
            if not base_url or not api_key:
                raise RuntimeError("OPENAI_BASE_URL/OPENAI_API_KEY가 필요합니다")
            self._openai_client = OpenAI(base_url=base_url, api_key=f"{api_key}")

    def decide(self, prompt: str, images: Optional[List[Dict[str, Any]]] = None) -> str:
        """텍스트/이미지 입력을 받아 의사결정 응답을 문자열로 반환."""
        if self.provider == "gemini":
            response = self._call_gemini(
                prompt,
                images=images,
                response_mime_type="text/plain",
                max_output_tokens=8192,
            )
            return self._extract_gemini_text(response)

        model = os.environ.get("OPENAI_MODEL", "deepseek-reasoner")
        response = self._openai_client.chat.completions.create(
            model=model, messages=[{"role": "system", "content": prompt}]
        )
        return response.choices[0].message.content

    def decide_json(
        self, prompt: str, images: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """모델이 직접 JSON을 반환하도록 강제."""
        if self.provider == "gemini":
            response = self._call_gemini(
                prompt,
                images=images,
                response_mime_type="application/json",
                max_output_tokens=4096,
                response_schema=genai.protos.Schema(
                    type=genai.protos.Type.OBJECT,
                    required=["Status"],
                    properties={
                        "Status": genai.protos.Schema(
                            type=genai.protos.Type.STRING,
                            enum=["hold", "short", "long", "stop"],
                        ),
                        "tp": genai.protos.Schema(type=genai.protos.Type.NUMBER),
                        "sl": genai.protos.Schema(type=genai.protos.Type.NUMBER),
                        "price": genai.protos.Schema(type=genai.protos.Type.NUMBER),
                        "buy_now": genai.protos.Schema(type=genai.protos.Type.BOOLEAN),
                        "stop_order": genai.protos.Schema(
                            type=genai.protos.Type.BOOLEAN
                        ),
                        "leverage": genai.protos.Schema(type=genai.protos.Type.NUMBER),
                        "close_now": genai.protos.Schema(
                            type=genai.protos.Type.BOOLEAN
                        ),
                        "close_percent": genai.protos.Schema(
                            type=genai.protos.Type.NUMBER
                        ),
                        "reduce_only": genai.protos.Schema(
                            type=genai.protos.Type.BOOLEAN
                        ),
                        "update_existing": genai.protos.Schema(
                            type=genai.protos.Type.BOOLEAN
                        ),
                        "explain": genai.protos.Schema(type=genai.protos.Type.STRING),
                    },
                ),
            )
            content = self._extract_gemini_text(response)
            if not content:
                raise RuntimeError("Gemini JSON 응답이 비었습니다.")
            return json.loads(content)

        model = os.environ.get("OPENAI_MODEL", "deepseek-reasoner")
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
                extra_body={
                    "provider": {
                        "order": ["google-vertex", "fireworks"],
                        "allow_fallbacks": True,
                    }
                },
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
        return json.loads(content)

    def confirm_trade_json(self, prompt: str) -> Dict[str, Any]:
        """TP/SL, 가격, 레버리지 제안에 대해 확인/수정 여부를 JSON으로 반환."""
        if self.provider == "gemini":
            response = self._call_gemini(
                prompt,
                images=None,
                response_mime_type="application/json",
                max_output_tokens=2048,
                response_schema=genai.protos.Schema(
                    type=genai.protos.Type.OBJECT,
                    required=["confirm"],
                    properties={
                        "confirm": genai.protos.Schema(type=genai.protos.Type.BOOLEAN),
                        "tp": genai.protos.Schema(type=genai.protos.Type.NUMBER),
                        "sl": genai.protos.Schema(type=genai.protos.Type.NUMBER),
                        "price": genai.protos.Schema(type=genai.protos.Type.NUMBER),
                        "buy_now": genai.protos.Schema(type=genai.protos.Type.BOOLEAN),
                        "leverage": genai.protos.Schema(type=genai.protos.Type.NUMBER),
                        "explain": genai.protos.Schema(type=genai.protos.Type.STRING),
                    },
                ),
            )
            content = self._extract_gemini_text(response)
            if not content:
                raise RuntimeError("Gemini JSON 응답이 비었습니다.")
            return json.loads(content)

        model = os.environ.get("OPENAI_MODEL", "deepseek-reasoner")
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
                extra_body={
                    "provider": {
                        "order": ["google-vertex", "fireworks"],
                        "allow_fallbacks": True,
                    }
                },
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
        return json.loads(content)

    def _call_gemini(
        self,
        prompt: str,
        *,
        images: Optional[List[Dict[str, Any]]],
        response_mime_type: str,
        max_output_tokens: int,
        response_schema: Optional[Any] = None,
    ) -> Any:
        generation_config: Dict[str, Any] = {
            "temperature": 0.2,
            "top_p": 0.95,
            "top_k": 64,
            "max_output_tokens": max_output_tokens,
            "response_mime_type": response_mime_type,
        }
        if response_schema is not None:
            generation_config["response_schema"] = response_schema

        model = genai.GenerativeModel(
            model_name=self._gemini_model_name,
            generation_config=generation_config,
        )

        last_error: Optional[Exception] = None
        for attempt in range(1, self._gemini_max_retries + 1):
            try:
                parts = self._build_gemini_parts(prompt, images)
                return model.generate_content([{"role": "user", "parts": parts}])
            except Exception as exc:
                if self._is_gemini_rate_limit(exc):
                    last_error = exc
                    wait = self._gemini_retry_wait or 300
                    LOGGER.warning(
                        "Gemini rate limit 감지. %s초 후 재시도 (%s/%s)",
                        wait,
                        attempt,
                        self._gemini_max_retries,
                    )
                    if attempt >= self._gemini_max_retries:
                        raise
                    if wait > 0:
                        time.sleep(wait)
                    continue
                raise

        if last_error is not None:
            raise last_error
        raise RuntimeError("Gemini 호출이 연속 실패했습니다.")

    def _build_gemini_parts(
        self, prompt: str, images: Optional[List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        parts: List[Dict[str, Any]] = [{"text": prompt}]
        for image in images or []:
            b64 = image.get("b64")
            if not b64:
                continue
            try:
                data = base64.b64decode(b64)
            except (binascii.Error, ValueError) as exc:  # pragma: no cover - 로그 용도
                LOGGER.warning("Gemini 이미지 디코딩 실패: %s", exc)
                continue
            mime = image.get("mime", "image/png")
            parts.append({"mime_type": mime, "data": data})
        return parts

    @staticmethod
    def _is_gemini_rate_limit(exc: Exception) -> bool:
        if ResourceExhausted is not None and isinstance(exc, ResourceExhausted):
            return True
        if TooManyRequests is not None and isinstance(exc, TooManyRequests):
            return True
        message = str(exc).lower()
        return (
            "429" in message
            or "rate limit" in message
            or "quota" in message
            or "too many requests" in message
        )

    @staticmethod
    def _extract_gemini_text(response: Any) -> str:
        text = getattr(response, "text", None)
        if isinstance(text, str) and text.strip():
            return text

        candidates = getattr(response, "candidates", None)
        if not candidates:
            return ""

        for candidate in candidates:
            content = getattr(candidate, "content", None)
            parts = getattr(content, "parts", None) if content is not None else None
            texts: List[str] = []
            for part in parts or []:
                part_text = getattr(part, "text", None)
                if isinstance(part_text, str) and part_text:
                    texts.append(part_text)
            combined = "".join(texts).strip()
            if combined:
                return combined
        return ""
