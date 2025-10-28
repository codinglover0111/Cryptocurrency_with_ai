from __future__ import annotations

import os
from typing import Dict, Any, Optional
import json
from openai import OpenAI  # type: ignore

import google.generativeai as genai


class AIProvider:
    """Gemini(기본) + OpenAI 호환(DeepSeek/Qwen) 라우팅

    환경변수
      - AI_PROVIDER: gemini | openai
      - GEMINI_API_KEY
      - OPENAI_BASE_URL (예: https://api.deepseek.com/v1 혹은 https://dashscope.aliyuncs.com/compatible-mode/v1)
      - OPENAI_API_KEY
      - OPENAI_MODEL (예: deepseek-reasoner, qwen2.5-72b-instruct 등)
    """

    def __init__(self) -> None:
        self.provider = os.getenv("AI_PROVIDER", "gemini").lower()
        if self.provider == "gemini":
            genai.configure(api_key=os.environ["GEMINI_API_KEY"])  # KeyError 발생 의도
        else:
            # 지연 임포트로 openai 선택시에만 의존성 사용
            base_url = os.environ.get("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
            api_key = os.environ.get("OPENAI_API_KEY")
            if not base_url or not api_key:
                raise RuntimeError("OPENAI_BASE_URL/OPENAI_API_KEY가 필요합니다")
            self._openai_client = OpenAI(base_url=base_url, api_key=f"{api_key}")

    # 텍스트 + 선택 이미지 컨텍스트로 의사결정 프롬프트 수행
    def decide(self, prompt: str, _images: Optional[list] = None) -> str:
        if self.provider == "gemini":
            model = genai.GenerativeModel(
                model_name=os.getenv(
                    "GEMINI_MODEL", "gemini-2.0-flash-thinking-exp-01-21"
                ),
                generation_config={
                    "temperature": 0.2,
                    "top_p": 0.95,
                    "top_k": 64,
                    "max_output_tokens": 8192,
                    "response_mime_type": "text/plain",
                },
            )
            chat = model.start_chat(history=[])
            res = chat.send_message(prompt)
            return res.text
        else:
            # OpenAI 호환
            model = os.environ.get("OPENAI_MODEL", "deepseek-reasoner")
            # 이미지는 이 간단 버전에서 무시
            response = self._openai_client.chat.completions.create(
                model=model, messages=[{"role": "system", "content": prompt}]
            )
            return response.choices[0].message.content

    def decide_json(self, prompt: str) -> Dict[str, Any]:
        """모델이 직접 JSON을 반환하도록 강제.
        - Gemini: response_schema 사용
        - OpenAI 호환: (1) tools(function calling) 또는 (2) response_format=json_object
        """
        if self.provider == "gemini":
            model = genai.GenerativeModel(
                model_name=os.getenv(
                    "GEMINI_MODEL", "gemini-2.0-flash-thinking-exp-01-21"
                ),
                generation_config={
                    "temperature": 0.2,
                    "top_p": 0.95,
                    "top_k": 64,
                    "max_output_tokens": 4096,
                    "response_mime_type": "application/json",
                    "response_schema": genai.protos.Schema(
                        type=genai.protos.Type.OBJECT,
                        required=["Status"],
                        properties={
                            "Status": genai.protos.Schema(
                                type=genai.protos.Type.STRING,
                                enum=["hold", "sell", "buy"],
                            ),
                            "tp": genai.protos.Schema(type=genai.protos.Type.NUMBER),
                            "sl": genai.protos.Schema(type=genai.protos.Type.NUMBER),
                            "price": genai.protos.Schema(type=genai.protos.Type.NUMBER),
                            "buy_now": genai.protos.Schema(
                                type=genai.protos.Type.BOOLEAN
                            ),
                            "stop_order": genai.protos.Schema(
                                type=genai.protos.Type.BOOLEAN
                            ),
                            "leverage": genai.protos.Schema(
                                type=genai.protos.Type.NUMBER
                            ),
                            "explain": genai.protos.Schema(
                                type=genai.protos.Type.STRING
                            ),
                        },
                    ),
                },
            )
            chat = model.start_chat(history=[])
            res = chat.send_message(prompt)
            return json.loads(res.text)

        # OpenAI 호환 경로
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
                                    "enum": ["hold", "sell", "buy"],
                                },
                                "tp": {"type": "number"},
                                "sl": {"type": "number"},
                                "price": {"type": "number"},
                                "buy_now": {"type": "boolean"},
                                "stop_order": {"type": "boolean"},
                                "leverage": {"type": "number"},
                                "explain": {"type": "string"},
                            },
                            "required": ["Status"],
                        },
                    },
                }
            ]
            resp = self._openai_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                tools=tools,
                tool_choice="auto",
                parallel_tool_calls=False,
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
            # 도구 미사용 시 응답 본문에서 JSON 파싱 시도
            content = choice.message.content
            try:
                return json.loads(content)
            except Exception:
                pass
        # response_format 방식
        resp = self._openai_client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "Return strictly a JSON object with keys: Status, price, sl, tp, buy_now, leverage, explain.",
                },
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
        )
        content = resp.choices[0].message.content
        return json.loads(content)
