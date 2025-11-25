"""Trend Agent 구현."""

from __future__ import annotations

from typing import Dict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from .prompts import TREND_TEMPLATE
from .schemas import TrendResult
from .state import TradingState


def _image_payload(chart_images: Dict[str, str]) -> list:
    payload = []
    for timeframe in ("4h", "1h"):
        b64 = chart_images.get(timeframe)
        if not b64:
            continue
        payload.append({"type": "text", "text": f"{timeframe} 지지/저항 시각화"})
        payload.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"},
            }
        )
    return payload


class TrendAgent:
    """추세/지지저항 분석 에이전트."""

    def __init__(self, llm: BaseChatModel) -> None:
        self.llm = llm.with_structured_output(
            TrendResult, method="function_calling"
        )

    def __call__(self, state: TradingState):
        chart_images = state.get("chart_images") or {}
        pattern = state.get("pattern")
        pattern_summary = pattern.analysis if pattern else "패턴 분석 없음"

        human_content = [
            {
                "type": "text",
                "text": TREND_TEMPLATE.format(
                    symbol=state.get("symbol", "UNKNOWN"),
                    regime=state.get("regime", "unknown"),
                    pattern_summary=pattern_summary,
                ),
            }
        ]
        human_content.extend(_image_payload(chart_images))

        result = self.llm.invoke(
            [
                SystemMessage(
                    content="당신은 전문 추세 분석가입니다. 모든 답변은 한국어입니다."
                ),
                HumanMessage(content=human_content),
            ]
        )
        return {"trend": result}

