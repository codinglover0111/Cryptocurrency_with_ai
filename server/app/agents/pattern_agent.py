"""Pattern Agent 구현 (비전)."""

from __future__ import annotations

from typing import Dict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from .prompt_service import get_prompt
from .schemas import PatternResult
from .state import TradingState


def _build_image_content(chart_images: Dict[str, str]) -> list:
    content = []
    for timeframe in ("4h", "1h", "15m"):
        b64 = chart_images.get(timeframe)
        if not b64:
            continue
        content.append({"type": "text", "text": f"{timeframe} 차트"})
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"},
            }
        )
    return content


class PatternAgent:
    """캔들 패턴 분석 에이전트."""

    def __init__(self, llm: BaseChatModel) -> None:
        self.llm = llm.with_structured_output(
            PatternResult, method="function_calling"
        )

    def __call__(self, state: TradingState):
        chart_images = state.get("chart_images") or {}
        indicator = state.get("indicator")
        indicator_summary = indicator.summary if indicator else "지표 분석 없음"

        template_vars = {
            "symbol": state.get("symbol", "UNKNOWN"),
            "regime": state.get("regime", "unknown"),
            "indicator_summary": indicator_summary,
        }

        # DB에서 프롬프트 로드 (없으면 기본값 사용)
        pattern_template = get_prompt("pattern")

        system_prompt = (
            "당신은 캔들 패턴 및 비전 차트 분석 전문가입니다. 응답은 한국어입니다."
        )
        human_content = [
            {
                "type": "text",
                "text": pattern_template.format(**template_vars),
            }
        ]
        human_content.extend(_build_image_content(chart_images))

        result = self.llm.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_content),
            ]
        )
        return {"pattern": result}

