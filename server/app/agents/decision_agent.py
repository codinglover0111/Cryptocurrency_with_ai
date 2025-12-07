"""Decision Agent 구현."""

from __future__ import annotations

from typing import Dict

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

from .prompt_service import get_prompt
from .schemas import TradeDecision
from .state import TradingState


class DecisionAgent:
    """최종 의사결정 에이전트."""

    def __init__(self, llm: BaseChatModel) -> None:
        # DB에서 프롬프트 로드 (없으면 기본값 사용)
        prompt_template = get_prompt("decision")
        prompt = ChatPromptTemplate.from_template(prompt_template)
        self.chain = prompt | llm.with_structured_output(
            TradeDecision, method="function_calling"
        )

    def __call__(self, state: TradingState) -> Dict[str, TradeDecision]:
        indicator = state.get("indicator")
        pattern = state.get("pattern")
        trend = state.get("trend")

        indicator_summary = indicator.summary if indicator else "지표 분석 없음"
        pattern_summary = pattern.analysis if pattern else "패턴 분석 없음"
        trend_summary = trend.analysis if trend else "추세 분석 없음"

        meta_prompt = state.get("meta_prompt") or "Adaptive-OPRO 메타 프롬프트 생성 중단됨"

        result = self.chain.invoke(
            {
                "indicator_summary": indicator_summary,
                "pattern_summary": pattern_summary,
                "trend_summary": trend_summary,
                "regime": state.get("regime", "unknown"),
                "meta_prompt": meta_prompt,
            }
        )
        return {"decision": result}

