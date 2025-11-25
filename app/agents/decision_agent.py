"""Decision Agent 구현."""

from __future__ import annotations

from typing import Dict

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

from .prompts import DECISION_TEMPLATE
from .schemas import TradeDecision
from .state import TradingState


class DecisionAgent:
    """최종 의사결정 에이전트."""

    def __init__(self, llm: BaseChatModel) -> None:
        prompt = ChatPromptTemplate.from_template(DECISION_TEMPLATE)
        self.chain = prompt | llm.with_structured_output(TradeDecision, strict=True)

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

