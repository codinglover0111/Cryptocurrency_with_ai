"""LangGraph 기반 거래 워크플로우."""

from __future__ import annotations

from typing import Any, Dict, Optional

from langgraph.graph import START, END, StateGraph

from app.agents import TradingState
from app.agents.decision_agent import DecisionAgent
from app.agents.indicator_agent import IndicatorAgent
from app.agents.pattern_agent import PatternAgent
from app.agents.trend_agent import TrendAgent
from app.config import (
    ADAPTIVE_OPRO_CONFIG,
    AGENT_CONFIG,
    load_runtime_config,
)
from app.opro import AdaptiveOPRO, MetaPromptManager, PerformanceScorer, RegimeDetector

from .llm_factory import create_llm


class TradingGraph:
    """LangGraph 트레이딩 파이프라인."""

    def __init__(
        self,
        compiled,
        *,
        regime_detector: RegimeDetector,
        adaptive_opro: AdaptiveOPRO,
    ) -> None:
        self._compiled = compiled
        self._regime_detector = regime_detector
        self._adaptive_opro = adaptive_opro

    def run(self, state: TradingState) -> TradingState:
        """그래프 실행."""

        working_state: TradingState = dict(state)
        timeframe_data = working_state.get("timeframe_data") or {}
        df_4h = timeframe_data.get("4h")
        df_1h = timeframe_data.get("1h")
        if df_4h is not None and not df_4h.empty:
            primary_df = df_4h
        elif df_1h is not None and not df_1h.empty:
            primary_df = df_1h
        else:
            primary_df = None
        regime = self._regime_detector.detect_regime(primary_df)
        working_state["regime"] = regime
        working_state.setdefault("prompt_trace", [])

        output: TradingState = self._compiled.invoke(working_state)
        return output


def _opro_node(adaptive_opro: AdaptiveOPRO):
    def node(state: TradingState) -> Dict[str, Any]:
        opro_result = adaptive_opro.generate_meta_prompt(state)
        return opro_result

    return node


def build_trading_graph(
    agent_config: Optional[Dict[str, Dict[str, Any]]] = None,
    opro_config: Optional[Dict[str, Any]] = None,
) -> TradingGraph:
    """TradingGraph 인스턴스 생성."""

    runtime_config = load_runtime_config()
    config = agent_config or runtime_config.get("agents") or AGENT_CONFIG
    adaptive_config = opro_config or runtime_config.get("adaptive_opro") or ADAPTIVE_OPRO_CONFIG

    indicator_agent = IndicatorAgent(create_llm(**config["indicator_agent"]))
    pattern_agent = PatternAgent(create_llm(**config["pattern_agent"]))
    trend_agent = TrendAgent(create_llm(**config["trend_agent"]))
    decision_agent = DecisionAgent(create_llm(**config["decision_agent"]))

    regime_detector = RegimeDetector(
        adx_threshold=float(adaptive_config.get("sideways_threshold", 25.0))
    )
    adaptive_opro = AdaptiveOPRO(
        meta_prompt_manager=MetaPromptManager(),
        scorer=PerformanceScorer(),
        optimizer_model=adaptive_config.get("optimizer_model", "openai:gpt-4o-mini"),
        performance_window=int(adaptive_config.get("performance_window", 20)),
        min_trades=int(adaptive_config.get("min_trades_for_update", 5)),
    )

    workflow = StateGraph(TradingState)
    workflow.add_node("adaptive_opro", _opro_node(adaptive_opro))
    workflow.add_node("indicator", indicator_agent)
    workflow.add_node("pattern", pattern_agent)
    workflow.add_node("trend", trend_agent)
    workflow.add_node("decision", decision_agent)

    workflow.add_edge(START, "adaptive_opro")
    workflow.add_edge("adaptive_opro", "indicator")
    workflow.add_edge("indicator", "pattern")
    workflow.add_edge("pattern", "trend")
    workflow.add_edge("trend", "decision")
    workflow.add_edge("decision", END)

    compiled = workflow.compile()
    return TradingGraph(
        compiled,
        regime_detector=regime_detector,
        adaptive_opro=adaptive_opro,
    )

