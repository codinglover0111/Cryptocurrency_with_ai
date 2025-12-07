"""LangGraph 상태 정의."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, TypedDict

import pandas as pd

from .schemas import IndicatorResult, PatternResult, TrendResult, TradeDecision


class TradingState(TypedDict, total=False):
    """멀티 에이전트 파이프라인 공유 상태."""

    symbol: str
    spot_symbol: str
    timeframe_data: Dict[str, pd.DataFrame]
    chart_images: Dict[str, str]
    positions: List[Dict[str, Any]]
    context_blocks: Dict[str, str]
    regime: Optional[Literal["bullish", "bearish", "sideways", "unknown"]]
    meta_prompt: Optional[str]
    prompt_trace: List[Dict[str, Any]]
    indicator: IndicatorResult
    pattern: PatternResult
    trend: TrendResult
    decision: TradeDecision
    performance_snapshot: Dict[str, Any]

