"""Indicator Agent 구현."""

from __future__ import annotations

import json
from typing import Any, Dict

import pandas as pd
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD
from ta.volatility import BollingerBands

from .prompt_service import get_prompt
from .schemas import IndicatorResult
from .state import TradingState


def _compute_indicator_block(df: pd.DataFrame) -> Dict[str, Any]:
    if df is None or df.empty or len(df) < 5:
        return {
            "rsi": 50.0,
            "macd": {"value": 0.0, "signal": 0.0, "hist": 0.0},
            "stochastic": 50.0,
            "bollinger_position": "middle",
            "momentum_score": 0.0,
        }

    close = df["close"]
    high = df["high"]
    low = df["low"]

    rsi = RSIIndicator(close=close, window=14).rsi().iloc[-1]
    macd_indicator = MACD(close=close)
    macd = macd_indicator.macd().iloc[-1]
    macd_signal = macd_indicator.macd_signal().iloc[-1]
    macd_hist = macd - macd_signal
    stochastic = StochasticOscillator(high=high, low=low, close=close).stoch().iloc[-1]
    boll = BollingerBands(close=close)
    upper = boll.bollinger_hband().iloc[-1]
    lower = boll.bollinger_lband().iloc[-1]
    mid = boll.bollinger_mavg().iloc[-1]
    latest_close = close.iloc[-1]

    if latest_close >= upper:
        bb_position = "upper"
    elif latest_close <= lower:
        bb_position = "lower"
    else:
        bb_position = "middle"

    try:
        momentum_score = float(
            (
                (close.iloc[-1] - close.iloc[-2]) / max(close.iloc[-2], 1e-8)
                + (close.iloc[-2] - close.iloc[-3]) / max(close.iloc[-3], 1e-8)
            )
            / 2
        )
    except Exception:
        momentum_score = 0.0

    macd_signal_tag = "bullish"
    if macd_hist < -0.0005:
        macd_signal_tag = "bearish"
    elif abs(macd_hist) <= 0.0005:
        macd_signal_tag = "neutral"

    return {
        "rsi": float(rsi),
        "macd": {
            "value": float(macd),
            "signal": float(macd_signal),
            "hist": float(macd_hist),
        },
        "macd_signal_tag": macd_signal_tag,
        "stochastic": float(stochastic),
        "bollinger": {
            "upper": float(upper),
            "lower": float(lower),
            "middle": float(mid),
            "position": bb_position,
        },
        "momentum_score": momentum_score,
        "bollinger_position": bb_position,
    }


class IndicatorAgent:
    """기술적 지표 분석 에이전트."""

    def __init__(self, llm: BaseChatModel) -> None:
        # DB에서 프롬프트 로드 (없으면 기본값 사용)
        prompt_template = get_prompt("indicator")
        prompt = ChatPromptTemplate.from_template(prompt_template)
        self.chain = prompt | llm.with_structured_output(
            IndicatorResult, method="function_calling"
        )

    def __call__(self, state: TradingState) -> Dict[str, IndicatorResult]:
        timeframe_data = state.get("timeframe_data") or {}
        df_1h = timeframe_data.get("1h")
        df_4h = timeframe_data.get("4h")
        if df_1h is not None and not df_1h.empty:
            df = df_1h
        elif df_4h is not None and not df_4h.empty:
            df = df_4h
        else:
            df = None
        metrics = _compute_indicator_block(df)

        positions = state.get("positions") or []
        position_summary = (
            "\n".join(
                f"{p.get('symbol', '?')} {p.get('side', '?')} size={p.get('contracts') or p.get('size')}"
                for p in positions[:5]
            )
            or "포지션 없음"
        )

        result = self.chain.invoke(
            {
                "symbol": state.get("symbol", "UNKNOWN"),
                "regime": state.get("regime", "unknown"),
                "indicator_block": json.dumps(metrics, ensure_ascii=False, indent=2),
                "position_summary": position_summary,
            }
        )

        result.bollinger_position = metrics["bollinger"]["position"]  # type: ignore[attr-defined]
        result.macd_signal = metrics["macd_signal_tag"]  # type: ignore[attr-defined]
        result.stochastic = float(metrics["stochastic"])
        result.rsi = float(metrics["rsi"])
        result.momentum_score = float(metrics["momentum_score"])
        return {"indicator": result}
