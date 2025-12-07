"""에이전트 출력 스키마."""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class IndicatorResult(BaseModel):
    """기술적 지표 요약."""

    rsi: float = Field(description="RSI 값 (0-100)")
    macd_signal: Literal["bullish", "bearish", "neutral"]
    stochastic: float = Field(description="%K Fast Stochastic (0-100)")
    bollinger_position: Literal["upper", "middle", "lower"]
    momentum_score: float = Field(description="최근 3개 봉 기준 모멘텀 점수 -1~1 범위")
    summary: str = Field(description="지표 종합 분석 (한국어)")


class PatternResult(BaseModel):
    """캔들 패턴 분석."""

    patterns_found: List[str] = Field(description="발견된 패턴 목록")
    pattern_signal: Literal["bullish", "bearish", "neutral"]
    confidence: float = Field(ge=0.0, le=1.0)
    analysis: str = Field(description="패턴 해석 (한국어)")


class TrendResult(BaseModel):
    """추세 분석."""

    trend_direction: Literal["uptrend", "downtrend", "sideways"]
    support_levels: List[float]
    resistance_levels: List[float]
    volatility: float = Field(description="ATR 기반 변동성 (퍼센트)")
    analysis: str = Field(description="추세 해석 (한국어)")


class TradeDecision(BaseModel):
    """최종 매매 결정."""

    status: Literal["long", "short", "hold", "stop"]
    tp: Optional[float] = None
    sl: Optional[float] = None
    leverage: Optional[float] = Field(default=5.0, ge=1, le=75)
    buy_now: bool = False
    close_now: bool = False
    close_percent: Optional[float] = Field(
        default=None, description="현재 포지션 청산 비율 (1-100)"
    )
    explain: str = Field(description="판단 근거 (한국어)")

