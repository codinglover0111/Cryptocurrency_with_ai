"""Adaptive-OPRO Scorer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List


@dataclass
class ScoreSummary:
    roi: float
    sharpe: float
    max_drawdown: float


class PerformanceScorer:
    """최근 거래 성과 기반 스코어 산출."""

    def summarize(self, trades: Iterable[Dict[str, float]]) -> ScoreSummary:
        pnl_series: List[float] = []
        roi = 0.0
        peak = 0.0
        trough = 0.0
        max_dd = 0.0

        for trade in trades:
            pnl = float(trade.get("pnl", 0.0))
            pnl_series.append(pnl)
            roi += pnl
            peak = max(peak, roi)
            trough = min(trough, roi)
            max_dd = min(max_dd, roi - peak)

        sharpe = 0.0
        if pnl_series:
            mean = sum(pnl_series) / len(pnl_series)
            variance = sum((x - mean) ** 2 for x in pnl_series) / len(pnl_series)
            std = variance**0.5
            if std > 0:
                sharpe = mean / std

        return ScoreSummary(
            roi=roi,
            sharpe=sharpe,
            max_drawdown=max_dd,
        )

