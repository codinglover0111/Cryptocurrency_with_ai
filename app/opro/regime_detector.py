"""시장 레짐 감지."""

from __future__ import annotations

from typing import Literal

import pandas as pd
from ta.trend import ADXIndicator


class RegimeDetector:
    """ADX 기반 시장 레짐 감지기."""

    def __init__(self, adx_window: int = 14, adx_threshold: float = 25.0) -> None:
        self.adx_window = adx_window
        self.adx_threshold = adx_threshold

    def detect_regime(
        self, df: pd.DataFrame | None
    ) -> Literal["bullish", "bearish", "sideways", "unknown"]:
        if df is None or df.empty or len(df) < self.adx_window + 5:
            return "unknown"

        try:
            indicator = ADXIndicator(
                high=df["high"], low=df["low"], close=df["close"], window=self.adx_window
            )
            adx_series = indicator.adx()
            plus_di = indicator.adx_pos()
            minus_di = indicator.adx_neg()
            latest_adx = float(adx_series.iloc[-1])
            plus_val = float(plus_di.iloc[-1])
            minus_val = float(minus_di.iloc[-1])
        except Exception:
            return "unknown"

        if latest_adx < self.adx_threshold:
            return "sideways"
        if plus_val >= minus_val:
            return "bullish"
        return "bearish"

