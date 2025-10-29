"""Utilities for fetching and formatting market data."""

from __future__ import annotations

import ccxt
import pandas as pd


_TIMEFRAME_TO_MS = {
    "1m": 60_000,
    "3m": 180_000,
    "5m": 300_000,
    "15m": 900_000,
    "1h": 3_600_000,
    "4h": 14_400_000,
    "1d": 86_400_000,
}


def ohlcv_csv_between(
    spot_symbol: str,
    timeframe: str,
    since_ms: int,
    until_ms: int,
) -> str:
    """Return OHLCV data as CSV limited to the requested interval."""
    try:
        step = _TIMEFRAME_TO_MS.get(timeframe, 60_000)
        est_bars = max(1, int((until_ms - since_ms) / step) + 5)
        limit = min(1000, est_bars)
        exchange = ccxt.bybit()
        rows = exchange.fetch_ohlcv(spot_symbol, timeframe, since=since_ms, limit=limit)
        rows = [
            r for r in rows if len(r) >= 6 and r[0] <= until_ms  # type: ignore[index]
        ]
        if not rows:
            return ""

        df = pd.DataFrame(
            rows,
            columns=["timestamp", "open", "high", "low", "close", "volume"],
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        return df.to_csv()
    except Exception:
        return ""
