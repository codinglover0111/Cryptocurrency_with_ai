"""Symbol helpers shared across runtime components."""

from __future__ import annotations

import os
from typing import List, Sequence, Tuple

DEFAULT_SYMBOLS: Tuple[str, ...] = (
    "XRPUSDT",
    "WLDUSDT",
    "ETHUSDT",
    "BTCUSDT",
    "SOLUSDT",
    "DOGEUSDT",
)


def round_price(price: float) -> float:
    """Round prices to four decimal places, matching XRP/USDT precision."""
    return round(float(price), 4)


def parse_trading_symbols(raw: str | None = None) -> List[str]:
    """Return the configured list of symbols from environment or provided raw string."""
    raw_symbols = raw or os.getenv("TRADING_SYMBOLS", ",".join(DEFAULT_SYMBOLS))
    symbols = [sym.strip().upper() for sym in raw_symbols.split(",") if sym.strip()]
    return symbols or list(DEFAULT_SYMBOLS)


def to_ccxt_symbols(symbol_usdt: str) -> Tuple[str, str]:
    """Map BYBIT symbol (e.g. BTCUSDT) to CCXT spot/contract symbols."""
    symbol = symbol_usdt.upper().replace(":USDT", "").replace("/", "")
    base = symbol[:-4] if symbol.endswith("USDT") else symbol
    spot_symbol = f"{base}/USDT"
    contract_symbol = f"{base}/USDT:USDT"
    return spot_symbol, contract_symbol


def contract_to_spot_symbol(contract_symbol: str) -> str:
    """Return the spot symbol for a contract symbol."""
    try:
        return str(contract_symbol).replace(":USDT", "")
    except Exception:
        return contract_symbol


def per_symbol_allocation(total_symbols: Sequence[str]) -> float:
    """Return the percentage allocation per symbol given the configured symbols."""
    count = max(1, len(total_symbols))
    return 100.0 / float(count)
