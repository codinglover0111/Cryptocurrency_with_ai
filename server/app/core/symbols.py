"""Symbol helpers shared across runtime components."""

# pylint: disable=broad-except
# ruff: noqa: E722, BLE001
from __future__ import annotations

import json
import os
from typing import List, Optional, Sequence, Tuple

DEFAULT_SYMBOLS: Tuple[str, ...] = (
    "XRPUSDT",
    "ETHUSDT",
    "BTCUSDT",
    "SOLUSDT",
    # "WLDUSDT",
    # "DOGEUSDT",
)

# Bybit에서 거래 가능한 주요 USDT 선물 심볼 목록
AVAILABLE_SYMBOLS: Tuple[str, ...] = (
    "BTCUSDT",
    "ETHUSDT",
    "XRPUSDT",
    "SOLUSDT",
    "DOGEUSDT",
    "ADAUSDT",
    "AVAXUSDT",
    "DOTUSDT",
    "LINKUSDT",
    "MATICUSDT",
    "UNIUSDT",
    "ATOMUSDT",
    "LTCUSDT",
    "ETCUSDT",
    "FILUSDT",
    "APTUSDT",
    "ARBUSDT",
    "OPUSDT",
    "NEARUSDT",
    "FTMUSDT",
    "SANDUSDT",
    "MANAUSDT",
    "AXSUSDT",
    "AAVEUSDT",
    "MKRUSDT",
    "COMPUSDT",
    "CRVUSDT",
    "LDOUSDT",
    "RNDRUSDT",
    "INJUSDT",
    "SUIUSDT",
    "SEIUSDT",
    "TIAUSDT",
    "WLDUSDT",
    "PEPEUSDT",
    "SHIBUSDT",
    "FLOKIUSDT",
    "BONKUSDT",
    "ORDIUSDT",
    "1000SATSUSDT",
    "WIFUSDT",
    "JUPUSDT",
    "STXUSDT",
    "IMXUSDT",
    "GMXUSDT",
    "DYDXUSDT",
    "AGIXUSDT",
    "FETUSDT",
    "OCEANUSDT",
    "BNBUSDT",
    "TRXUSDT",
    "BCHUSDT",
    "EOSUSDT",
    "XLMUSDT",
    "ALGOUSDT",
    "VETUSDT",
    "ICPUSDT",
    "HBARUSDT",
    "QNTUSDT",
    "EGLDUSDT",
    "XMRUSDT",
    "RUNEUSDT",
    "GRTUSDT",
    "SNXUSDT",
    "1INCHUSDT",
    "ENSUSDT",
    "APEUSDT",
    "CHZUSDT",
    "GALAUSDT",
    "ENJUSDT",
    "FLOWUSDT",
    "MINAUSDT",
    "KAVAUSDT",
    "ROSEUSDT",
    "ZILUSDT",
    "KNCUSDT",
    "SKLUSDT",
    "ANKRUSDT",
    "BLURUSDT",
    "CFXUSDT",
    "CKBUSDT",
    "JASMYUSDT",
    "KASUSDT",
    "LUNCUSDT",
    "MASKUSDT",
    "NFPUSDT",
    "ONDOUSDT",
    "PENDLEUSDT",
    "PYTHUSDT",
    "STRKUSDT",
    "TONUSDT",
    "TRBUSDT",
    "TURBOUSDT",
    "WOOUSDT",
    "ZROUSDT",
    "EIGENUSDT",
    "TAOUSDT",
)


def round_price(price: float) -> float:
    """Round prices to four decimal places, matching XRP/USDT precision."""
    return round(float(price), 4)


def _get_symbols_from_db() -> Optional[List[str]]:
    """DB에서 거래 심볼 목록을 읽어옵니다."""
    try:
        from utils.storage import TradeStore, StorageConfig

        store = TradeStore(
            StorageConfig(
                mysql_url=os.getenv("MYSQL_URL"),
                sqlite_path=os.getenv("SQLITE_PATH"),
            )
        )
        config_data = store.get_runtime_config("trading_symbols")
        if config_data:
            data = json.loads(config_data)
            symbols = data.get("symbols", [])
            if symbols and isinstance(symbols, list):
                return [s.strip().upper() for s in symbols if s.strip()]
        return None
    except Exception as e:
        print(f"Warning: Failed to read trading symbols from DB: {e}")
        return None


def get_trading_symbols_from_db() -> Optional[List[str]]:
    """DB에서 거래 심볼 목록을 가져옵니다 (외부 호출용)."""
    return _get_symbols_from_db()


def save_trading_symbols_to_db(symbols: List[str]) -> bool:
    """거래 심볼 목록을 DB에 저장합니다."""
    try:
        from utils.storage import TradeStore, StorageConfig

        store = TradeStore(
            StorageConfig(
                mysql_url=os.getenv("MYSQL_URL"),
                sqlite_path=os.getenv("SQLITE_PATH"),
            )
        )
        # 심볼 정규화
        normalized = [s.strip().upper() for s in symbols if s.strip()]
        config_data = json.dumps({"symbols": normalized})
        return store.save_runtime_config("trading_symbols", config_data)
    except Exception as e:
        print(f"Error saving trading symbols to DB: {e}")
        return False


def parse_trading_symbols(raw: str | None = None) -> List[str]:
    """Return the configured list of symbols.

    우선순위:
    1. 제공된 raw 문자열
    2. DB에 저장된 심볼 목록
    3. 환경변수 TRADING_SYMBOLS
    4. DEFAULT_SYMBOLS
    """
    # raw 문자열이 제공된 경우
    if raw:
        symbols = [sym.strip().upper() for sym in raw.split(",") if sym.strip()]
        return symbols or list(DEFAULT_SYMBOLS)

    # DB에서 먼저 확인
    db_symbols = _get_symbols_from_db()
    if db_symbols:
        return db_symbols

    # 환경변수에서 확인
    env_symbols = os.getenv("TRADING_SYMBOLS")
    if env_symbols:
        symbols = [sym.strip().upper() for sym in env_symbols.split(",") if sym.strip()]
        return symbols or list(DEFAULT_SYMBOLS)

    # 기본값 반환
    return list(DEFAULT_SYMBOLS)


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
