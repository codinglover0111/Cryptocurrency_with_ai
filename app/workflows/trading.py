"""Automated trading workflow orchestration."""

# pylint: disable=broad-except
# ruff: noqa: E722, BLE001
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from utils import BybitUtils, Open_Position, bybit_utils, make_to_object
from utils.ai_provider import AIProvider
from utils.risk import calculate_position_size, enforce_max_loss_sl
from utils.storage import StorageConfig, TradeStore

from app.core.symbols import (
    parse_trading_symbols,
    per_symbol_allocation,
    to_ccxt_symbols,
)
from app.services.journal import JournalService


LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class AutomationDependencies:
    """Dependencies shared throughout a trading cycle."""

    symbol_usdt: str
    symbols: Sequence[str]
    spot_symbol: str
    contract_symbol: str
    per_symbol_alloc_pct: float
    bybit: BybitUtils
    store: TradeStore
    ai_provider: AIProvider
    parser: Any
    journal_service: JournalService = field(init=False)

    def __post_init__(self) -> None:
        self.journal_service = JournalService(self.store, self.ai_provider)


@dataclass(slots=True)
class PromptContext:
    """Snapshot of data forwarded to the LLM."""

    current_price: float
    csv_4h: str
    csv_1h: str
    csv_15m: str
    current_position: List[Dict[str, Any]]
    pos_side: Optional[str]
    current_positions_lines: List[str]
    journal_today_text: str
    decision_docs_text: str
    review_docs_text: str
    since_open_text: str


def _compute_tp_sl_percentages(
    *,
    entry_price: float,
    tp: Optional[float],
    sl: Optional[float],
    ai_status: str,
    leverage: float,
) -> Dict[str, Optional[float]]:
    """Calculate tp/sl percentages (raw and leverage-adjusted)."""

    result: Dict[str, Optional[float]] = {
        "tp_pct": None,
        "sl_pct": None,
        "tp_pct_leverage": None,
        "sl_pct_leverage": None,
    }

    try:
        e = float(entry_price)
        if e <= 0:
            return result

        lev = abs(float(leverage or 0.0))

        def _safe(value: Optional[float]) -> Optional[float]:
            try:
                if value is None:
                    return None
                return float(value)
            except Exception:
                return None

        tp_v = _safe(tp)
        sl_v = _safe(sl)
        status = (ai_status or "").lower()

        if tp_v is not None:
            if status == "long":
                result["tp_pct"] = (tp_v - e) / e * 100.0
            elif status == "short":
                result["tp_pct"] = (e - tp_v) / e * 100.0

        if sl_v is not None:
            if status == "long":
                result["sl_pct"] = (e - sl_v) / e * 100.0
            elif status == "short":
                result["sl_pct"] = (sl_v - e) / e * 100.0

        if lev > 0:
            tp_pct = result["tp_pct"]
            sl_pct = result["sl_pct"]
            if tp_pct is not None:
                result["tp_pct_leverage"] = tp_pct * lev
            if sl_pct is not None:
                result["sl_pct_leverage"] = sl_pct * lev
    except Exception:
        return result

    return result


def _init_dependencies(
    symbol_usdt: str, symbols: Sequence[str] | None
) -> AutomationDependencies:
    is_testnet = bool(int(os.getenv("TESTNET", "1")))
    all_symbols = list(symbols) if symbols else parse_trading_symbols()
    spot_symbol, contract_symbol = to_ccxt_symbols(symbol_usdt)
    per_symbol_pct = per_symbol_allocation(all_symbols)

    bybit = BybitUtils(is_testnet)
    store = TradeStore(
        StorageConfig(
            mysql_url=os.getenv("MYSQL_URL"),
            sqlite_path=os.getenv("SQLITE_PATH"),
        )
    )
    ai_provider = AIProvider()
    parser = make_to_object()

    deps = AutomationDependencies(
        symbol_usdt=symbol_usdt,
        symbols=all_symbols,
        spot_symbol=spot_symbol,
        contract_symbol=contract_symbol,
        per_symbol_alloc_pct=per_symbol_pct,
        bybit=bybit,
        store=store,
        ai_provider=ai_provider,
        parser=parser,
    )

    return deps


def _filter_recent_ohlcv(df: pd.DataFrame, *, days: float) -> pd.DataFrame:
    """Limit OHLCV data to the requested recent window (in days)."""

    if df is None or getattr(df, "empty", True):
        return df

    try:
        cutoff = pd.Timestamp(datetime.utcnow() - timedelta(days=float(days)))
    except Exception:
        return df

    try:
        filtered = df[df.index >= cutoff]
    except Exception:
        return df

    if getattr(filtered, "empty", True):
        return df

    return filtered


def _summarize_positions(
    positions: Iterable[Dict[str, Any]],
    contract_symbol: str,
    current_price: float,
) -> Tuple[List[str], Optional[str]]:
    lines: List[str] = []
    primary_side: Optional[str] = None
    last_fallback = float(current_price or 0)

    def _safe_float(value: Any) -> Optional[float]:
        try:
            if value is None:
                return None
            return float(value)
        except Exception:
            return None

    def _fmt_price(value: Optional[float]) -> str:
        return f"{value:.6f}" if value is not None else "n/a"

    def _fmt_pct(value: Optional[float]) -> str:
        return f"{value:.4f}%" if value is not None else "n/a"

    def _fmt_side(value: Optional[str]) -> str:
        side_str = (value or "").strip().lower()
        if side_str in {"long", "buy"}:
            return "롱"
        if side_str in {"short", "sell"}:
            return "숏"
        return value if value else "미정"

    for raw in positions or []:
        try:
            info = raw.get("info") or {}
            sym = raw.get("symbol") or (info if isinstance(info, dict) else {}).get(
                "symbol"
            )
            if sym != contract_symbol:
                continue

            side = raw.get("side") or (info if isinstance(info, dict) else {}).get(
                "side"
            )
            if primary_side is None and side:
                primary_side = side

            entry = raw.get("entryPrice") or (
                info if isinstance(info, dict) else {}
            ).get("avgPrice")
            mark = raw.get("markPrice") or (info if isinstance(info, dict) else {}).get(
                "markPrice"
            )
            try:
                last = float(mark) if mark is not None else float(last_fallback)
            except Exception:
                last = last_fallback

            entry_f = _safe_float(entry)
            tp_raw = raw.get("takeProfit") or (
                info if isinstance(info, dict) else {}
            ).get("takeProfit")
            sl_raw = raw.get("stopLoss") or (
                info if isinstance(info, dict) else {}
            ).get("stopLoss")
            lev_raw = raw.get("leverage") or (
                info if isinstance(info, dict) else {}
            ).get("leverage")
            pct_raw = raw.get("percentage") or (
                info if isinstance(info, dict) else {}
            ).get("unrealisedPnlPcnt")

            lev_f = _safe_float(lev_raw)
            pct_raw_f = _safe_float(pct_raw)
            tp_f = _safe_float(tp_raw)
            sl_f = _safe_float(sl_raw)

            directional_pct: Optional[float] = None
            if entry_f is not None and entry_f not in (0.0, -0.0):
                try:
                    base_pct = (last - entry_f) / entry_f * 100.0
                    side_str = (side or "").lower()
                    if side_str in {"short", "sell"}:
                        base_pct *= -1.0
                    directional_pct = base_pct
                except Exception:
                    directional_pct = None

            pct_leverage: Optional[float] = None
            if directional_pct is not None:
                if lev_f is not None and lev_f not in (0.0, -0.0):
                    pct_leverage = directional_pct * lev_f
                else:
                    pct_leverage = directional_pct
            elif pct_raw_f is not None:
                pct_leverage = pct_raw_f

            tp_sl_pct = _compute_tp_sl_percentages(
                entry_price=entry_f if entry_f is not None else 0.0,
                tp=tp_f,
                sl=sl_f,
                ai_status=side or "",
                leverage=lev_f if lev_f is not None else 0.0,
            )

            lines.append(
                "  "
                + ", ".join(
                    [
                        f"포지션 종류={_fmt_side(side)}",
                        f"익절가={_fmt_price(tp_f)}",
                        f"손절가={_fmt_price(sl_f)}",
                        f"레버리지 기준 익절 퍼센트={_fmt_pct(tp_sl_pct.get('tp_pct_leverage'))}",
                        f"레버리지 기준 손절 퍼센트={_fmt_pct(tp_sl_pct.get('sl_pct_leverage'))}",
                        f"레버리지 기준 현재 수익률={_fmt_pct(pct_leverage)}",
                    ]
                )
            )
        except Exception:
            continue

    return lines, primary_side


def _filter_active_positions(
    positions: Iterable[Dict[str, Any]], contract_symbol: str
) -> List[Dict[str, Any]]:
    """Return only active positions that match the provided contract symbol."""

    active: List[Dict[str, Any]] = []
    for raw in positions or []:
        try:
            info = raw.get("info") or {}
            sym = raw.get("symbol") or (info if isinstance(info, dict) else {}).get(
                "symbol"
            )
            if sym != contract_symbol:
                continue

            size_candidates = [
                raw.get("contracts"),
                raw.get("amount"),
                raw.get("size"),
            ]
            if isinstance(info, dict):
                size_candidates.extend(
                    [
                        info.get("contracts"),
                        info.get("amount"),
                        info.get("size"),
                        info.get("positionAmt"),
                    ]
                )

            max_size = 0.0
            for cand in size_candidates:
                if cand is None:
                    continue
                try:
                    max_size = max(max_size, abs(float(cand)))
                except Exception:
                    continue

            if max_size > 1e-8:
                active.append(raw)
        except Exception:
            continue

    return active


def _gather_prompt_context(deps: AutomationDependencies) -> PromptContext:
    price_helper = bybit_utils(deps.spot_symbol, "4h", 200)

    df_4h_all = price_helper.get_ohlcv()
    df_4h = _filter_recent_ohlcv(df_4h_all, days=7)
    if getattr(df_4h, "empty", True):
        df_4h = df_4h_all
    csv_4h = df_4h.to_csv()

    price_helper.set_timeframe("1h")
    price_helper.limit = max(getattr(price_helper, "limit", 200), 200)
    df_1h_all = price_helper.get_ohlcv()
    df_1h = _filter_recent_ohlcv(df_1h_all, days=3)
    if getattr(df_1h, "empty", True):
        df_1h = df_1h_all
    csv_1h = df_1h.to_csv()

    price_helper.set_timeframe("15m")
    price_helper.limit = max(getattr(price_helper, "limit", 200), 200)
    df_15m_all = price_helper.get_ohlcv()
    df_15m = _filter_recent_ohlcv(df_15m_all, days=1)
    if getattr(df_15m, "empty", True):
        df_15m = df_15m_all
    csv_15m = df_15m.to_csv()
    current_price = df_15m["close"].iloc[-1]

    all_positions = deps.bybit.get_positions() or []

    position_lines: List[str] = []
    pos_side: Optional[str] = None
    current_position: List[Dict[str, Any]] = []
    summaries: Dict[str, List[str]] = {}
    symbol_contract_map: Dict[str, str] = {}

    for sym in deps.symbols:
        try:
            _, contract_sym = to_ccxt_symbols(sym)
        except Exception:
            # Skip malformed symbols while still keeping prompt generation alive
            continue

        symbol_contract_map[sym] = contract_sym

        symbol_positions = _filter_active_positions(all_positions, contract_sym)

        fallback_price = current_price if contract_sym == deps.contract_symbol else 0.0
        if symbol_positions and contract_sym != deps.contract_symbol:
            try:
                fallback_price = (
                    deps.bybit.get_last_price(contract_sym) or fallback_price
                )
            except Exception:
                pass

        lines, symbol_primary = _summarize_positions(
            symbol_positions, contract_sym, fallback_price
        )

        summaries[sym] = lines
        if contract_sym == deps.contract_symbol:
            current_position = symbol_positions
            pos_side = symbol_primary

    seen = set()
    for sym in deps.symbols:
        if sym in seen:
            continue
        seen.add(sym)
        lines = summaries.get(sym, [])
        if lines:
            position_lines.append(f"{sym}:")
            position_lines.extend(lines)
        else:
            position_lines.append(f"{sym}: (none)")

    if not position_lines:
        position_lines.append("(none)")

    contract_symbols = set(symbol_contract_map.values())
    for raw in all_positions:
        try:
            info = raw.get("info") or {}
            sym = raw.get("symbol") or (info if isinstance(info, dict) else {}).get(
                "symbol"
            )
        except Exception:
            continue
        if not sym or sym in contract_symbols:
            continue
        extra_positions = _filter_active_positions(all_positions, sym)
        if not extra_positions:
            continue
        fallback_price = current_price
        try:
            fallback_price = deps.bybit.get_last_price(sym) or fallback_price
        except Exception:
            pass
        lines, _ = _summarize_positions(extra_positions, sym, fallback_price)
        if not lines:
            continue
        position_lines.append(f"{sym}:")
        position_lines.extend(lines)

    try:
        journals_today_df = deps.store.fetch_journals(
            symbol=deps.contract_symbol, today_only=True, limit=50, ascending=True
        )
        journal_today_text = ""
        if journals_today_df is not None and not journals_today_df.empty:
            lines = []
            for _, row in journals_today_df.iterrows():
                ts = row.get("ts")
                ts_str = ts.strftime("%H:%M:%S") if hasattr(ts, "strftime") else str(ts)
                lines.append(
                    f"[{ts_str}] ({row.get('entry_type')}) {row.get('reason') or ''} | {row.get('content') or ''}"
                )
            journal_today_text = "\n".join(lines)
    except Exception:
        journal_today_text = ""

    decision_docs_text = ""
    try:
        decisions_df = deps.store.fetch_journals(
            symbol=deps.contract_symbol,
            types=["decision"],
            since_ts=datetime.utcnow() - timedelta(hours=24),
            limit=200,
            ascending=True,
        )
        if decisions_df is not None and not decisions_df.empty:
            lines = []
            for _, row in decisions_df.sort_values("ts").iterrows():
                ts = row.get("ts")
                ts_str = (
                    ts.strftime("%m-%d %H:%M") if hasattr(ts, "strftime") else str(ts)
                )
                entry_type = (row.get("entry_type") or "").strip()
                reason = (row.get("reason") or "").strip()
                content = (row.get("content") or "").strip()
                if len(content) > 500:
                    content = content[:500]
                parts = [f"[{ts_str}]"]
                if entry_type:
                    parts.append(f"({entry_type})")
                if reason:
                    parts.append(reason)
                if content:
                    if reason or entry_type:
                        parts.append(f"| {content}")
                    else:
                        parts.append(content)
                line = " ".join(parts).strip()
                lines.append(line)
            decision_docs_text = "\n".join(lines)
    except Exception:
        decision_docs_text = ""

    review_docs_text = ""
    try:
        reviews_df = deps.store.fetch_journals(
            symbol=deps.contract_symbol,
            types=["review"],
            limit=5,
            ascending=False,
        )
        if reviews_df is not None and not reviews_df.empty:
            lines = []
            for _, row in reviews_df.iterrows():
                ts = row.get("ts")
                ts_str = (
                    ts.strftime("%m-%d %H:%M") if hasattr(ts, "strftime") else str(ts)
                )
                reason = (row.get("reason") or "").strip()
                content = (row.get("content") or "").strip()
                if len(content) > 500:
                    content = content[:500]
                parts = [f"[{ts_str}]"]
                if reason:
                    parts.append(reason)
                if content:
                    if reason:
                        parts.append(f"| {content}")
                    else:
                        parts.append(content)
                line = " ".join(parts).strip()
                lines.append(line)
            review_docs_text = "\n".join(lines)
    except Exception:
        review_docs_text = ""

    since_open_text = ""
    try:
        if pos_side is not None:
            trades_df = deps.store.load_trades()
            if trades_df is not None and not trades_df.empty:
                df_sym = trades_df[trades_df["symbol"] == deps.contract_symbol]
                df_opened = df_sym[df_sym["status"] == "opened"]
                if not df_opened.empty:
                    last_open_ts = df_opened["ts"].max()
                    journals_since_df = deps.store.fetch_journals(
                        symbol=deps.contract_symbol,
                        today_only=True,
                        since_ts=last_open_ts,
                        limit=50,
                        ascending=True,
                    )
                    if journals_since_df is not None and not journals_since_df.empty:
                        lines = []
                        for _, row in journals_since_df.iterrows():
                            ts = row.get("ts")
                            ts_str = (
                                ts.strftime("%H:%M:%S")
                                if hasattr(ts, "strftime")
                                else str(ts)
                            )
                            lines.append(
                                f"[{ts_str}] ({row.get('entry_type')}) {row.get('reason') or ''} | {row.get('content') or ''}"
                            )
                        since_open_text = "\n".join(lines)
    except Exception:
        since_open_text = ""

    return PromptContext(
        current_price=current_price,
        csv_4h=csv_4h,
        csv_1h=csv_1h,
        csv_15m=csv_15m,
        current_position=current_position,
        pos_side=pos_side,
        current_positions_lines=position_lines,
        journal_today_text=journal_today_text,
        decision_docs_text=decision_docs_text,
        review_docs_text=review_docs_text,
        since_open_text=since_open_text,
    )


def _build_prompt(deps: AutomationDependencies, ctx: PromptContext) -> str:
    now_utc = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    prompt = (
        f"당신은 세계 최고의 암호화폐 트레이더입니다. 한국어로 답변하세요.\n"
        f"당신은 최소 5배에서 최대 75배까지 레버리지를 사용할 수 있습니다.\n"
        f"현재 UTC 시간: {now_utc}\n"
        f"심볼: {deps.contract_symbol} (spot={deps.spot_symbol})\n"
        f"현재가: {ctx.current_price}\n"
        f"심볼당 기본 배분 비율: {deps.per_symbol_alloc_pct:.2f}%\n"
        "[OLCHV_4H_LAST_7D]\n"
        f"{ctx.csv_4h}\n"
        "[/OLCHV_4H_LAST_7D]\n"
        "[OLCHV_1H_LAST_3D]\n"
        f"{ctx.csv_1h}\n"
        "[/OLCHV_1H_LAST_3D]\n"
        "[OLCHV_15M_LAST_1D]\n"
        f"{ctx.csv_15m}\n"
        "[/OLCHV_15M_LAST_1D]\n"
        "[CURRENT_POSITIONS]\n"
        + (
            "\n".join(ctx.current_positions_lines)
            if ctx.current_positions_lines
            else "(none)"
        )
        + "\n[/CURRENT_POSITIONS]\n"
    )

    if ctx.decision_docs_text:
        prompt += (
            "[DECISION_DOCS_24H]\n"
            + ctx.decision_docs_text
            + "\n[/DECISION_DOCS_24H]\n"
        )
    if ctx.review_docs_text:
        prompt += (
            "[REVIEW_DOCS_LATEST5]\n"
            + ctx.review_docs_text
            + "\n[/REVIEW_DOCS_LATEST5]\n"
        )
    if ctx.journal_today_text:
        prompt += (
            "[JOURNALS_TODAY]\n" + ctx.journal_today_text + "\n[/JOURNALS_TODAY]\n"
        )
    if ctx.pos_side is not None and ctx.since_open_text:
        prompt += "[SINCE_LAST_OPEN]\n" + ctx.since_open_text + "\n[/SINCE_LAST_OPEN]\n"

    prompt += (
        "Choose one of: watch/hold, short, long, or stop.\n"
        "Return your decision as JSON with the following fields: order type (market/limit), "
        "price, stop loss (sl), take profit (tp), buy_now (boolean), leverage (number).\n"
        "If you want to immediately take profit or cut loss on an existing position, set close_now=true "
        "and optionally close_percent (1~100).\n"
        "When close_now is true, we will reduceOnly market-close the current position(s) without opening a new one in this cycle.\n"
        f"Per-symbol max allocation is {deps.per_symbol_alloc_pct:.2f}% of account equity (for your reference).\n"
        "Include your reasoning for the decision in the 'explain' field. Output JSON, Korean only."
    )
    return prompt


def _request_trade_decision(
    deps: AutomationDependencies,
    prompt: str,
    _ctx: PromptContext,
) -> Dict[str, Any]:
    try:
        parsed = deps.ai_provider.decide_json(prompt)
        LOGGER.info(
            json.dumps(
                {
                    "event": "llm_response_parsed",
                    "provider": deps.ai_provider.provider,
                    "parsed": parsed,
                },
                ensure_ascii=False,
            )
        )
        return parsed
    except Exception:
        response = deps.ai_provider.decide(prompt)
        try:
            LOGGER.info(
                json.dumps(
                    {
                        "event": "llm_response_raw",
                        "provider": deps.ai_provider.provider,
                        "response": response,
                    },
                    ensure_ascii=False,
                )
            )
        except Exception as exc:
            LOGGER.error("LLM raw logging failed: %s", exc)

        parser = deps.parser
        value = parser.make_it_object(response)
        try:
            LOGGER.info(
                json.dumps(
                    {
                        "event": "llm_response_parsed",
                        "provider": deps.ai_provider.provider,
                        "parsed": value,
                    },
                    ensure_ascii=False,
                )
            )
        except Exception as exc:
            LOGGER.error("LLM parsed logging failed: %s", exc)
        return value


def _normalize_bool(val: Any) -> bool:
    try:
        if isinstance(val, bool):
            return val
        if isinstance(val, (int, float)):
            return float(val) == 1.0
        if isinstance(val, str):
            return val.strip().lower() in {
                "1",
                "true",
                "yes",
                "y",
                "market",
                "now",
                "immediate",
            }
    except Exception:
        pass
    return False


def _extract_order_type(data: Dict[str, Any]) -> Optional[str]:
    try:
        value = (
            data.get("type")
            or data.get("order_type")
            or data.get("order type")
            or data.get("orderType")
            or ""
        )
        value = str(value).strip().lower()
    except Exception:
        value = ""
    if value in {"market", "mkt"}:
        return "market"
    if value in {"limit", "lmt"}:
        return "limit"
    return None


def _handle_close_now(
    deps: AutomationDependencies,
    ctx: PromptContext,
    decision: Dict[str, Any],
) -> bool:
    if not bool(decision.get("close_now")):
        return False

    try:
        percent = float(decision.get("close_percent") or 100.0)
    except Exception:
        percent = 100.0
    percent = 100.0 if percent <= 0 else (percent if percent <= 100 else 100.0)

    positions = deps.bybit.get_positions_by_symbol(deps.contract_symbol) or []
    if not positions:
        LOGGER.info("close_now ignored: no active position for symbol")
        return True

    try:
        if percent >= 99.999:
            res = deps.bybit.close_symbol_positions(deps.contract_symbol)
        else:
            res = deps.bybit.reduce_symbol_positions_percent(
                deps.contract_symbol, percent
            )
        try:
            last = deps.bybit.get_last_price(deps.contract_symbol) or ctx.current_price
            for position in positions:
                entry = float(
                    position.get("entryPrice")
                    or position.get("info", {}).get("avgPrice")
                    or 0
                )
                amount = float(
                    position.get("contracts")
                    or position.get("amount")
                    or position.get("size")
                    or 0
                )
                side = position.get("side") or position.get("info", {}).get("side")
                if amount <= 0 or entry <= 0 or not side:
                    continue
                closed_qty = amount if percent >= 99.999 else amount * (percent / 100.0)
                pnl = (
                    (last - entry) * closed_qty
                    if side == "long"
                    else (entry - last) * closed_qty
                )
                deps.store.record_trade(
                    {
                        "ts": datetime.utcnow(),
                        "symbol": deps.contract_symbol,
                        "side": side,
                        "type": "market",
                        "price": last,
                        "quantity": float(closed_qty),
                        "tp": None,
                        "sl": None,
                        "leverage": None,
                        "status": "closed",
                        "order_id": None,
                        "pnl": float(pnl),
                    }
                )
        except Exception:
            pass
        try:
            deps.store.record_journal(
                {
                    "symbol": deps.contract_symbol,
                    "entry_type": "action",
                    "content": f"close_now percent={float(percent)}",
                    "reason": decision.get("explain"),
                    "meta": {"result": str(res)[:500]},
                }
            )
        except Exception:
            pass
    except Exception as exc:
        LOGGER.error("close_now failed: %s", exc)
    return True


def _run_confirm_step(
    *,
    deps: AutomationDependencies,
    decision: Dict[str, Any],
    ai_status: str,
    side: str,
    order_type: str,
    entry_price: float,
    use_tp: Optional[float],
    use_sl: Optional[float],
    leverage: float,
) -> Tuple[
    str, float, Optional[float], Optional[float], float, Optional[Dict[str, Any]], bool
]:
    confirm_meta: Optional[Dict[str, Any]] = None

    try:
        tp_valid = isinstance(use_tp, (int, float)) and float(use_tp) > 0
        sl_valid = isinstance(use_sl, (int, float)) and float(use_sl) > 0
        if not (tp_valid and sl_valid and float(entry_price) > 0):
            return order_type, entry_price, use_tp, use_sl, leverage, None, False

        e = float(entry_price)
        tp_v = float(use_tp)
        sl_v = float(use_sl)

        pct_info = _compute_tp_sl_percentages(
            entry_price=e,
            tp=tp_v,
            sl=sl_v,
            ai_status=ai_status,
            leverage=leverage,
        )
        tp_pct = float(pct_info.get("tp_pct") or 0.0)
        sl_pct = float(pct_info.get("sl_pct") or 0.0)
        tp_pct_leverage = float(pct_info.get("tp_pct_leverage") or 0.0)
        sl_pct_leverage = float(pct_info.get("sl_pct_leverage") or 0.0)

        confirm_prompt = (
            "당신이 제안한 주문 파라미터를 최종 확인하세요. JSON만 응답. 한국어로.\n"
            f"심볼: {deps.contract_symbol}\n"
            f"포지션: {ai_status} (내부 side={side})\n"
            f"진입가(entry): {float(e)}\n"
            f"TP: {float(tp_v)} (예상 수익률: {tp_pct:.4f}% | 레버리지 기준: {tp_pct_leverage:.4f}%)\n"
            f"SL: {float(sl_v)} (예상 손실률: {sl_pct:.4f}% | 레버리지 기준: {sl_pct_leverage:.4f}%)\n"
            f"레버리지: {float(leverage)}x\n"
            "레버리지 기준 손실률은 청산 방지를 위해 85%를 넘으면 안 됩니다. 필요시 조정하세요.\n"
            "필수: confirm(boolean). 선택: tp, sl, price, buy_now, leverage, explain.\n"
            "확신하면 confirm=true. 수정이 필요하면 값을 조정해 응답하세요."
        )
        confirm = deps.ai_provider.confirm_trade_json(confirm_prompt)
        confirm_meta = confirm
        LOGGER.info(
            json.dumps(
                {
                    "event": "llm_confirm_response_parsed",
                    "provider": deps.ai_provider.provider,
                    "parsed": confirm,
                },
                ensure_ascii=False,
            )
        )
        if not bool(confirm.get("confirm")):
            _record_skip(
                deps,
                reason="skip_after_confirm",
                decision=decision,
                meta={"first": decision, "confirm": confirm},
            )
            return order_type, entry_price, use_tp, use_sl, leverage, confirm_meta, True

        if confirm.get("tp") is not None:
            try:
                use_tp = float(confirm.get("tp"))
            except Exception:
                pass
        if confirm.get("sl") is not None:
            try:
                use_sl = float(confirm.get("sl"))
            except Exception:
                pass
        if confirm.get("price") is not None:
            try:
                entry_price = float(confirm.get("price"))
            except Exception:
                pass

        override_type = _extract_order_type(confirm)
        if override_type:
            order_type = override_type
        elif confirm.get("buy_now") is not None:
            order_type = (
                "market" if _normalize_bool(confirm.get("buy_now")) else "limit"
            )
        if confirm.get("leverage") is not None:
            try:
                leverage = float(confirm.get("leverage"))
            except Exception:
                pass
    except Exception as exc:
        LOGGER.error("AI confirm failed: %s", exc)

    return (
        order_type,
        float(entry_price),
        use_tp,
        use_sl,
        leverage,
        confirm_meta,
        False,
    )


def _execute_trade(
    deps: AutomationDependencies,
    ctx: PromptContext,
    decision: Dict[str, Any],
) -> None:
    ai_status = str(decision.get("Status") or decision.get("status") or "").lower()
    if ai_status not in {"long", "short"}:
        _handle_non_trade_actions(deps, decision, ctx.current_price, ai_status)
        return

    side = "buy" if ai_status == "long" else "sell"
    order_type = _extract_order_type(decision) or (
        "market" if _normalize_bool(decision.get("buy_now")) else "limit"
    )

    balance_info = deps.bybit.get_balance("USDT") or {}
    balance_total = balance_info.get("total") or 0
    balance_free = balance_info.get("free") or 0

    risk_percent = 20.0
    max_alloc = float(os.getenv("MAX_ALLOC_PERCENT", str(deps.per_symbol_alloc_pct)))
    leverage = float(decision.get("leverage") or os.getenv("DEFAULT_LEVERAGE", "5"))

    try:
        market = deps.bybit.exchange.market(deps.contract_symbol)
        symbol_min_qty = ((market.get("limits", {}) or {}).get("amount", {}) or {}).get(
            "min"
        )
    except Exception:
        market = None
        symbol_min_qty = None
    env_min_qty = os.getenv("MIN_QTY")
    min_qty = (
        float(symbol_min_qty)
        if symbol_min_qty is not None
        else float(env_min_qty if env_min_qty is not None else "0")
    )

    raw_entry_price = (
        ctx.current_price if order_type == "market" else decision.get("price")
    ) or ctx.current_price
    try:
        entry_price = float(raw_entry_price)
    except Exception:
        entry_price = float(ctx.current_price)

    use_tp = decision.get("tp")
    if use_tp is not None:
        try:
            use_tp = float(use_tp)
        except Exception:
            use_tp = None

    use_sl = decision.get("sl")
    if use_sl is not None:
        try:
            use_sl = float(use_sl)
        except Exception:
            use_sl = None

    (
        order_type,
        entry_price,
        use_tp,
        use_sl,
        leverage,
        confirm_meta,
        should_skip,
    ) = _run_confirm_step(
        deps=deps,
        decision=decision,
        ai_status=ai_status,
        side=side,
        order_type=order_type,
        entry_price=entry_price,
        use_tp=use_tp,
        use_sl=use_sl,
        leverage=leverage,
    )
    if should_skip:
        return

    max_loss_env = os.getenv("MAX_LOSS_PERCENT")
    leverage_factor = max(1.0, float(leverage or 1.0))
    if max_loss_env is not None:
        try:
            max_loss_pct = float(max_loss_env)
        except Exception:
            max_loss_pct = 80.0
    else:
        base_max_loss_pct = 4.0
        max_loss_pct = base_max_loss_pct * leverage_factor
        cap_env = os.getenv("MAX_LOSS_PERCENT_CAP")
        cap_value: Optional[float]
        if cap_env is not None:
            try:
                cap_value = float(cap_env)
            except Exception:
                cap_value = None
        else:
            cap_value = 95.0
        if cap_value is not None:
            max_loss_pct = min(max_loss_pct, cap_value)

    leveraged_cap_env = os.getenv("MAX_LEVERAGED_LOSS_PERCENT", "85")
    try:
        leveraged_cap = float(leveraged_cap_env)
    except Exception:
        leveraged_cap = 85.0
    if leveraged_cap > 0:
        leveraged_raw_cap = (
            leveraged_cap / leverage_factor if leverage_factor > 0 else leveraged_cap
        )
        if leveraged_raw_cap > 0:
            max_loss_pct = min(max_loss_pct, leveraged_raw_cap)

    max_loss_pct = max(0.0, max_loss_pct)
    if isinstance(use_sl, (int, float)) and float(use_sl) > 0:
        try:
            use_sl = enforce_max_loss_sl(
                entry_price=float(entry_price),
                proposed_sl=float(use_sl),
                position=ai_status,
                max_loss_percent=max_loss_pct,
            )
        except Exception:
            pass

    stop_price = (
        float(use_sl)
        if isinstance(use_sl, (int, float)) and float(use_sl) > 0
        else entry_price * (0.99 if side == "buy" else 1.01)
    )

    quantity = calculate_position_size(
        balance_usdt=float(balance_total or 0),
        entry_price=float(entry_price),
        stop_price=float(stop_price),
        risk_percent=risk_percent,
        max_allocation_percent=max_alloc,
        leverage=leverage,
        min_quantity=min_qty,
    )

    try:
        num_symbols_avail = max(1, len(deps.symbols))
    except Exception:
        num_symbols_avail = 1

    try:
        avail_safety = float(os.getenv("AVAILABLE_NOTIONAL_SAFETY", "0.95"))
        effective_lev_for_avail = max(1.0, float(leverage))
        free_usdt = float(balance_free or 0.0)
        per_symbol_available_notional = (
            free_usdt * effective_lev_for_avail * avail_safety
        ) / float(num_symbols_avail)
        if float(entry_price) > 0:
            target_qty_by_available = per_symbol_available_notional / float(entry_price)
            if target_qty_by_available > 0:
                quantity = min(float(quantity), float(target_qty_by_available))
    except Exception:
        pass

    try:
        positions_same_symbol = ctx.current_position or []
        last_price_fallback = float(ctx.current_price or entry_price)
        if last_price_fallback <= 0:
            last_price_fallback = float(entry_price)
        existing_notional = 0.0
        pos_max_leverage = 0.0
        for pos in positions_same_symbol:
            try:
                contract_size = pos.get("contractSize") or (
                    pos.get("info", {}) or {}
                ).get("contractSize")
                size_raw = pos.get("size") or pos.get("amount") or pos.get("contracts")
                try:
                    size_f = float(size_raw) if size_raw is not None else None
                except Exception:
                    size_f = None
                if (
                    size_f is not None
                    and pos.get("size") is None
                    and pos.get("contracts") is not None
                    and contract_size is not None
                ):
                    try:
                        size_f = size_f * float(contract_size)
                    except Exception:
                        pass
                if size_f is None:
                    continue
                mark = pos.get("markPrice") or (pos.get("info", {}) or {}).get(
                    "markPrice"
                )
                try:
                    px = float(mark) if mark is not None else float(last_price_fallback)
                except Exception:
                    px = last_price_fallback
                existing_notional += abs(float(size_f)) * float(px)
                try:
                    levp = pos.get("leverage") or (pos.get("info", {}) or {}).get(
                        "leverage"
                    )
                    if levp is not None:
                        pos_max_leverage = max(pos_max_leverage, float(levp))
                except Exception:
                    pass
            except Exception:
                continue
        effective_leverage = max(1.0, float(leverage), float(pos_max_leverage or 0.0))
        max_notional_for_symbol = (
            float(balance_total or 0)
            * (float(max_alloc) / 100.0)
            * float(effective_leverage)
        )
        remaining_notional = max(0.0, max_notional_for_symbol - existing_notional)
        if remaining_notional <= 0:
            _record_skip(
                deps,
                reason="per_symbol_cap_reached",
                decision=decision,
                meta={
                    "existing_notional": float(existing_notional),
                    "max_notional_for_symbol": float(max_notional_for_symbol),
                    "confirm": confirm_meta,
                },
            )
            return
        max_qty_by_remaining = (
            remaining_notional / float(entry_price)
            if float(entry_price) > 0
            else quantity
        )
        if max_qty_by_remaining <= 0:
            _record_skip(
                deps,
                reason="no_remaining_capacity",
                decision=decision,
                meta={
                    "existing_notional": float(existing_notional),
                    "max_notional_for_symbol": float(max_notional_for_symbol),
                    "confirm": confirm_meta,
                },
            )
            return
        quantity = min(float(quantity), float(max_qty_by_remaining))
    except Exception:
        pass

    try:
        quantity = float(
            deps.bybit.exchange.amount_to_precision(deps.contract_symbol, quantity)
        )
    except Exception:
        quantity = float(quantity)

    if float(quantity) < float(min_qty):
        _record_skip(
            deps,
            reason="below_min_lot",
            decision=decision,
            meta={
                "min_qty": float(min_qty),
                "entry_price": float(entry_price),
                "target_qty": float(quantity),
                "confirm": confirm_meta,
            },
        )
        return

    position_params = Open_Position(
        symbol=deps.contract_symbol,
        type=order_type,
        price=float(entry_price),
        side=side,
        tp=float(use_tp) if isinstance(use_tp, (int, float)) else None,
        sl=float(use_sl) if isinstance(use_sl, (int, float)) else None,
        quantity=quantity,
        leverage=leverage,
    )

    price_precision = None
    try:
        market = deps.bybit.exchange.market(deps.contract_symbol)
        price_precision = (market.get("precision", {}) or {}).get("price") or (
            market.get("limits", {}) or {}
        ).get("price", {}).get("min")
    except Exception:
        market = None

    if position_params.type == "limit":
        try:
            price_precision_digits = (
                int(price_precision) if isinstance(price_precision, int) else None
            )
            if price_precision_digits is not None:
                position_params.price = round(
                    float(position_params.price), price_precision_digits
                )
        except Exception:
            pass

    order = deps.bybit.open_position(position_params)
    if order is None:
        LOGGER.error("Order execution failed.")
        _record_skip(
            deps,
            reason="order_execution_failed",
            decision=decision,
            meta={"params": position_params.dict(), "confirm": confirm_meta},
        )
        return

    executed_qty = quantity
    order_id: Optional[Any] = None
    fill_price = float(entry_price)
    try:
        executed_qty = order.get("amount") or order.get("filled") or quantity
        order_id = order.get("id") or order.get("orderId")
        fill_price = float(
            order.get("average")
            or order.get("averagePrice")
            or order.get("price")
            or position_params.price
        )
        deps.store.record_trade(
            {
                "ts": datetime.utcnow(),
                "symbol": deps.contract_symbol,
                "side": side,
                "type": position_params.type,
                "price": float(fill_price),
                "quantity": float(executed_qty),
                "tp": position_params.tp,
                "sl": position_params.sl,
                "leverage": leverage,
                "status": "opened",
                "order_id": order_id,
                "pnl": None,
            }
        )
    except Exception as exc:
        LOGGER.error("Trade store write failed: %s", exc)

    try:
        entry_for_pct = float(fill_price)
        if entry_for_pct <= 0:
            entry_for_pct = float(position_params.price)
    except Exception:
        entry_for_pct = float(position_params.price)

    pct_info_final = _compute_tp_sl_percentages(
        entry_price=entry_for_pct,
        tp=position_params.tp,
        sl=position_params.sl,
        ai_status=ai_status,
        leverage=leverage,
    )

    meta_payload: Dict[str, Any] = {
        "decision": decision,
        "status": ai_status,
        "side": side,
        "order_type": order_type,
        "entry_price": float(entry_price),
        "actual_entry_price": entry_for_pct,
        "executed_qty": float(executed_qty),
        "tp": position_params.tp,
        "sl": position_params.sl,
        "leverage": leverage,
    }
    meta_payload.update(
        {
            "tp_percent": pct_info_final.get("tp_pct"),
            "sl_percent": pct_info_final.get("sl_pct"),
            "tp_percent_leverage": pct_info_final.get("tp_pct_leverage"),
            "sl_percent_leverage": pct_info_final.get("sl_pct_leverage"),
        }
    )
    if confirm_meta is not None:
        meta_payload["confirm"] = confirm_meta

    try:
        deps.store.record_journal(
            {
                "symbol": deps.contract_symbol,
                "entry_type": "decision",
                "content": json.dumps(
                    {
                        "status": ai_status,
                        "ai_status": ai_status,
                        "side": side,
                        "type": order_type,
                        "price": float(entry_price),
                        "actual_price": entry_for_pct,
                        "tp": position_params.tp,
                        "sl": position_params.sl,
                        "leverage": leverage,
                        "tp_percent": pct_info_final.get("tp_pct"),
                        "sl_percent": pct_info_final.get("sl_pct"),
                        "tp_percent_leverage": pct_info_final.get("tp_pct_leverage"),
                        "sl_percent_leverage": pct_info_final.get("sl_pct_leverage"),
                    },
                    ensure_ascii=False,
                ),
                "reason": decision.get("explain"),
                "meta": meta_payload,
            }
        )
        deps.store.record_journal(
            {
                "symbol": deps.contract_symbol,
                "entry_type": "action",
                "content": f"open {side} {order_type} price={float(entry_for_pct)} qty={float(executed_qty)}",
                "reason": decision.get("explain"),
                "meta": {"order_id": order_id, "confirm": confirm_meta},
                "ref_order_id": order_id,
            }
        )
    except Exception as exc:
        LOGGER.error("Journal write failed: %s", exc)
    return


def _handle_non_trade_actions(
    deps: AutomationDependencies,
    decision: Dict[str, Any],
    current_price: float,
    ai_status: str,
) -> None:
    if ai_status == "hold":
        LOGGER.info("No trading signal generated")
        try:
            meta_payload: Dict[str, Any]
            if isinstance(decision, dict):
                meta_payload = dict(decision)
            else:
                meta_payload = {"decision": decision}
            meta_payload.setdefault("status", "hold")
            deps.store.record_journal(
                {
                    "symbol": deps.contract_symbol,
                    "entry_type": "decision",
                    "content": json.dumps({"status": "hold"}, ensure_ascii=False),
                    "reason": decision.get("explain"),
                    "meta": meta_payload,
                }
            )
        except Exception as exc:
            LOGGER.error("Journal write failed: %s", exc)
        return

    if ai_status == "stop":
        LOGGER.info("stop position(close_all)")
        try:
            positions = deps.bybit.get_positions_by_symbol(deps.contract_symbol) or []
            last = deps.bybit.get_last_price(deps.contract_symbol) or current_price
            for p in positions:
                try:
                    entry = float(
                        p.get("entryPrice") or p.get("info", {}).get("avgPrice") or 0
                    )
                    amount = float(
                        p.get("contracts") or p.get("amount") or p.get("size") or 0
                    )
                    side = p.get("side") or p.get("info", {}).get("side")
                    if amount <= 0 or entry <= 0 or not side:
                        continue
                    pnl = (
                        (last - entry) * amount
                        if side == "long"
                        else (entry - last) * amount
                    )
                    deps.store.record_trade(
                        {
                            "ts": datetime.utcnow(),
                            "symbol": p.get("symbol"),
                            "side": side,
                            "type": "market",
                            "price": last,
                            "quantity": amount,
                            "tp": None,
                            "sl": None,
                            "leverage": None,
                            "status": "closed",
                            "order_id": None,
                            "pnl": float(pnl),
                        }
                    )
                except Exception:
                    pass
        except Exception as exc:
            LOGGER.error("PNL calc failed: %s", exc)
        deps.bybit.close_symbol_positions(deps.contract_symbol)
        try:
            deps.store.record_journal(
                {
                    "symbol": deps.contract_symbol,
                    "entry_type": "action",
                    "content": "close_all",
                    "reason": decision.get("explain") or "stop signal",
                    "meta": decision,
                }
            )
        except Exception as exc:
            LOGGER.error("Journal write failed: %s", exc)
        return

    LOGGER.info("Unknown AI status: %s", ai_status)


def _record_skip(
    deps: AutomationDependencies,
    *,
    reason: str,
    decision: Dict[str, Any],
    meta: Optional[Dict[str, Any]] = None,
) -> None:
    try:
        deps.store.record_journal(
            {
                "symbol": deps.contract_symbol,
                "entry_type": "decision",
                "content": json.dumps(
                    {"status": "skip", "reason": reason},
                    ensure_ascii=False,
                ),
                "reason": decision.get("explain"),
                "meta": (
                    {"decision": decision, "details": meta}
                    if meta is not None
                    else decision
                ),
            }
        )
    except Exception:
        pass


def automation_for_symbol(
    symbol_usdt: str, *, symbols: Sequence[str] | None = None
) -> None:
    try:
        deps = _init_dependencies(symbol_usdt, symbols)
        ctx = _gather_prompt_context(deps)
        prompt = _build_prompt(deps, ctx)
        decision = _request_trade_decision(deps, prompt, ctx)
        if _handle_close_now(deps, ctx, decision):
            return
        _execute_trade(deps, ctx, decision)
    except Exception as exc:
        LOGGER.error("Error in automation: %s", exc)


def run_loss_review(
    symbols: Sequence[str] | None = None, since_minutes: int = 600
) -> None:
    del symbols
    store = TradeStore(
        StorageConfig(
            mysql_url=os.getenv("MYSQL_URL"),
            sqlite_path=os.getenv("SQLITE_PATH"),
        )
    )
    journal_service = JournalService(store, AIProvider())
    journal_service.review_losing_trades(since_minutes=since_minutes)
