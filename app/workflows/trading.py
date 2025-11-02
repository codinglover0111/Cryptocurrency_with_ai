"""Automated trading workflow orchestration."""

# pylint: disable=broad-except
# ruff: noqa: E722, BLE001
from __future__ import annotations

import json
import logging
import math
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import ta

from utils import BybitUtils, Open_Position, bybit_utils, make_to_object
from utils.price_utils import dataframe_to_candlestick_base64
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


class InvalidDecisionError(Exception):
    """Raised when the LLM trade decision payload is invalid."""


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
    recent_reports_text: str
    reviews_text: str
    since_open_text: str
    chart_images: Dict[str, str] = field(default_factory=dict)
    rsi_4h: Optional[float] = None
    rsi_1h: Optional[float] = None
    rsi_15m: Optional[float] = None


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


def _summarize_positions(
    positions: Iterable[Dict[str, Any]],
    contract_symbol: str,
    current_price: float,
) -> Tuple[List[str], Optional[str]]:
    lines: List[str] = []
    primary_side: Optional[str] = None
    last_fallback = float(current_price or 0)

    for raw in positions or []:
        try:
            sym = raw.get("symbol") or (raw.get("info", {}) or {}).get("symbol")
            if sym != contract_symbol:
                continue
            side = raw.get("side") or (raw.get("info", {}) or {}).get("side")
            if primary_side is None and side:
                primary_side = side
            entry = raw.get("entryPrice") or (raw.get("info", {}) or {}).get("avgPrice")
            contract_size = raw.get("contractSize") or (raw.get("info", {}) or {}).get(
                "contractSize"
            )
            size_raw = raw.get("size") or raw.get("contracts") or raw.get("amount")
            try:
                size_f = float(size_raw) if size_raw is not None else None
            except Exception:
                size_f = None
            if (
                size_f is not None
                and raw.get("size") is None
                and raw.get("contracts") is not None
                and contract_size is not None
            ):
                try:
                    size_f = size_f * float(contract_size)
                except Exception:
                    pass
            mark = raw.get("markPrice") or (raw.get("info", {}) or {}).get("markPrice")
            try:
                last = float(mark) if mark is not None else float(last_fallback)
            except Exception:
                last = last_fallback
            try:
                entry_f = float(entry) if entry is not None else None
            except Exception:
                entry_f = None
            unreal = raw.get("unrealizedPnl") or (raw.get("info", {}) or {}).get(
                "unrealisedPnl"
            )
            pct_raw = raw.get("percentage") or (raw.get("info", {}) or {}).get(
                "unrealisedPnlPcnt"
            )
            tp = raw.get("takeProfit") or (raw.get("info", {}) or {}).get("takeProfit")
            sl = raw.get("stopLoss") or (raw.get("info", {}) or {}).get("stopLoss")
            lev = raw.get("leverage") or (raw.get("info", {}) or {}).get("leverage")

            def _safe_float(value: Any) -> Optional[float]:
                try:
                    if value is None:
                        return None
                    return float(value)
                except Exception:
                    return None

            lev_f = _safe_float(lev)
            pct_raw_f = _safe_float(pct_raw)

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

            pct_baseline: Optional[float] = None
            pct_leverage: Optional[float] = None

            if directional_pct is not None:
                pct_baseline = directional_pct
                if lev_f is not None and lev_f != 0.0:
                    pct_leverage = directional_pct * lev_f
                else:
                    pct_leverage = directional_pct
            elif pct_raw_f is not None:
                pct_leverage = pct_raw_f
                if lev_f is not None and lev_f not in (0.0, -0.0):
                    try:
                        pct_baseline = pct_raw_f / lev_f
                    except Exception:
                        pct_baseline = None
                else:
                    pct_baseline = None
            else:
                pct_baseline = None
                pct_leverage = None

            def _fmt(value: Any) -> str:
                try:
                    return f"{float(value):.6f}"
                except Exception:
                    return str(value)

            side_fmt = side if side is not None else "n/a"
            size_fmt = _fmt(size_f) if size_f is not None else "n/a"
            entry_fmt = _fmt(entry_f) if entry_f is not None else "n/a"
            last_fmt = _fmt(last)
            tp_fmt = _fmt(tp) if tp is not None else "n/a"
            sl_fmt = _fmt(sl) if sl is not None else "n/a"
            lev_fmt = _fmt(lev) if lev is not None else "n/a"
            unreal_fmt = _fmt(unreal) if unreal is not None else "n/a"
            pct_fmt = _fmt(pct_leverage) if pct_leverage is not None else "n/a"
            pct_raw_fmt = _fmt(pct_baseline) if pct_baseline is not None else "n/a"

            lines.append(
                f"side={side_fmt}, size={size_fmt}, entry={entry_fmt}, last={last_fmt}, "
                f"tp={tp_fmt}, sl={sl_fmt}, lev={lev_fmt}, unreal_PNL={unreal_fmt} "
                f"(ROI_leverage={pct_fmt}%, ROI_baseline={pct_raw_fmt}%)"
            )
        except Exception:
            continue
    return lines, primary_side


def _latest_rsi_value(df: Any, window: int = 14) -> Optional[float]:
    try:
        if df is None or getattr(df, "empty", True):
            return None
        rsi_series = ta.momentum.rsi(df["close"], window=window)
        if rsi_series is None or rsi_series.empty:
            return None
        value = rsi_series.iloc[-1]
        if value is None:
            return None
        value_f = float(value)
        if math.isnan(value_f):
            return None
        return round(value_f, 2)
    except Exception:
        return None


def _gather_prompt_context(deps: AutomationDependencies) -> PromptContext:
    price_helper = bybit_utils(deps.spot_symbol, "4h", 100)
    df_4h = price_helper.get_ohlcv()
    csv_4h = df_4h.to_csv()
    price_helper.set_timeframe("1h")
    df_1h = price_helper.get_ohlcv()
    csv_1h = df_1h.to_csv()
    price_helper.set_timeframe("15m")
    df_15m = price_helper.get_ohlcv()
    csv_15m = df_15m.to_csv()
    current_price = df_15m["close"].iloc[-1]

    rsi_4h = _latest_rsi_value(df_4h)
    rsi_1h = _latest_rsi_value(df_1h)
    rsi_15m = _latest_rsi_value(df_15m)

    chart_images: Dict[str, str] = {}
    if deps.ai_provider.provider == "gemini":
        for timeframe, frame_df in ("4h", df_4h), ("1h", df_1h), ("15m", df_15m):
            try:
                image_b64 = dataframe_to_candlestick_base64(
                    frame_df.tail(120),
                    deps.spot_symbol,
                    timeframe,
                )
                if image_b64:
                    chart_images[timeframe] = image_b64
            except Exception as exc:
                LOGGER.warning("%s 차트 생성 실패: %s", timeframe, exc)

    current_position = deps.bybit.get_positions_by_symbol(deps.contract_symbol) or []
    position_lines, pos_side = _summarize_positions(
        current_position, deps.contract_symbol, current_price
    )

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

    recent_reports_text = ""
    try:
        recent_df = deps.store.fetch_journals(
            symbol=deps.contract_symbol,
            types=["decision", "review"],
            limit=10,
            ascending=False,
        )
        if recent_df is not None and not recent_df.empty:
            lines = []
            for _, row in recent_df.iterrows():
                ts = row.get("ts")
                ts_str = (
                    ts.strftime("%m-%d %H:%M") if hasattr(ts, "strftime") else str(ts)
                )
                lines.append(
                    f"[{ts_str}] ({row.get('entry_type')}) {row.get('reason') or ''} | {row.get('content') or ''}"
                )
            recent_reports_text = "\n".join(lines)
    except Exception:
        recent_reports_text = ""

    reviews_text = deps.journal_service.format_trade_reviews_for_prompt(
        deps.contract_symbol
    )

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
        recent_reports_text=recent_reports_text,
        reviews_text=reviews_text,
        since_open_text=since_open_text,
        chart_images=chart_images,
        rsi_4h=rsi_4h,
        rsi_1h=rsi_1h,
        rsi_15m=rsi_15m,
    )


def _build_prompt(deps: AutomationDependencies, ctx: PromptContext) -> str:
    now_utc = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

    def _format_rsi(value: Optional[float]) -> str:
        try:
            if value is None:
                return "n/a"
            return f"{float(value):.2f}"
        except Exception:
            return "n/a"

    rsi_4h_str = _format_rsi(ctx.rsi_4h)
    rsi_1h_str = _format_rsi(ctx.rsi_1h)
    rsi_15m_str = _format_rsi(ctx.rsi_15m)

    prompt = (
        f"당신은 세계 최고의 암호화폐 트레이더입니다.\n"
        "당신은 4시간 봉과 1시간 봉을 참고하여 15분 봉을 중심을 이용한 단타 트레이더입니다.\n"
        "낮은 승률에 고레버리지를 이용해 최대 수익을 추구하는 트레이더입니다.\n"
        f"이미 진입한 포지션의 레버리지는 조절할 수 없습니다.\n"
        f"당신은 최소 5배에서 최대 75배까지 레버리지를 사용할 수 있습니다.\n"
        "기존 포지션의 TP/SL만 조정하려면 update_existing=true 로 표시하고 tp/sl 값만 제시하세요. 이때 leverage는 비워 두세요.\n"
        f"현재 UTC 시간: {now_utc}\n"
        f"심볼: {deps.contract_symbol} (spot={deps.spot_symbol})\n"
        f"현재가: {ctx.current_price}\n"
        f"심볼당 기본 배분 비율: {deps.per_symbol_alloc_pct:.2f}%\n"
        "[CSV_4H]\n"
        f"{ctx.csv_4h}\n"
        "[/CSV_4H]\n"
        "[CSV_1H]\n"
        f"{ctx.csv_1h}\n"
        "[/CSV_1H]\n"
        "[CSV_15M]\n"
        f"{ctx.csv_15m}\n"
        "[/CSV_15M]\n"
        f"[RSI_4H] {rsi_4h_str} [/RSI_4H]\n"
        f"[RSI_1H] {rsi_1h_str} [/RSI_1H]\n"
        f"[RSI_15M] {rsi_15m_str} [/RSI_15M]\n"
        "[CURRENT_POSITIONS]\n"
        + (
            "\n".join(ctx.current_positions_lines)
            if ctx.current_positions_lines
            else "(none)"
        )
        + "\n[/CURRENT_POSITIONS]\n"
    )

    if ctx.reviews_text:
        prompt += "[RECENT_REVIEWS]\n" + ctx.reviews_text + "\n[/RECENT_REVIEWS]\n"
    if ctx.recent_reports_text:
        prompt += (
            "[RECENT_REPORTS]\n" + ctx.recent_reports_text + "\n[/RECENT_REPORTS]\n"
        )
    if ctx.journal_today_text:
        prompt += (
            "[JOURNALS_TODAY]\n" + ctx.journal_today_text + "\n[/JOURNALS_TODAY]\n"
        )
    if ctx.pos_side is not None and ctx.since_open_text:
        prompt += "[SINCE_LAST_OPEN]\n" + ctx.since_open_text + "\n[/SINCE_LAST_OPEN]\n"

    prompt += (
        "Choose one of: hold, short, long, or stop.\n"
        "Return your decision as JSON with the following fields: order type (market/limit), "
        "price, stop loss (sl), take profit (tp), buy_now (boolean), leverage (number), update_existing (boolean).\n"
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
    ctx: PromptContext,
) -> Dict[str, Any]:
    images_payload: Optional[List[Dict[str, Any]]] = None
    if deps.ai_provider.provider == "gemini" and ctx.chart_images:
        images_payload = []
        for timeframe in ("4h", "1h", "15m"):
            chart_b64 = ctx.chart_images.get(timeframe)
            if not chart_b64:
                continue
            images_payload.append(
                {
                    "b64": chart_b64,
                    "mime": "image/png",
                    "metadata": {
                        "symbol": deps.spot_symbol,
                        "contract_symbol": deps.contract_symbol,
                        "timeframe": timeframe,
                        "type": "candlestick",
                    },
                }
            )
        if not images_payload:
            images_payload = None

    try:
        parsed = deps.ai_provider.decide_json(prompt, images=images_payload)
        LOGGER.info(
            json.dumps(
                {
                    "event": "llm_response_parsed",
                    "provider": os.getenv("AI_PROVIDER", "gemini").lower(),
                    "parsed": parsed,
                },
                ensure_ascii=False,
            )
        )
        return parsed
    except Exception:
        response = deps.ai_provider.decide(prompt, images=images_payload)
        try:
            LOGGER.info(
                json.dumps(
                    {
                        "event": "llm_response_raw",
                        "provider": os.getenv("AI_PROVIDER", "gemini").lower(),
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
                        "provider": os.getenv("AI_PROVIDER", "gemini").lower(),
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


def _normalize_position_side(value: Any) -> Optional[str]:
    try:
        if value is None:
            return None
        side = str(value).strip().lower()
    except Exception:
        return None
    if side in {"long", "buy"}:
        return "long"
    if side in {"short", "sell"}:
        return "short"
    return None


def _compute_max_loss_percent(leverage: float) -> float:
    leverage_factor = max(1.0, float(leverage or 1.0))
    max_loss_env = os.getenv("MAX_LOSS_PERCENT")
    if max_loss_env is not None:
        try:
            max_loss_pct = float(max_loss_env)
        except Exception:
            max_loss_pct = 80.0
    else:
        base_max_loss_pct = 4.0
        max_loss_pct = base_max_loss_pct * leverage_factor
        cap_env = os.getenv("MAX_LOSS_PERCENT_CAP")
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

    return max(0.0, max_loss_pct)


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


def _handle_update_existing_positions(
    *,
    deps: AutomationDependencies,
    ctx: PromptContext,
    decision: Dict[str, Any],
    ai_status: str,
    use_tp: Optional[float],
    use_sl: Optional[float],
) -> None:
    tp_val = float(use_tp) if isinstance(use_tp, (int, float)) else None
    sl_val = float(use_sl) if isinstance(use_sl, (int, float)) else None

    if tp_val is None and sl_val is None:
        LOGGER.info("update_existing 요청이지만 TP/SL 값이 없습니다.")
        _record_skip(
            deps,
            reason="update_missing_tp_sl",
            decision=decision,
            meta={"decision": decision},
            reason_text="update_existing=true 이지만 tp/sl 값이 제공되지 않았습니다.",
        )
        return

    positions = list(ctx.current_position or [])
    if not positions:
        try:
            positions = deps.bybit.get_positions_by_symbol(deps.contract_symbol) or []
        except Exception:
            positions = []

    if not positions:
        LOGGER.info("update_existing 요청이지만 활성 포지션이 없습니다.")
        _record_skip(
            deps,
            reason="update_no_position",
            decision=decision,
            meta={"decision": decision},
            reason_text="update_existing=true 이지만 활성 포지션이 없습니다.",
        )
        return

    target_side = _normalize_position_side(ai_status)

    def _safe_float(value: Any) -> Optional[float]:
        try:
            if value is None:
                return None
            return float(value)
        except Exception:
            return None

    def _extract_position_idx(position: Dict[str, Any]) -> Optional[int]:
        candidates = [
            position.get("positionIdx"),
            (position.get("info", {}) or {}).get("positionIdx"),
            (position.get("info", {}) or {}).get("position_idx"),
        ]
        for cand in candidates:
            try:
                if cand is None:
                    continue
                idx_val = int(float(cand))
                if idx_val >= 0:
                    return idx_val
            except Exception:
                continue
        return None

    matching: List[Tuple[Dict[str, Any], Optional[str]]] = []
    for pos in positions:
        info = pos.get("info") if isinstance(pos.get("info"), dict) else {}
        pos_side = _normalize_position_side(pos.get("side"))
        if pos_side is None:
            pos_side = _normalize_position_side((info or {}).get("side"))
        if target_side is not None and pos_side is not None and pos_side != target_side:
            continue
        matching.append((pos, pos_side))

    if not matching:
        matching = [
            (pos, _normalize_position_side(pos.get("side"))) for pos in positions
        ]

    updates: List[Dict[str, Any]] = []
    success = False

    for pos, pos_side in matching:
        info = pos.get("info") if isinstance(pos.get("info"), dict) else {}

        entry_price: Optional[float] = None
        for candidate in (
            pos.get("entryPrice"),
            info.get("avgPrice"),
            info.get("entryPrice"),
        ):
            entry_price = _safe_float(candidate)
            if entry_price is not None and entry_price > 0:
                break
        if entry_price is None or entry_price <= 0:
            entry_price = _safe_float(ctx.current_price)

        leverage_val: float = 1.0
        for candidate in (pos.get("leverage"), info.get("leverage")):
            cand_val = _safe_float(candidate)
            if cand_val is not None and cand_val > 0:
                leverage_val = cand_val
                break

        applied_sl = sl_val
        if (
            sl_val is not None
            and entry_price is not None
            and entry_price > 0
            and pos_side is not None
        ):
            try:
                applied_sl = enforce_max_loss_sl(
                    entry_price=float(entry_price),
                    proposed_sl=float(sl_val),
                    position=pos_side,
                    max_loss_percent=_compute_max_loss_percent(leverage_val),
                )
            except Exception:
                applied_sl = sl_val

        try:
            result = deps.bybit.update_symbol_tpsl(
                deps.contract_symbol,
                take_profit=tp_val,
                stop_loss=applied_sl,
                side=pos_side,
                position_idx=_extract_position_idx(pos),
            )
        except Exception as exc:
            LOGGER.error("TP/SL 업데이트 실패: %s", exc)
            result = {"status": "error", "error": str(exc)}

        status = (result or {}).get("status") if isinstance(result, dict) else None
        success = success or status in {"ok", "noop"}

        meta_result: Any
        if isinstance(result, dict):
            try:
                meta_result = json.loads(json.dumps(result, default=str))
            except Exception:
                meta_result = {"status": result.get("status"), "error": str(result)}
        else:
            meta_result = str(result)

        updates.append(
            {
                "side": pos_side,
                "position_idx": _extract_position_idx(pos),
                "entry_price": entry_price,
                "leverage": leverage_val,
                "requested_tp": tp_val,
                "requested_sl": sl_val,
                "applied_sl": applied_sl,
                "result": meta_result,
            }
        )

    if not success:
        LOGGER.error("update_existing 요청으로 TP/SL을 수정하지 못했습니다.")
        _record_skip(
            deps,
            reason="update_failed",
            decision=decision,
            meta={"decision": decision, "updates": updates},
            reason_text="TP/SL 업데이트 API 호출에 실패했습니다.",
        )
        return

    try:
        deps.store.record_journal(
            {
                "symbol": deps.contract_symbol,
                "entry_type": "action",
                "content": "update_tp_sl",
                "reason": decision.get("explain"),
                "meta": {
                    "decision": decision,
                    "tp": tp_val,
                    "sl": sl_val,
                    "updates": updates,
                },
            }
        )
    except Exception as exc:
        LOGGER.error("Journal write failed: %s", exc)

    LOGGER.info("기존 포지션 TP/SL 업데이트 완료: %s", updates)


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
            "confirm=false이면 반드시 explain에 거부 사유를 한국어로 작성하세요.\n"
            "확신하면 confirm=true. 수정이 필요하면 값을 조정해 응답하세요."
        )
        confirm = deps.ai_provider.confirm_trade_json(confirm_prompt)
        confirm_meta = confirm
        LOGGER.info(
            json.dumps(
                {
                    "event": "llm_confirm_response_parsed",
                    "provider": os.getenv("AI_PROVIDER", "gemini").lower(),
                    "parsed": confirm,
                },
                ensure_ascii=False,
            )
        )
        if not bool(confirm.get("confirm")):
            skip_reason = str(confirm.get("explain") or "").strip()
            if not skip_reason:
                skip_reason = "확인 단계에서 거부 사유가 제공되지 않았습니다."
            _record_skip(
                deps,
                reason="skip_after_confirm",
                decision=decision,
                meta={"first": decision, "confirm": confirm},
                reason_text=skip_reason,
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


def _extract_ai_status(decision: Any) -> str:
    if not isinstance(decision, dict):
        LOGGER.warning("Trade decision payload is not a dict: %r", decision)
        raise InvalidDecisionError("LLM trade decision must be a JSON object")

    raw_status = decision.get("Status")
    if not raw_status:
        raw_status = decision.get("status")

    if isinstance(raw_status, str):
        normalized = raw_status.strip().lower()
    elif raw_status is None:
        normalized = ""
    else:
        normalized = str(raw_status).strip().lower()

    if normalized in {"watch", "watch/hold"}:
        normalized = "hold"

    allowed_statuses = {"hold", "short", "long", "stop"}
    if not normalized or normalized not in allowed_statuses:
        LOGGER.warning("Invalid AI status in decision: %r", decision)
        raise InvalidDecisionError(f"Unknown AI status: {raw_status}")

    decision["status"] = normalized
    if (
        not isinstance(decision.get("Status"), str)
        or not str(decision.get("Status")).strip()
    ):
        decision["Status"] = normalized

    return normalized


def _require_explain(decision: Any) -> str:
    if not isinstance(decision, dict):
        LOGGER.warning("Trade decision payload is not a dict: %r", decision)
        raise InvalidDecisionError("LLM trade decision must be a JSON object")

    explain_value = decision.get("explain") or decision.get("Explain")
    if isinstance(explain_value, str):
        explain_normalized = explain_value.strip()
    elif explain_value is None:
        explain_normalized = ""
    else:
        explain_normalized = str(explain_value).strip()

    if not explain_normalized:
        LOGGER.warning("LLM trade decision missing explain field: %r", decision)
        raise InvalidDecisionError(
            "LLM trade decision must include a non-empty 'explain' field"
        )

    if len(explain_normalized) <= 30:
        LOGGER.warning(
            "LLM trade decision explain too short (%s chars): %r",
            len(explain_normalized),
            decision,
        )
        raise InvalidDecisionError(
            "LLM trade decision must include an 'explain' field longer than 30 characters"
        )

    decision["explain"] = explain_normalized
    if "Explain" in decision:
        decision["Explain"] = explain_normalized
    return explain_normalized


def _execute_trade(
    deps: AutomationDependencies,
    ctx: PromptContext,
    decision: Dict[str, Any],
) -> None:
    ai_status = _extract_ai_status(decision)
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

    try:
        num_symbols_avail = max(1, len(deps.symbols))
    except Exception:
        num_symbols_avail = 1

    equity_for_sizing = float(balance_total or 0)
    try:
        free_equity = float(balance_free or 0)
        if free_equity > 0:
            equity_for_sizing = min(equity_for_sizing, free_equity)
    except Exception:
        pass

    per_symbol_equity = equity_for_sizing / float(num_symbols_avail)

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

    update_existing = _normalize_bool(
        decision.get("update_existing")
        or decision.get("modify_existing")
        or decision.get("update_tp_sl")
    )
    if update_existing:
        _handle_update_existing_positions(
            deps=deps,
            ctx=ctx,
            decision=decision,
            ai_status=ai_status,
            use_tp=use_tp,
            use_sl=use_sl,
        )
        return

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

    max_loss_pct = _compute_max_loss_percent(leverage)
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
        balance_usdt=float(per_symbol_equity),
        entry_price=float(entry_price),
        stop_price=float(stop_price),
        risk_percent=risk_percent,
        max_allocation_percent=max_alloc,
        leverage=leverage,
        min_quantity=min_qty,
    )

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

    LOGGER.warning("Unknown AI status encountered: %s", ai_status)
    raise InvalidDecisionError(f"Unknown AI status: {ai_status}")


def _record_skip(
    deps: AutomationDependencies,
    *,
    reason: str,
    decision: Dict[str, Any],
    meta: Optional[Dict[str, Any]] = None,
    reason_text: Optional[str] = None,
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
                "reason": reason_text or decision.get("explain"),
                "meta": (
                    {"decision": decision, "details": meta}
                    if meta is not None
                    else decision
                ),
            }
        )
    except Exception:
        pass


def _record_journal_fetch_error(
    deps: Optional[AutomationDependencies],
    symbol_usdt: str,
    symbols: Sequence[str] | None,
    exc: Exception,
) -> None:
    try:
        use_deps = deps or _init_dependencies(symbol_usdt, symbols)
    except Exception as init_exc:
        LOGGER.error(
            "Failed to prepare dependencies for journal error logging: %s",
            init_exc,
        )
        return

    try:
        use_deps.store.record_journal(
            {
                "symbol": use_deps.contract_symbol,
                "entry_type": "error",
                "content": "저널 조회에 오류가 발생했습니다.",
                "reason": str(exc),
                "meta": {"error": str(exc)},
            }
        )
    except Exception as journal_exc:
        LOGGER.error(
            "Journal write failed while recording automation error: %s",
            journal_exc,
        )


def automation_for_symbol(
    symbol_usdt: str, *, symbols: Sequence[str] | None = None
) -> None:
    delays = [5, 10, 60]

    for attempt in range(len(delays) + 1):
        deps: Optional[AutomationDependencies] = None
        try:
            deps = _init_dependencies(symbol_usdt, symbols)
            ctx = _gather_prompt_context(deps)
            prompt = _build_prompt(deps, ctx)
            decision = _request_trade_decision(deps, prompt, ctx)
            _require_explain(decision)
            if _handle_close_now(deps, ctx, decision):
                return
            _execute_trade(deps, ctx, decision)
            return
        except Exception as exc:
            LOGGER.error(
                "Error in automation (attempt %s/%s): %s",
                attempt + 1,
                len(delays) + 1,
                exc,
            )
            if attempt == len(delays):
                LOGGER.error("저널 조회에 오류가 발생했습니다.")
                _record_journal_fetch_error(deps, symbol_usdt, symbols, exc)
                return

            delay = delays[attempt]
            LOGGER.info(
                "Retrying automation for %s in %s seconds",
                symbol_usdt,
                delay,
            )
            time.sleep(delay)


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
