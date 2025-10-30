"""Automated trading workflow orchestration."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

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
    recent_reports_text: str
    reviews_text: str
    since_open_text: str


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
            pct = raw.get("percentage") or (raw.get("info", {}) or {}).get(
                "unrealisedPnlPcnt"
            )
            tp = raw.get("takeProfit") or (raw.get("info", {}) or {}).get("takeProfit")
            sl = raw.get("stopLoss") or (raw.get("info", {}) or {}).get("stopLoss")
            lev = raw.get("leverage") or (raw.get("info", {}) or {}).get("leverage")

            def _fmt(value: Any) -> str:
                try:
                    return f"{float(value):.6f}"
                except Exception:
                    return str(value)

            lines.append(
                "side={side}, size={size}, entry={entry}, last={last}, "
                "tp={tp}, sl={sl}, lev={lev}, unreal={unreal} ({pct}%)".format(
                    side=side,
                    size=_fmt(size_f) if size_f is not None else "n/a",
                    entry=_fmt(entry_f) if entry_f is not None else "n/a",
                    last=_fmt(last),
                    tp=_fmt(tp) if tp is not None else "n/a",
                    sl=_fmt(sl) if sl is not None else "n/a",
                    lev=_fmt(lev) if lev is not None else "n/a",
                    unreal=_fmt(unreal) if unreal is not None else "n/a",
                    pct=_fmt(pct),
                )
            )
        except Exception:
            continue
    return lines, primary_side


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
    )


def _build_prompt(deps: AutomationDependencies, ctx: PromptContext) -> str:
    now_utc = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    prompt = (
        f"당신은 암호화폐 트레이딩 보조 도구입니다. 한국어로 답변하세요.\n"
        f"당신은 1~최대 75배까지 레버리지를 사용할 수 있습니다.\n"
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
) -> Dict[str, Any]:
    try:
        parsed = deps.ai_provider.decide_json(prompt)
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
        response = deps.ai_provider.decide(prompt)
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
        if ai_status == "long":
            tp_pct = (tp_v - e) / e * 100.0
            sl_pct = (e - sl_v) / e * 100.0
        else:
            tp_pct = (e - tp_v) / e * 100.0
            sl_pct = (sl_v - e) / e * 100.0

        confirm_prompt = (
            "당신이 제안한 주문 파라미터를 최종 확인하세요. JSON만 응답. 한국어로.\n"
            f"심볼: {deps.contract_symbol}\n"
            f"포지션: {ai_status} (내부 side={side})\n"
            f"진입가(entry): {float(e)}\n"
            f"TP: {float(tp_v)} (예상 수익률: {tp_pct:.4f}%)\n"
            f"SL: {float(sl_v)} (예상 손실률: {sl_pct:.4f}%)\n"
            f"레버리지: {float(leverage)}x\n"
            "필수: confirm(boolean). 선택: tp, sl, price, buy_now, leverage, explain.\n"
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

    max_loss_pct = float(os.getenv("MAX_LOSS_PERCENT", "80"))
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

    meta_payload: Dict[str, Any] = {"decision": decision}
    if confirm_meta is not None:
        meta_payload["confirm"] = confirm_meta

    try:
        deps.store.record_journal(
            {
                "symbol": deps.contract_symbol,
                "entry_type": "decision",
                "content": json.dumps(
                    {
                        "status": side,
                        "ai_status": ai_status,
                        "type": order_type,
                        "price": float(entry_price),
                        "tp": position_params.tp,
                        "sl": position_params.sl,
                        "leverage": leverage,
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
                "content": f"open {side} {order_type} price={float(entry_price)} qty={float(executed_qty)}",
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
            deps.store.record_journal(
                {
                    "symbol": deps.contract_symbol,
                    "entry_type": "decision",
                    "content": json.dumps({"status": "hold"}, ensure_ascii=False),
                    "reason": decision.get("explain"),
                    "meta": decision,
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
        decision = _request_trade_decision(deps, prompt)
        if _handle_close_now(deps, ctx, decision):
            return
        _execute_trade(deps, ctx, decision)
    except Exception as exc:
        LOGGER.error("Error in automation: %s", exc)


def run_loss_review(
    symbols: Sequence[str] | None = None, since_minutes: int = 600
) -> None:
    store = TradeStore(
        StorageConfig(
            mysql_url=os.getenv("MYSQL_URL"),
            sqlite_path=os.getenv("SQLITE_PATH"),
        )
    )
    journal_service = JournalService(store, AIProvider())
    journal_service.review_losing_trades(since_minutes=since_minutes)
