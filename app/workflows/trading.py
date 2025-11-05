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
    current_positions: List[Dict[str, Any]]
    pos_side: Optional[str]
    current_positions_lines: List[str]
    journal_today_text: str
    recent_reports_text: str
    reviews_text: str
    since_open_text: str
    chart_images: Dict[str, str] = field(default_factory=dict)


def _safe_float(value: Any) -> Optional[float]:
    """Return ``float`` representation of ``value`` or ``None`` on failure."""

    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _values_almost_equal(
    first: Any, second: Any, *, abs_tol: float = 1e-8, rel_tol: float = 1e-9
) -> bool:
    """Return True when two numeric values are effectively identical."""

    try:
        if first is None or second is None:
            return False
        return math.isclose(
            float(first), float(second), rel_tol=rel_tol, abs_tol=abs_tol
        )
    except Exception:
        return False


def _extract_position_symbol(position: Dict[str, Any]) -> Optional[str]:
    """Best-effort extraction of the symbol from a position payload."""

    if not isinstance(position, dict):
        return None

    symbol = position.get("symbol")
    if symbol:
        return str(symbol)

    info = position.get("info")
    if isinstance(info, dict):
        symbol_info = info.get("symbol")
        if symbol_info:
            return str(symbol_info)

    return None


def _result_indicates_no_change(result: Any) -> bool:
    """Heuristically detect exchange responses that mean 'no TP/SL change applied'."""

    if result is None:
        return False

    snippets: List[str] = []
    if isinstance(result, dict):
        for key in (
            "error",
            "message",
            "msg",
            "retMsg",
            "ret_msg",
            "detail",
            "reason",
            "status",
        ):
            value = result.get(key)
            if value:
                snippets.append(str(value))
        if not snippets:
            try:
                snippets.append(json.dumps(result, ensure_ascii=False))
            except Exception:
                snippets.append(str(result))
    else:
        snippets.append(str(result))

    lowered = " ".join(snippets).lower()
    keywords = (
        "no change",
        "unchanged",
        "no modification",
        "nothing to update",
        "already set",
        "same as",
    )
    return any(keyword in lowered for keyword in keywords)


def _format_decimal(value: Any, *, default: str = "n/a", precision: int = 6) -> str:
    """Format numeric ``value`` with the given precision, handling errors gracefully."""

    try:
        return f"{float(value):.{precision}f}"
    except Exception:
        if value is None:
            return default if default is not None else "n/a"
        return str(value)


def _format_journal_dataframe(
    df: Any,
    *,
    timestamp_format: str,
    empty_value: str = "",
) -> str:
    """Convert a journal dataframe into a newline-joined textual summary."""

    if df is None or getattr(df, "empty", True):
        return empty_value

    lines: List[str] = []
    for _, row in df.iterrows():
        ts = row.get("ts")
        ts_str = ts.strftime(timestamp_format) if hasattr(ts, "strftime") else str(ts)
        entry_type = (row.get("entry_type") or "").strip()
        reason = (row.get("reason") or "").strip()
        content = (row.get("content") or "").strip()
        line = f"[{ts_str}]"
        if entry_type:
            line += f" ({entry_type})"
        if reason:
            line += f" {reason}"
        if content:
            line += f" | {content}"
        lines.append(line.strip())

    return "\n".join(lines)


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
    """Build and return dependency bundle required for a trading cycle."""

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
    """Summarise active positions for logging and prompt context generation."""

    lines: List[str] = []
    primary_side: Optional[str] = None
    last_fallback = _safe_float(current_price)

    for raw in positions or []:
        try:
            info = raw.get("info") if isinstance(raw.get("info"), dict) else {}
            sym = _extract_position_symbol(raw)

            if sym is None or sym != contract_symbol:
                continue

            side = raw.get("side") or (info or {}).get("side")
            if primary_side is None and side:
                primary_side = side

            contract_size = raw.get("contractSize") or (info or {}).get("contractSize")

            size_candidates = [
                raw.get("size"),
                raw.get("contracts"),
                raw.get("amount"),
            ]
            if isinstance(info, dict):
                size_candidates.extend(
                    [
                        info.get("size"),
                        info.get("contracts"),
                        info.get("amount"),
                        info.get("positionAmt"),
                    ]
                )

            size_f: Optional[float] = None
            for cand in size_candidates:
                value = _safe_float(cand)
                if value is not None:
                    size_f = value
                    break

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

            if size_f is None or abs(size_f) <= 1e-12:
                continue

            entry_candidates = [
                raw.get("entryPrice"),
                (info or {}).get("avgPrice"),
                (info or {}).get("entryPrice"),
            ]
            entry_f: Optional[float] = None
            for cand in entry_candidates:
                candidate_value = _safe_float(cand)
                if candidate_value is not None:
                    entry_f = candidate_value
                    break

            mark_candidates = [
                raw.get("markPrice"),
                (info or {}).get("markPrice"),
                raw.get("lastPrice"),
                (info or {}).get("lastPrice"),
                raw.get("last"),
                (info or {}).get("last"),
                raw.get("price"),
                (info or {}).get("price"),
            ]
            last: Optional[float] = None
            for cand in mark_candidates:
                candidate_value = _safe_float(cand)
                if candidate_value is not None:
                    last = candidate_value
                    break
            if last is None:
                if entry_f is not None:
                    last = entry_f
                elif sym == contract_symbol and last_fallback is not None:
                    last = last_fallback
                else:
                    last = 0.0

            unreal = raw.get("unrealizedPnl") or (info or {}).get("unrealisedPnl")
            pct_raw = raw.get("percentage") or (info or {}).get("unrealisedPnlPcnt")
            tp = raw.get("takeProfit") or (info or {}).get("takeProfit")
            sl = raw.get("stopLoss") or (info or {}).get("stopLoss")
            lev = raw.get("leverage") or (info or {}).get("leverage")

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

            symbol_fmt = str(sym)
            side_fmt = side if side is not None else "n/a"
            size_fmt = _format_decimal(size_f) if size_f is not None else "n/a"
            entry_fmt = _format_decimal(entry_f) if entry_f is not None else "n/a"
            last_fmt = _format_decimal(last)
            tp_fmt = _format_decimal(tp) if tp is not None else "n/a"
            sl_fmt = _format_decimal(sl) if sl is not None else "n/a"
            lev_fmt = _format_decimal(lev) if lev is not None else "n/a"
            unreal_fmt = _format_decimal(unreal) if unreal is not None else "n/a"
            pct_fmt = (
                _format_decimal(pct_leverage) if pct_leverage is not None else "n/a"
            )
            pct_raw_fmt = (
                _format_decimal(pct_baseline) if pct_baseline is not None else "n/a"
            )

            lines.append(
                f"symbol={symbol_fmt}, side={side_fmt}, size={size_fmt}, entry={entry_fmt}, last={last_fmt}, "
                f"tp={tp_fmt}, sl={sl_fmt}, lev={lev_fmt}, unreal_PNL={unreal_fmt} "
                f"(ROI_leverage={pct_fmt}%, ROI_baseline={pct_raw_fmt}%)"
            )
        except Exception:
            continue

    return lines, primary_side


def _gather_prompt_context(deps: AutomationDependencies) -> PromptContext:
    """Collect market snapshots, journal excerpts, and position context."""

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

    raw_positions = deps.bybit.get_positions()
    current_positions: List[Dict[str, Any]] = []
    if isinstance(raw_positions, list):
        current_positions = [pos for pos in raw_positions if isinstance(pos, dict)]
    elif isinstance(raw_positions, dict):
        for value in raw_positions.values():
            if isinstance(value, dict):
                current_positions.append(value)
            elif isinstance(value, list):
                current_positions.extend(
                    item for item in value if isinstance(item, dict)
                )
    elif raw_positions:
        try:
            current_positions = [
                item for item in list(raw_positions) if isinstance(item, dict)
            ]
        except Exception:
            current_positions = []

    position_lines, pos_side = _summarize_positions(
        current_positions, deps.contract_symbol, current_price
    )

    try:
        journals_today_df = deps.store.fetch_journals(
            symbol=deps.contract_symbol, today_only=True, limit=50, ascending=True
        )
        journal_today_text = _format_journal_dataframe(
            journals_today_df,
            timestamp_format="%H:%M:%S",
        )
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
        recent_reports_text = _format_journal_dataframe(
            recent_df,
            timestamp_format="%m-%d %H:%M",
        )
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
                    since_open_text = _format_journal_dataframe(
                        journals_since_df,
                        timestamp_format="%H:%M:%S",
                    )
    except Exception:
        since_open_text = ""

    return PromptContext(
        current_price=current_price,
        csv_4h=csv_4h,
        csv_1h=csv_1h,
        csv_15m=csv_15m,
        current_positions=current_positions,
        pos_side=pos_side,
        current_positions_lines=position_lines,
        journal_today_text=journal_today_text,
        recent_reports_text=recent_reports_text,
        reviews_text=reviews_text,
        since_open_text=since_open_text,
        chart_images=chart_images,
    )


def _build_prompt(deps: AutomationDependencies, ctx: PromptContext) -> str:
    """Compose the LLM prompt while preserving the established block order."""

    now_utc = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    prompt = (
        f"당신은 세계 최고의 암호화폐 트레이더입니다. 한국어로 답변하세요.\n"
        f"이미 진입한 포지션의 레버리지는 조절할 수 없습니다.\n"
        f"당신은 최소 5배에서 최대 75배까지 레버리지를 사용할 수 있습니다.\n"
        """[Explain 출력 양식]
- 차트 분석
- 1차,2차,3차 저항선 분석
- 1차,2차,3차 지지선 분석
- 차트의 패턴은 어떤가?
- 고점은 낮아지고 있는가
- 고점은 높아지고 있는가
- 저점은 낮아지고 있는가
- 저점은 높아지고 있는가
- 거래량은 증가하고 있는가
- 거래량은 감소하고 있는가
- 차트의 패턴은 어떤가
- 매수를 할때의 권장 시나리오
    - 어디까지 올라갈 것인가?
    - 어떤 경우 임의로 익절을 해야하는가?
    - 어떤 경우 임의로 손절을 해야하는가?
    - 어디까지가 마지노선(SL)인가?
    - 어디까지가 베스트 익절(TP)인가?
- 매도를 할때의 권장 시나리오
    - 어디까지 내려갈 것인가?
    - 어떤 경우 임의로 익절을 해야하는가?
    - 어떤 경우 임의로 손절을 해야하는가?
    - 어디까지가 마지노선(SL)인가?
    - 어디까지가 베스트 익절(TP)인가?
- 최종 결론
    - 롱, 숏, 보유 중 선택
    - 차트의 패턴은 어떤가?
    - 차트가 어떤 모양으로 흘러갈 것인가?
    - 어느 저항선, 지시선을 건들거나 돌파할 것인가?
    - 어떤 경우 임의로 손절 할건가?
    - 어떤 경우 임의로 익절 할건가?
[/Explain 출력 양식]\n"""
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
        "price, stop loss (sl), take profit (tp), buy_now (boolean), leverage (number), update_existing (boolean).\n"
        "If you want to immediately take profit or cut loss on an existing position, set close_now=true "
        "and optionally close_percent (1~100).\n"
        "When close_now is true, we will reduceOnly market-close the current position(s) without opening a new one in this cycle.\n"
        f"Per-symbol max allocation is {deps.per_symbol_alloc_pct:.2f}% of account equity (for your reference).\n"
        "Include your reasoning for the decision in the 'explain' field. Output JSON, Korean only."
    )
    return prompt


def _normalize_decision_payload(payload: Any) -> Dict[str, Any]:
    """
    Flatten tool-call formatted AI responses into a uniform decision dict.
    """

    if isinstance(payload, dict):
        if payload.get("_type") == "tool_call":
            arguments = payload.get("arguments")
            if isinstance(arguments, dict):
                merged = dict(arguments)
                tool_name = payload.get("tool")
                if tool_name and "_tool_name" not in merged:
                    merged["_tool_name"] = tool_name
                return merged
        return payload
    return {}


def _normalize_status_value(value: Any) -> str:
    """Normalize status text to supported keywords."""

    try:
        text = str(value).strip().lower()
    except Exception:
        return ""

    if not text:
        return ""

    if text in {"watch", "wait", "observe", "관망"}:
        return "hold"

    if text in {"long", "short", "hold", "stop"}:
        return text

    return ""


def _decision_is_actionable(decision: Dict[str, Any]) -> bool:
    """Determine whether the AI decision includes enough info to act."""

    if not isinstance(decision, dict):
        return False

    if bool(decision.get("close_now")):
        return True

    status = _normalize_status_value(decision.get("Status") or decision.get("status"))
    return bool(status)


def _request_trade_decision(
    deps: AutomationDependencies,
    prompt: str,
    ctx: PromptContext,
) -> Dict[str, Any]:
    """Obtain a structured decision from the AI provider with logging fallbacks."""

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
        max_attempts = max(1, int(os.getenv("AI_DECISION_MAX_RETRIES", "3")))
    except Exception:
        max_attempts = 3

    try:
        retry_delay = float(os.getenv("AI_DECISION_RETRY_DELAY_SECONDS", "2"))
    except Exception:
        retry_delay = 2.0
    retry_delay = max(0.0, retry_delay)

    last_decision: Dict[str, Any] = {}

    for attempt in range(1, max_attempts + 1):
        try:
            parsed = deps.ai_provider.decide_json(prompt, images=images_payload)
            normalized = _normalize_decision_payload(parsed)
            LOGGER.info(
                json.dumps(
                    {
                        "event": "llm_response_parsed",
                        "provider": os.getenv("AI_PROVIDER", "gemini").lower(),
                        "attempt": attempt,
                        "parsed": parsed,
                        "normalized": normalized,
                    },
                    ensure_ascii=False,
                )
            )
        except Exception:
            response = deps.ai_provider.decide(prompt, images=images_payload)
            try:
                LOGGER.info(
                    json.dumps(
                        {
                            "event": "llm_response_raw",
                            "provider": os.getenv("AI_PROVIDER", "gemini").lower(),
                            "attempt": attempt,
                            "response": response,
                        },
                        ensure_ascii=False,
                    )
                )
            except Exception as exc:
                LOGGER.error("LLM raw logging failed: %s", exc)

            parser = deps.parser
            value = parser.make_it_object(response)
            normalized = _normalize_decision_payload(value)
            try:
                LOGGER.info(
                    json.dumps(
                        {
                            "event": "llm_response_parsed",
                            "provider": os.getenv("AI_PROVIDER", "gemini").lower(),
                            "attempt": attempt,
                            "parsed": value,
                            "normalized": normalized,
                        },
                        ensure_ascii=False,
                    )
                )
            except Exception as exc:
                LOGGER.error("LLM parsed logging failed: %s", exc)

        last_decision = normalized if isinstance(normalized, dict) else {}

        if _decision_is_actionable(last_decision):
            return last_decision

        status_display = last_decision.get("Status") or last_decision.get("status")
        LOGGER.warning(
            "AI decision attempt %s/%s missing actionable status: %s",
            attempt,
            max_attempts,
            status_display,
        )
        if attempt < max_attempts and retry_delay > 0:
            time.sleep(retry_delay)

    return last_decision


def _normalize_bool(val: Any) -> bool:
    """Interpret truthy indicators coming from heterogeneous AI responses."""

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
    """Map provider-specific side representations into canonical long/short."""

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
    """Derive maximum tolerated loss percentage based on leverage settings."""

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
    """Normalize order type values coming from loosely structured payloads."""

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
    """Execute immediate close instructions and persist the resulting journal."""

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
    """Update TP/SL for existing positions when the AI requests modifications."""

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

    positions = [
        pos
        for pos in (ctx.current_positions or [])
        if _extract_position_symbol(pos) == deps.contract_symbol
    ]
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

        current_tp: Optional[float] = None
        for candidate in (
            pos.get("takeProfit"),
            info.get("takeProfit") if isinstance(info, dict) else None,
            info.get("tp") if isinstance(info, dict) else None,
            info.get("take_profit") if isinstance(info, dict) else None,
        ):
            current_tp = _safe_float(candidate)
            if current_tp is not None:
                break

        current_sl: Optional[float] = None
        for candidate in (
            pos.get("stopLoss"),
            info.get("stopLoss") if isinstance(info, dict) else None,
            info.get("sl") if isinstance(info, dict) else None,
            info.get("stop_loss") if isinstance(info, dict) else None,
        ):
            current_sl = _safe_float(candidate)
            if current_sl is not None:
                break

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

        tp_needs_update = bool(
            tp_val is not None and not _values_almost_equal(current_tp, tp_val)
        )
        sl_needs_update = bool(
            applied_sl is not None and not _values_almost_equal(current_sl, applied_sl)
        )
        needs_update = tp_needs_update or sl_needs_update

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

        result_no_change = _result_indicates_no_change(result) or not needs_update

        updates.append(
            {
                "side": pos_side,
                "position_idx": _extract_position_idx(pos),
                "entry_price": entry_price,
                "leverage": leverage_val,
                "requested_tp": tp_val,
                "requested_sl": sl_val,
                "applied_sl": applied_sl,
                "current_tp": current_tp,
                "current_sl": current_sl,
                "tp_needs_update": tp_needs_update,
                "sl_needs_update": sl_needs_update,
                "needs_update": needs_update,
                "result_no_change": result_no_change,
                "result": meta_result,
            }
        )

    if not success:
        change_requested = any(update.get("needs_update") for update in updates)
        no_change_only = not change_requested or all(
            update.get("result_no_change") for update in updates
        )
        if no_change_only:
            LOGGER.error(
                "update_existing 요청이 있었지만 TP/SL 값이 기존과 동일하여 변경을 적용하지 않았습니다."
            )
            _record_skip(
                deps,
                reason="update_failed_no_change",
                decision=decision,
                meta={"decision": decision, "updates": updates},
                reason_text="제공된 TP/SL 값이 기존 값과 동일합니다.",
            )
        else:
            LOGGER.error("update_existing 요청으로 TP/SL을 수정하지 못했습니다.")
            _record_skip(
                deps,
                reason="update_failed_other",
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
    """Run confirm prompt to sanity-check order parameters before execution."""

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
            "필수: confirm(boolean), explain(string). 선택: tp, sl, price, buy_now, leverage.\n"
            "confirm=false이면 반드시 explain에 거부 사유를 한국어로 작성하세요. confirm=true라면 결정 근거를 간단히 남기세요.\n"
            "확신하면 confirm=true. 수정이 필요하면 값을 조정해 응답하세요."
        )

        confirm: Optional[Dict[str, Any]] = None
        confirm_error: Optional[Exception] = None
        max_attempts = 3
        retry_delay_seconds = 5

        for attempt in range(1, max_attempts + 1):
            try:
                candidate = deps.ai_provider.confirm_trade_json(confirm_prompt)
                if not isinstance(candidate, dict):
                    raise ValueError("confirm 응답이 dict 형식이 아닙니다.")
                internal_error = candidate.get("_internal.tool_calls_error")
                if internal_error:
                    raise RuntimeError(str(internal_error))
                if "confirm" not in candidate:
                    raise ValueError("confirm 키가 포함되지 않은 응답입니다.")
                confirm = candidate
                LOGGER.info(
                    json.dumps(
                        {
                            "event": "llm_confirm_response_parsed",
                            "provider": os.getenv("AI_PROVIDER", "gemini").lower(),
                            "parsed": confirm,
                            "attempt": attempt,
                        },
                        ensure_ascii=False,
                    )
                )
                break
            except Exception as exc:
                confirm_error = exc
                LOGGER.warning(
                    "AI confirm attempt %s/%s failed: %s",
                    attempt,
                    max_attempts,
                    exc,
                )
                if attempt < max_attempts:
                    time.sleep(retry_delay_seconds)

        if confirm is None:
            LOGGER.error(
                "AI confirm failed after %s attempts: %s",
                max_attempts,
                confirm_error,
            )
            _record_skip(
                deps,
                reason="confirm_failed",
                decision=decision,
                meta={
                    "first": decision,
                    "confirm_error": str(confirm_error) if confirm_error else None,
                },
                reason_text="confirm 단계 호출에 실패했습니다.",
            )
            return (
                order_type,
                float(entry_price),
                use_tp,
                use_sl,
                leverage,
                None,
                True,
            )

        confirm_meta = confirm
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


def _execute_trade(
    deps: AutomationDependencies,
    ctx: PromptContext,
    decision: Dict[str, Any],
) -> None:
    """Translate a validated AI decision into exchange orders and journal logs."""

    raw_status = decision.get("Status") or decision.get("status")
    ai_status = _normalize_status_value(raw_status)
    if not ai_status and raw_status is not None:
        try:
            ai_status = str(raw_status).strip().lower()
        except Exception:
            ai_status = ""

    if ai_status:
        decision.setdefault("status", ai_status)
        if "Status" not in decision:
            decision["Status"] = ai_status
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
        positions_same_symbol = [
            pos
            for pos in (ctx.current_positions or [])
            if _extract_position_symbol(pos) == deps.contract_symbol
        ]
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

    if position_params.type == "limit":
        try:
            position_params.price = float(
                deps.bybit.exchange.price_to_precision(
                    deps.contract_symbol, position_params.price
                )
            )
        except Exception:
            position_params.price = float(position_params.price)

    for attr in ("tp", "sl"):
        value = getattr(position_params, attr)
        if value is None:
            continue
        try:
            quantized = float(
                deps.bybit.exchange.price_to_precision(deps.contract_symbol, value)
            )
            setattr(position_params, attr, quantized)
        except Exception:
            setattr(position_params, attr, float(value))

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
    """Persist journaling or emergency close instructions when no trade opens."""

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
    reason_text: Optional[str] = None,
) -> None:
    """Log skip decisions to the journal for downstream auditing."""

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
    """Persist failures encountered while loading journals to aid debugging."""

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
    """Execute the full automation pipeline for a single symbol with retries."""

    delays = [5, 10, 60]

    for attempt in range(len(delays) + 1):
        deps: Optional[AutomationDependencies] = None
        try:
            deps = _init_dependencies(symbol_usdt, symbols)
            ctx = _gather_prompt_context(deps)
            prompt = _build_prompt(deps, ctx)
            decision = _request_trade_decision(deps, prompt, ctx)
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
    """Trigger the AI-powered loss review job for recently closed positions."""

    del symbols
    store = TradeStore(
        StorageConfig(
            mysql_url=os.getenv("MYSQL_URL"),
            sqlite_path=os.getenv("SQLITE_PATH"),
        )
    )
    journal_service = JournalService(store, AIProvider())
    journal_service.review_losing_trades(since_minutes=since_minutes)
