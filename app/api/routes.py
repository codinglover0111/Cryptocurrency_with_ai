from __future__ import annotations

import json
import math
import os
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.core.symbols import parse_trading_symbols, to_ccxt_symbols
from utils.bybit_utils import BybitUtils
from utils.storage import StorageConfig, TradeStore

router = APIRouter()


CLOSED_BY_TARGET_PROFIT = "목표수익달성"
CLOSED_BY_STOP_LOSS = "SL 실행으로 인한 손절"
CLOSED_BY_UNKNOWN = "unknown"
PNL_EPSILON = 1e-8
TP_TOLERANCE_RATIO = 0.0002
SL_TOLERANCE_RATIO = 0.0002

STATUS_CACHE_TTL = 5.0
STATUS_CACHE: Dict[str, Any] = {"ts": 0.0, "data": None}
STATUS_CACHE_LOCK = threading.Lock()


class LeverageBody(BaseModel):
    symbol: str
    leverage: float
    margin_mode: Optional[str] = "cross"


class JournalBody(BaseModel):
    symbol: Optional[str] = None
    entry_type: str
    content: str
    reason: Optional[str] = None
    meta: Optional[dict] = None
    ref_order_id: Optional[str] = None
    ts: Optional[datetime] = None


def _redact_sensitive(obj: Any):
    sensitive_keys = (
        "key",
        "secret",
        "token",
        "auth",
        "password",
        "passwd",
        "mysql",
        "dsn",
        "bearer",
        "authorization",
        "openai",
        "gemini",
        "bybit",
    )
    try:
        if isinstance(obj, dict):
            redacted: Dict[str, Any] = {}
            for k, v in obj.items():
                if any(s in str(k).lower() for s in sensitive_keys):
                    redacted[k] = "[REDACTED]"
                else:
                    redacted[k] = _redact_sensitive(v)
            return redacted
        if isinstance(obj, list):
            return [_redact_sensitive(v) for v in obj]
        return obj
    except Exception:
        return obj


def _to_utc_iso(value: Any) -> str:
    try:
        if isinstance(value, datetime):
            if value.tzinfo is None:
                value = value.replace(tzinfo=timezone.utc)
            else:
                value = value.astimezone(timezone.utc)
            return value.isoformat().replace("+00:00", "Z")
    except Exception:
        pass
    return str(value)


def _normalize_decision_status(value: Any) -> Optional[str]:
    if value is None:
        return None
    try:
        raw = str(value).strip().lower()
    except Exception:
        return None
    if not raw:
        return None
    if raw in {"long", "buy"}:
        return "long"
    if raw in {"short", "sell"}:
        return "short"
    if raw in {"hold", "watch"}:
        return "hold"
    if raw in {"stop"}:
        return "stop"
    if raw in {"skip"}:
        return "skip"
    return raw


def _first_number(values: List[Any]) -> Optional[float]:
    for val in values:
        if val is None:
            continue
        try:
            num = float(val)
        except (TypeError, ValueError):
            continue
        if math.isfinite(num):
            return float(num)
    return None


def _summarize_positions(
    bybit: BybitUtils,
    positions: List[dict],
    symbol: Optional[str] = None,
    use_mark_price: bool = False,
    force_exchange_pnl: bool = False,
    force_roe: bool = False,
):
    summary: List[Dict[str, Any]] = []
    for pos in positions or []:
        try:
            sym = pos.get("symbol") or pos.get("info", {}).get("symbol")
            if symbol and sym != symbol:
                continue
            side = pos.get("side") or pos.get("info", {}).get("side")
            entry = pos.get("entryPrice") or pos.get("info", {}).get("entryPrice")
            size = pos.get("contracts") or pos.get("size") or pos.get("info", {}).get("size")
            if size in (None, 0):
                continue
            size_val = float(size)
            entry_val = float(entry) if entry is not None else None
            mark = pos.get("markPrice") if use_mark_price else None
            if mark is None:
                mark = bybit.get_last_price(sym)
            mark_val = float(mark) if mark is not None else None
            unrealized = pos.get("unrealizedPnl")
            if unrealized is None or force_exchange_pnl:
                if entry_val is not None and mark_val is not None:
                    if (side or "").lower() in ("long", "buy"):
                        unrealized = (mark_val - entry_val) * size_val
                    else:
                        unrealized = (entry_val - mark_val) * size_val
            roe = pos.get("percentage")
            if roe is None or force_roe:
                if unrealized is not None and entry_val:
                    roe = (float(unrealized) / (entry_val * size_val)) * 100.0
            summary.append(
                {
                    "symbol": sym,
                    "side": side,
                    "entry": entry_val,
                    "size": size_val,
                    "mark_price": mark_val,
                    "pnl": float(unrealized) if unrealized is not None else None,
                    "roe": float(roe) if roe is not None else None,
                }
            )
        except Exception:
            continue
    return summary


def _serialize_journal_row(row: Dict[str, Any]) -> Tuple[Dict[str, Any], str, Optional[str]]:
    ts_iso = _to_utc_iso(row.get("ts"))
    meta_val = row.get("meta")
    if isinstance(meta_val, str):
        try:
            meta_val = json.loads(meta_val)
        except Exception:
            meta_val = None
    content_val = row.get("content")
    if isinstance(content_val, str):
        try:
            content_val = json.loads(content_val)
        except Exception:
            content_val = None
    decision_meta = meta_val.get("decision") if isinstance(meta_val, dict) else None
    decision_status = None
    for cand in [
        row.get("decision_status"),
        meta_val.get("status") if isinstance(meta_val, dict) else None,
        meta_val.get("side") if isinstance(meta_val, dict) else None,
        decision_meta.get("status") if isinstance(decision_meta, dict) else None,
        decision_meta.get("Status") if isinstance(decision_meta, dict) else None,
        decision_meta.get("ai_status") if isinstance(decision_meta, dict) else None,
        decision_meta.get("side") if isinstance(decision_meta, dict) else None,
        content_val.get("status") if isinstance(content_val, dict) else None,
        content_val.get("side") if isinstance(content_val, dict) else None,
    ]:
        norm = _normalize_decision_status(cand)
        if norm:
            decision_status = norm
            break

    decision_entry = _first_number(
        [
            row.get("decision_entry"),
            meta_val.get("entry_price") if isinstance(meta_val, dict) else None,
            meta_val.get("entry") if isinstance(meta_val, dict) else None,
            decision_meta.get("entry") if isinstance(decision_meta, dict) else None,
            decision_meta.get("entry_price") if isinstance(decision_meta, dict) else None,
            decision_meta.get("price") if isinstance(decision_meta, dict) else None,
            content_val.get("entry") if isinstance(content_val, dict) else None,
            content_val.get("entry_price") if isinstance(content_val, dict) else None,
            content_val.get("price") if isinstance(content_val, dict) else None,
        ]
    )
    decision_tp = _first_number(
        [
            row.get("decision_tp"),
            meta_val.get("tp") if isinstance(meta_val, dict) else None,
            decision_meta.get("tp") if isinstance(decision_meta, dict) else None,
            content_val.get("tp") if isinstance(content_val, dict) else None,
        ]
    )
    decision_sl = _first_number(
        [
            row.get("decision_sl"),
            meta_val.get("sl") if isinstance(meta_val, dict) else None,
            decision_meta.get("sl") if isinstance(decision_meta, dict) else None,
            content_val.get("sl") if isinstance(content_val, dict) else None,
        ]
    )

    safe_meta = _redact_sensitive(meta_val)

    payload = {
        "ts": ts_iso,
        "symbol": row.get("symbol"),
        "entry_type": row.get("entry_type"),
        "content": row.get("content"),
        "reason": row.get("reason"),
        "meta": safe_meta,
        "ref_order_id": row.get("ref_order_id"),
        "decision_status": decision_status,
        "decision_entry": decision_entry,
        "decision_tp": decision_tp,
        "decision_sl": decision_sl,
    }
    entry_type_lower = str(row.get("entry_type") or "").strip().lower()
    return payload, entry_type_lower, decision_status


def _reconcile_auto_closed_positions(store: TradeStore) -> None:
    df = store.fetch_positions(closed_only=True)
    if df.empty:
        return
    rows_to_fix = []
    for _, row in df.iterrows():
        try:
            close_type = row.get("close_type") or row.get("closed_by")
            if close_type not in {CLOSED_BY_TARGET_PROFIT, CLOSED_BY_STOP_LOSS}:
                continue
            pnl = float(row.get("pnl") or 0.0)
            if abs(pnl) < PNL_EPSILON:
                continue
            if row.get("stats_recorded"):
                continue
            rows_to_fix.append(row)
        except Exception:
            continue
    for row in rows_to_fix:
        try:
            store.record_trade_from_row(row)
        except Exception:
            continue


def _build_store() -> TradeStore:
    return TradeStore(
        StorageConfig(
            mysql_url=os.getenv("MYSQL_URL"),
            sqlite_path=os.getenv("SQLITE_PATH"),
        )
    )


@router.get("/status")
def status():
    now = time.monotonic()
    with STATUS_CACHE_LOCK:
        cached_payload = STATUS_CACHE.get("data")
        cached_ts = STATUS_CACHE.get("ts", 0.0)
    if cached_payload is not None and now - cached_ts < STATUS_CACHE_TTL:
        return cached_payload

    bybit = BybitUtils(is_testnet=bool(int(os.getenv("TESTNET", "1"))))
    try:
        data = bybit.get_account_overview()
        if isinstance(data, dict):
            bal = data.get("balance")
            if isinstance(bal, dict) and "raw" in bal:
                bal.pop("raw", None)
            positions = data.get("positions") if isinstance(data.get("positions"), list) else []
            try:
                data["positionsSummary"] = _summarize_positions(
                    bybit,
                    positions,
                    symbol=None,
                    use_mark_price=False,
                    force_exchange_pnl=False,
                    force_roe=False,
                )
            except Exception:
                data["positionsSummary"] = []
    except Exception as exc:
        if cached_payload is not None:
            return cached_payload
        raise HTTPException(status_code=502, detail=str(exc))

    with STATUS_CACHE_LOCK:
        STATUS_CACHE["data"] = data
        STATUS_CACHE["ts"] = time.monotonic()
    return data


@router.post("/leverage")
def set_leverage(body: LeverageBody):
    bybit = BybitUtils(is_testnet=bool(int(os.getenv("TESTNET", "1"))))
    res = bybit.set_leverage(body.symbol, body.leverage, body.margin_mode or "cross")
    return {"ok": True, "result": res}


@router.get("/stats")
def stats():
    store = _build_store()
    try:
        _reconcile_auto_closed_positions(store)
    except Exception:
        pass
    summary = store.compute_stats()
    try:
        summary["realized_pnl"] = float(summary.get("realized_pnl") or 0.0)
        summary["avg_pnl"] = float(summary.get("avg_pnl") or 0.0)
    except Exception:
        pass
    return summary


@router.get("/stats/range")
def stats_range(
    since: Optional[str] = None,
    until: Optional[str] = None,
    symbol: Optional[str] = None,
    group: Optional[str] = None,
):
    store = _build_store()

    def _parse_iso(ts: Optional[str]):
        if not ts:
            return None
        try:
            if len(ts) == 10:
                return datetime.fromisoformat(ts + "T00:00:00+00:00")
            return datetime.fromisoformat(ts)
        except Exception:
            return None

    since_ts = _parse_iso(since)
    until_ts = _parse_iso(until)
    g = group if group in ("day", "week", "month") else None
    result = store.compute_stats_range(
        since_ts=since_ts, until_ts=until_ts, symbol=symbol, group=g
    )
    try:
        if isinstance(result, dict) and "summary" in result:
            s = result["summary"]
            s["realized_pnl"] = float(s.get("realized_pnl") or 0.0)
            s["avg_pnl"] = float(s.get("avg_pnl") or 0.0)
    except Exception:
        pass
    return result


@router.post("/journals")
def create_journal(body: JournalBody):
    store = _build_store()
    payload = {k: v for k, v in body.model_dump().items() if v is not None}
    store.record_journal(payload)
    return {"ok": True}


@router.get("/journals")
def list_journals(
    symbol: Optional[str] = None,
    types: Optional[str] = None,
    today_only: int = 1,
    limit: int = 20,
    ascending: int = 0,
):
    store = _build_store()
    raw_types: List[str] = [t.strip() for t in types.split(",") if t.strip()] if types else []
    req_types = tuple(raw_types) if raw_types else None
    df = store.fetch_journals(
        symbol=symbol,
        types=req_types,
        today_only=bool(today_only),
        limit=min(max(int(limit), 1), 200),
        ascending=bool(ascending),
    )
    items = []
    for _, row in df.iterrows():
        payload, _, _ = _serialize_journal_row(row)
        items.append(payload)
    return {"items": items}


@router.get("/journals/filtered")
def list_journals_filtered(
    symbol: Optional[str] = None,
    types: Optional[str] = None,
    today_only: int = 1,
    limit: int = 20,
    ascending: int = 0,
    decision_statuses: Optional[str] = None,
    recent_minutes: Optional[int] = None,
    page: Optional[int] = None,
):
    store = _build_store()
    allowed_types = {"thought", "decision", "action", "review", "error"}
    raw_types: List[str] = [t.strip() for t in types.split(",") if t.strip()] if types else []
    req_types = tuple(t for t in raw_types if t in allowed_types)

    minutes_window: Optional[int] = None
    if recent_minutes is not None:
        try:
            minutes_window = max(1, min(int(recent_minutes), 24 * 60))
        except Exception:
            minutes_window = None

    since_ts: Optional[datetime] = None
    if minutes_window:
        since_ts = datetime.utcnow() - timedelta(minutes=minutes_window)

    effective_today_only = bool(today_only) if since_ts is None else False

    allowed_page_sizes: Tuple[int, ...] = (10, 30, 50, 100)
    try:
        requested_limit = int(limit)
    except Exception:
        requested_limit = 0

    page_number: Optional[int] = None
    page_size = max(1, min(requested_limit if requested_limit > 0 else 20, 200))
    limit_choices: Optional[Tuple[int, ...]] = None
    if page is not None:
        try:
            page_number = max(1, int(page))
        except Exception:
            page_number = 1
        page_size = allowed_page_sizes[0]
        for option in allowed_page_sizes:
            if requested_limit == option:
                page_size = option
                break
        limit_choices = allowed_page_sizes

    status_filters: Optional[set[str]] = None
    if decision_statuses is not None:
        raw_statuses = str(decision_statuses).strip()
        if not raw_statuses or raw_statuses == "__none__":
            status_filters = set()
        else:
            normalized: set[str] = set()
            for token in raw_statuses.split(","):
                token = token.strip()
                if not token:
                    continue
                norm = _normalize_decision_status(token)
                if norm:
                    normalized.add(norm)
            status_filters = normalized if normalized else set()

    if status_filters is not None and not status_filters:
        empty_payload: Dict[str, Any] = {"items": []}
        if page_number is not None:
            empty_payload.update({"page": page_number, "page_size": page_size, "total": 0})
        return empty_payload

    def _serialize_row(row):
        payload, entry_type_lower, decision_status = _serialize_journal_row(row)
        return payload, entry_type_lower, decision_status

    if page_number is not None:
        offset_value = (page_number - 1) * page_size
        df, total_count = store.fetch_journals(
            symbol=symbol,
            types=list(req_types) if req_types else None,
            today_only=effective_today_only,
            since_ts=since_ts,
            limit=page_size,
            ascending=bool(ascending),
            offset=offset_value,
            return_total=True,
            limit_choices=limit_choices,
        )
        total_count_int = int(total_count)
        total_pages = (
            ((total_count_int - 1) // page_size + 1)
            if page_size > 0 and total_count_int > 0
            else 0
        )
        if total_pages and page_number > total_pages:
            page_number = total_pages
            offset_value = (page_number - 1) * page_size
            df, total_count = store.fetch_journals(
                symbol=symbol,
                types=list(req_types) if req_types else None,
                today_only=effective_today_only,
                since_ts=since_ts,
                limit=page_size,
                ascending=bool(ascending),
                offset=offset_value,
                return_total=True,
                limit_choices=limit_choices,
            )
            total_count_int = int(total_count)
        if df.empty:
            return {
                "items": [],
                "page": page_number,
                "page_size": page_size,
                "total": total_count_int,
            }
        items = []
        for _, row in df.iterrows():
            payload, entry_type_lower, decision_status = _serialize_row(row)
            if status_filters and decision_status not in status_filters:
                continue
            if limit_choices and entry_type_lower not in {"decision", "action", "review", "thought", "error"}:
                continue
            items.append(payload)
        return {
            "items": items,
            "page": page_number,
            "page_size": page_size,
            "total": total_count_int,
        }

    df = store.fetch_journals(
        symbol=symbol,
        types=list(req_types) if req_types else None,
        today_only=effective_today_only,
        since_ts=since_ts,
        limit=page_size,
        ascending=bool(ascending),
    )
    items = []
    for _, row in df.iterrows():
        payload, _, decision_status = _serialize_row(row)
        if status_filters and decision_status not in status_filters:
            continue
        items.append(payload)
    return {"items": items}


@router.get("/symbols")
def symbols():
    codes = parse_trading_symbols()
    items = []
    for c in codes:
        spot, contract = to_ccxt_symbols(c)
        items.append({"code": c, "spot": spot, "contract": contract})
    return {"symbols": items}
