# pylint: disable=broad-except
# ruff: noqa: E722, BLE001
from __future__ import annotations

import os
import json
import math
import time
import threading
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Tuple, Any, Dict, cast

from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.core.symbols import parse_trading_symbols, to_ccxt_symbols
from utils.bybit_utils import BybitUtils
from utils.storage import TradeStore, StorageConfig
from utils.ai_provider import AIProvider


app = FastAPI(title="Crypto Bot UI")


CLOSED_BY_TARGET_PROFIT = "목표수익달성"
CLOSED_BY_STOP_LOSS = "SL 실행으로 인한 손절"
CLOSED_BY_UNKNOWN = "unknown"
PNL_EPSILON = 1e-8
# 0.02%
TP_TOLERANCE_RATIO = 0.0002
# 0.02%
SL_TOLERANCE_RATIO = 0.0002

STATUS_CACHE_TTL = 5.0
STATUS_CACHE = {"ts": 0.0, "data": None}
STATUS_CACHE_LOCK = threading.Lock()


def _redact_sensitive(obj):
    """Recursively redact sensitive-looking keys in dict/list structures."""
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
            redacted = {}
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
    """Convert datetime-like values to UTC ISO8601 string with trailing Z."""
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


app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
def index(request: Request, tz: Optional[str] = None):
    return templates.TemplateResponse("index.html", {"request": request, "tz": tz})


class LeverageBody(BaseModel):
    symbol: str
    leverage: float
    margin_mode: Optional[str] = "cross"


@app.on_event("startup")
def _startup():
    pass


@app.get("/health")
def health():
    return {
        "status": "ok",
        "ts": datetime.utcnow()
        .replace(tzinfo=timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
    }


@app.get("/status")
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
        try:
            if isinstance(data, dict):
                bal = data.get("balance")
                if isinstance(bal, dict) and "raw" in bal:
                    bal.pop("raw", None)
                positions = (
                    data.get("positions")
                    if isinstance(data.get("positions"), list)
                    else []
                )
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
        except Exception:
            pass
    except Exception:
        if cached_payload is not None:
            return cached_payload
        raise

    with STATUS_CACHE_LOCK:
        STATUS_CACHE["data"] = data
        STATUS_CACHE["ts"] = time.monotonic()
    return data


@app.post("/leverage")
def set_leverage(body: LeverageBody):
    bybit = BybitUtils(is_testnet=bool(int(os.getenv("TESTNET", "1"))))
    res = bybit.set_leverage(body.symbol, body.leverage, body.margin_mode or "cross")
    return {"ok": True, "result": res}


@app.get("/stats")
def stats():
    store = TradeStore(
        StorageConfig(
            mysql_url=os.getenv("MYSQL_URL"),
            sqlite_path=os.getenv("SQLITE_PATH"),
        )
    )
    # 자동(TP/SL) 청산으로 stats 누락된 건을 보완
    try:
        _reconcile_auto_closed_positions(store)
    except Exception:
        # 통계 조회는 실패 없이 반환되도록 방어
        pass
    s = store.compute_stats()
    # USD 표기 일관성을 위해 float로 보정
    try:
        s["realized_pnl"] = float(s.get("realized_pnl") or 0.0)
        s["avg_pnl"] = float(s.get("avg_pnl") or 0.0)
    except Exception:
        pass
    return s


@app.get("/stats_range")
def stats_range(
    since: Optional[str] = None,
    until: Optional[str] = None,
    symbol: Optional[str] = None,
    group: Optional[str] = None,
):
    store = TradeStore(
        StorageConfig(
            mysql_url=os.getenv("MYSQL_URL"),
            sqlite_path=os.getenv("SQLITE_PATH"),
        )
    )

    def _parse_iso(ts: Optional[str]):
        if not ts:
            return None
        try:
            # YYYY-MM-DD 또는 ISO8601 허용, UTC 가정
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
    # float 보정
    try:
        if isinstance(result, dict) and "summary" in result:
            s = result["summary"]
            s["realized_pnl"] = float(s.get("realized_pnl") or 0.0)
            s["avg_pnl"] = float(s.get("avg_pnl") or 0.0)
    except Exception:
        pass
    return result


@app.post("/close_all")
def close_all():
    bybit = BybitUtils(is_testnet=bool(int(os.getenv("TESTNET", "1"))))
    res = bybit.close_all_positions()
    return {"ok": True, "result": res}


@app.get("/symbols")
def symbols():
    codes = parse_trading_symbols()
    items = []
    for c in codes:
        spot, contract = to_ccxt_symbols(c)
        items.append({"code": c, "spot": spot, "contract": contract})
    return {"symbols": items}


class JournalBody(BaseModel):
    symbol: Optional[str] = None
    entry_type: str  # thought | decision | action | review
    content: str
    reason: Optional[str] = None
    meta: Optional[dict] = None
    ref_order_id: Optional[str] = None
    ts: Optional[datetime] = None


@app.post("/api/journals")
def create_journal(body: JournalBody):
    store = TradeStore(
        StorageConfig(
            mysql_url=os.getenv("MYSQL_URL"),
            sqlite_path=os.getenv("SQLITE_PATH"),
        )
    )
    store.record_journal({k: v for k, v in body.model_dump().items() if v is not None})
    return {"ok": True}


@app.get("/api/journals")
def list_journals(
    symbol: Optional[str] = None,
    types: Optional[str] = None,
    today_only: int = 1,
    limit: int = 20,
    ascending: int = 0,
):
    store = TradeStore(
        StorageConfig(
            mysql_url=os.getenv("MYSQL_URL"),
            sqlite_path=os.getenv("SQLITE_PATH"),
        )
    )
    type_list: Optional[List[str]] = (
        [t.strip() for t in types.split(",") if t.strip()] if types else None
    )
    df = store.fetch_journals(
        symbol=symbol,
        types=type_list,
        today_only=bool(today_only),
        limit=max(1, min(int(limit), 200)),
        ascending=bool(ascending),
    )
    if df.empty:
        return {"items": []}
    items = []
    for _, row in df.iterrows():
        ts_iso = _to_utc_iso(row.get("ts"))
        meta_val = row.get("meta")
        if isinstance(meta_val, str):
            try:
                meta_val = json.loads(meta_val)
            except Exception:
                pass
        content_val = row.get("content")
        content_obj = None
        if isinstance(content_val, str):
            try:
                content_obj = json.loads(content_val)
            except Exception:
                content_obj = None
        decision_meta = meta_val.get("decision") if isinstance(meta_val, dict) else None
        decision_status = None
        status_candidates = [
            row.get("decision_status"),
            meta_val.get("status") if isinstance(meta_val, dict) else None,
            meta_val.get("side") if isinstance(meta_val, dict) else None,
            decision_meta.get("status") if isinstance(decision_meta, dict) else None,
            decision_meta.get("Status") if isinstance(decision_meta, dict) else None,
            decision_meta.get("ai_status") if isinstance(decision_meta, dict) else None,
            decision_meta.get("side") if isinstance(decision_meta, dict) else None,
            content_obj.get("status") if isinstance(content_obj, dict) else None,
            content_obj.get("side") if isinstance(content_obj, dict) else None,
        ]
        for cand in status_candidates:
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
                decision_meta.get("entry_price")
                if isinstance(decision_meta, dict)
                else None,
                decision_meta.get("price") if isinstance(decision_meta, dict) else None,
                content_obj.get("entry") if isinstance(content_obj, dict) else None,
                content_obj.get("entry_price")
                if isinstance(content_obj, dict)
                else None,
                content_obj.get("price") if isinstance(content_obj, dict) else None,
            ]
        )
        decision_tp = _first_number(
            [
                row.get("decision_tp"),
                meta_val.get("tp") if isinstance(meta_val, dict) else None,
                decision_meta.get("tp") if isinstance(decision_meta, dict) else None,
                content_obj.get("tp") if isinstance(content_obj, dict) else None,
            ]
        )
        decision_sl = _first_number(
            [
                row.get("decision_sl"),
                meta_val.get("sl") if isinstance(meta_val, dict) else None,
                decision_meta.get("sl") if isinstance(decision_meta, dict) else None,
                content_obj.get("sl") if isinstance(content_obj, dict) else None,
            ]
        )

        safe_meta = _redact_sensitive(meta_val)
        items.append(
            {
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
        )
    return {"items": items}


@app.get("/api/journals_filtered")
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
    """안전 공개용 저널 API.

    - 허용 타입만 SQL에 반영하여 안전하게 필터링
    - 페이지네이션 메타데이터(page, page_size, total) 제공
    - 결정 상태 필터는 서버 측에서 적용
    """
    store = TradeStore(
        StorageConfig(
            mysql_url=os.getenv("MYSQL_URL"),
            sqlite_path=os.getenv("SQLITE_PATH"),
        )
    )

    allowed_types = {"thought", "decision", "action", "review", "error"}
    raw_types: List[str] = (
        [t.strip() for t in types.split(",") if t.strip()] if types else []
    )
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
        empty_payload = {"items": []}
        if page_number is not None:
            empty_payload.update(
                {"page": page_number, "page_size": page_size, "total": 0}
            )
        return empty_payload

    def _serialize_row(row):
        ts_iso = _to_utc_iso(row.get("ts"))
        meta_val = row.get("meta")
        if isinstance(meta_val, str):
            try:
                meta_val = json.loads(meta_val)
            except Exception:
                pass
        content_val = row.get("content")
        content_obj = None
        if isinstance(content_val, str):
            try:
                content_obj = json.loads(content_val)
            except Exception:
                content_obj = None
        decision_meta = meta_val.get("decision") if isinstance(meta_val, dict) else None
        decision_status = None
        status_candidates = [
            row.get("decision_status"),
            meta_val.get("status") if isinstance(meta_val, dict) else None,
            meta_val.get("side") if isinstance(meta_val, dict) else None,
            decision_meta.get("status") if isinstance(decision_meta, dict) else None,
            decision_meta.get("Status") if isinstance(decision_meta, dict) else None,
            decision_meta.get("ai_status") if isinstance(decision_meta, dict) else None,
            decision_meta.get("side") if isinstance(decision_meta, dict) else None,
            content_obj.get("status") if isinstance(content_obj, dict) else None,
            content_obj.get("side") if isinstance(content_obj, dict) else None,
        ]
        for cand in status_candidates:
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
                decision_meta.get("entry_price")
                if isinstance(decision_meta, dict)
                else None,
                decision_meta.get("price") if isinstance(decision_meta, dict) else None,
                content_obj.get("entry") if isinstance(content_obj, dict) else None,
                content_obj.get("entry_price")
                if isinstance(content_obj, dict)
                else None,
                content_obj.get("price") if isinstance(content_obj, dict) else None,
            ]
        )
        decision_tp = _first_number(
            [
                row.get("decision_tp"),
                meta_val.get("tp") if isinstance(meta_val, dict) else None,
                decision_meta.get("tp") if isinstance(decision_meta, dict) else None,
                content_obj.get("tp") if isinstance(content_obj, dict) else None,
            ]
        )
        decision_sl = _first_number(
            [
                row.get("decision_sl"),
                meta_val.get("sl") if isinstance(meta_val, dict) else None,
                decision_meta.get("sl") if isinstance(decision_meta, dict) else None,
                content_obj.get("sl") if isinstance(content_obj, dict) else None,
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

    fetch_types = list(req_types) if req_types else None

    if page_number is not None:
        offset_value = (page_number - 1) * page_size
        if status_filters is None:
            df, total_count = store.fetch_journals(
                symbol=symbol,
                types=fetch_types,
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
                ((total_count_int - 1) // page_size) + 1
                if page_size > 0 and total_count_int > 0
                else 0
            )

            if total_pages and page_number > total_pages:
                page_number = total_pages
                offset_value = (page_number - 1) * page_size
                df, total_count = store.fetch_journals(
                    symbol=symbol,
                    types=fetch_types,
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
                if status_filters is not None:
                    if entry_type_lower != "decision" or not decision_status:
                        continue
                    if decision_status not in status_filters:
                        continue
                items.append(payload)

            return {
                "items": items,
                "page": page_number,
                "page_size": page_size,
                "total": total_count_int,
            }

        # status_filters exists -> chunked scan to respect pagination
        chunk_limit = 200
        chunk_offset = 0
        rows: List[dict] = []
        matched_count = 0
        target_start = (page_number - 1) * page_size
        filters = cast(set[str], status_filters)

        while True:
            chunk = store.fetch_journals(
                symbol=symbol,
                types=fetch_types,
                today_only=effective_today_only,
                since_ts=since_ts,
                limit=chunk_limit,
                ascending=bool(ascending),
                offset=chunk_offset,
            )

            if isinstance(chunk, tuple):
                chunk = chunk[0]

            if chunk is None or getattr(chunk, "empty", True):
                break

            for _, row in chunk.iterrows():
                payload, entry_type_lower, decision_status = _serialize_row(row)
                if entry_type_lower != "decision" or not decision_status:
                    continue
                if decision_status not in filters:
                    continue

                if matched_count >= target_start and len(rows) < page_size:
                    rows.append(payload)

                matched_count += 1

            if len(chunk) < chunk_limit:
                break
            chunk_offset += chunk_limit

        return {
            "items": rows,
            "page": page_number,
            "page_size": page_size,
            "total": matched_count,
        }

    # 비 페이지네이션 (기존 동작 유지)
    df = store.fetch_journals(
        symbol=symbol,
        types=fetch_types,
        today_only=effective_today_only,
        since_ts=since_ts,
        limit=page_size,
        ascending=bool(ascending),
        limit_choices=limit_choices,
    )

    if isinstance(df, tuple):
        df = df[0]

    if df is None or getattr(df, "empty", True):
        return {"items": []}

    items = []
    filters_opt = cast(Optional[set[str]], status_filters)
    for _, row in df.iterrows():
        payload, entry_type_lower, decision_status = _serialize_row(row)
        if filters_opt is not None:
            if entry_type_lower != "decision" or not decision_status:
                continue
            if decision_status not in filters_opt:
                continue
        items.append(payload)

    return {"items": items}


@app.get("/api/positions_debug")
def positions_debug(symbol: Optional[str] = None):
    bybit = BybitUtils(is_testnet=bool(int(os.getenv("TESTNET", "1"))))
    positions = bybit.get_positions() or []
    debug_items = []
    for p in positions:
        try:
            sym = p.get("symbol") or (p.get("info", {}) or {}).get("symbol")
            if not sym:
                continue
            if symbol and sym != symbol:
                continue
            side = p.get("side") or (p.get("info", {}) or {}).get("side")
            entry = p.get("entryPrice") or (p.get("info", {}) or {}).get("avgPrice")
            contract_size = p.get("contractSize") or (p.get("info", {}) or {}).get(
                "contractSize"
            )
            size_raw = p.get("size") or p.get("contracts") or p.get("amount")
            try:
                size_f = float(size_raw) if size_raw is not None else None
            except Exception:
                size_f = None
            if (
                size_f is not None
                and p.get("size") is None
                and p.get("contracts") is not None
                and contract_size is not None
            ):
                try:
                    size_f = size_f * float(contract_size)
                except Exception:
                    pass

            mark = p.get("markPrice") or (p.get("info", {}) or {}).get("markPrice")
            last = mark if mark is not None else bybit.get_last_price(sym)

            entry_f = float(entry) if entry is not None else None
            last_f = float(last) if last is not None else None

            unreal = p.get("unrealizedPnl") or (p.get("info", {}) or {}).get(
                "unrealisedPnl"
            )
            pct = p.get("percentage") or (p.get("info", {}) or {}).get(
                "unrealisedPnlPcnt"
            )
            notional = p.get("notional") or (p.get("info", {}) or {}).get(
                "positionValue"
            )
            init_margin = (
                p.get("initialMargin")
                or p.get("margin")
                or (p.get("info", {}) or {}).get("positionIM")
                or (p.get("info", {}) or {}).get("positionInitialMargin")
                or (p.get("info", {}) or {}).get("positionMargin")
            )
            lev = p.get("leverage") or (p.get("info", {}) or {}).get("leverage")
            try:
                lev = float(lev) if lev is not None else None
            except Exception:
                lev = None

            # 계산 재현
            try:
                unreal_f = float(unreal) if unreal is not None else None
            except Exception:
                unreal_f = None
            try:
                pct_f = float(pct) if pct is not None else None
            except Exception:
                pct_f = None
            pnl_calc = None
            pnl_pct_calc = None
            if unreal_f is not None:
                pnl_calc = unreal_f
            elif (
                entry_f is not None
                and last_f is not None
                and size_f is not None
                and side
            ):
                if (side or "").lower() in ("long", "buy"):
                    pnl_calc = (last_f - entry_f) * size_f
                else:
                    pnl_calc = (entry_f - last_f) * size_f
            try:
                init_margin_f = float(init_margin) if init_margin is not None else None
            except Exception:
                init_margin_f = None
            if pct_f is not None:
                pnl_pct_calc = pct_f
            elif pnl_calc is not None:
                if init_margin_f and init_margin_f > 0:
                    pnl_pct_calc = (float(pnl_calc) / init_margin_f) * 100.0
                elif entry_f and last_f is not None:
                    if (side or "").lower() in ("long", "buy"):
                        pnl_pct_calc = ((last_f - entry_f) / entry_f) * 100.0
                    else:
                        pnl_pct_calc = ((entry_f - last_f) / entry_f) * 100.0

            debug_items.append(
                {
                    "symbol": sym,
                    "raw": p,
                    "side": side,
                    "entry": entry_f,
                    "size_raw": size_raw,
                    "contractSize": contract_size,
                    "size": size_f,
                    "markPrice": mark,
                    "lastPrice": last_f,
                    "unrealizedPnl": unreal,
                    "percentage": pct,
                    "notional": notional,
                    "initialMargin": init_margin,
                    "leverage": lev,
                    "pnl_calc": pnl_calc,
                    "pnl_pct_calc": pnl_pct_calc,
                }
            )
        except Exception:
            continue
    return {"items": debug_items}


@app.get("/overlay", response_class=HTMLResponse)
def overlay(
    request: Request,
    limit: int = 10,
    symbol: Optional[str] = None,
    types: Optional[str] = None,
    today_only: int = 1,
    ascending: int = 1,
    refresh: int = 5,
    tz: Optional[str] = None,
):
    store = TradeStore(
        StorageConfig(
            mysql_url=os.getenv("MYSQL_URL"),
            sqlite_path=os.getenv("SQLITE_PATH"),
        )
    )
    type_list: Optional[List[str]] = (
        [t.strip() for t in types.split(",") if t.strip()] if types else None
    )
    df = store.fetch_journals(
        symbol=symbol,
        types=type_list,
        today_only=bool(today_only),
        limit=max(1, min(int(limit), 200)),
        ascending=bool(ascending),
    )
    items = []
    if not df.empty:
        for _, row in df.iterrows():
            ts_iso = _to_utc_iso(row.get("ts"))
            items.append(
                {
                    "ts": ts_iso,
                    "symbol": row.get("symbol"),
                    "entry_type": row.get("entry_type"),
                    "content": row.get("content"),
                    "reason": row.get("reason"),
                }
            )
    return templates.TemplateResponse(
        "overlay.html",
        {
            "request": request,
            "items": items,
            "limit": limit,
            "symbol": symbol,
            "types": types,
            "today_only": today_only,
            "ascending": ascending,
            "refresh": refresh,
            "tz": tz,
        },
    )


@app.get("/api/positions_summary")
def positions_summary(
    symbol: Optional[str] = None,
    force_mark: int = 0,
    force_exchange_pnl: int = 0,
    force_roe: int = 0,
):
    bybit = BybitUtils(is_testnet=bool(int(os.getenv("TESTNET", "1"))))
    positions = bybit.get_positions() or []
    items = _summarize_positions(
        bybit,
        positions,
        symbol=symbol,
        use_mark_price=bool(force_mark),
        force_exchange_pnl=bool(force_exchange_pnl),
        force_roe=bool(force_roe),
    )
    return {"items": items}


def _reconcile_auto_closed_positions(store: TradeStore) -> None:
    """열린(opened) 거래가 있는데 현재 포지션이 없다면 자동 청산으로 간주하여
    closed 레코드를 기록하고 저널/AI 리뷰를 추가한다. (멱등)
    """
    try:
        df = store.load_trades()
    except Exception:
        return
    if df is None or getattr(df, "empty", True):
        return

    # 필수 컬럼 확인
    for col in (
        "ts",
        "symbol",
        "side",
        "type",
        "price",
        "quantity",
        "tp",
        "sl",
        "status",
        "leverage",
    ):
        if col not in df.columns:
            return

    try:
        bybit = BybitUtils(is_testnet=bool(int(os.getenv("TESTNET", "1"))))
    except Exception:
        return

    # opened 행과 이미 closed 기록 수집
    opened_df = df[df["status"].astype(str) == "opened"].copy()
    if opened_df.empty:
        return
    try:
        import pandas as pd  # 지역 임포트

        if not pd.api.types.is_datetime64_any_dtype(opened_df["ts"]):
            opened_df["ts"] = pd.to_datetime(opened_df["ts"], errors="coerce")
        closed_df = df[df["status"].astype(str) == "closed"].copy()
        if not closed_df.empty and not pd.api.types.is_datetime64_any_dtype(
            closed_df["ts"]
        ):
            closed_df["ts"] = pd.to_datetime(closed_df["ts"], errors="coerce")
    except Exception:
        closed_df = df[df["status"].astype(str) == "closed"].copy()

    def _ts_to_ms(value: Any) -> Optional[int]:
        if value is None:
            return None
        if isinstance(value, datetime):
            if value.tzinfo is None:
                value = value.replace(tzinfo=timezone.utc)
            else:
                value = value.astimezone(timezone.utc)
            return int(value.timestamp() * 1000)
        if hasattr(value, "to_pydatetime"):
            try:
                return _ts_to_ms(value.to_pydatetime())
            except Exception:
                pass
        if isinstance(value, str):
            try:
                parsed = datetime.fromisoformat(value)
                return _ts_to_ms(parsed)
            except Exception:
                return None
        try:
            return int(float(value))
        except Exception:
            return None

    def _ms_to_naive_utc(ms: Optional[int]) -> Optional[datetime]:
        if ms is None:
            return None
        try:
            return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc).replace(
                tzinfo=None
            )
        except Exception:
            return None

    def _infer_settle_coin(sym: str) -> Optional[str]:
        if not sym:
            return None
        if ":" in sym:
            return sym.split(":")[-1]
        return None

    def _infer_category(sym: str) -> str:
        settle = _infer_settle_coin(sym)
        if settle and settle.upper() in {"USDT", "USDC"}:
            return "linear"
        return "inverse"

    existing_close_ids: set[str] = set()
    if not closed_df.empty:
        try:
            existing_close_ids.update(
                str(x) for x in closed_df.get("order_id").dropna().astype(str).tolist()
            )
        except Exception:
            for val in closed_df.get("order_id", []):
                if val:
                    existing_close_ids.add(str(val))

    processed_close_keys: set[tuple[str, str]] = set()
    if not closed_df.empty:
        try:
            for _, closed_row in closed_df.iterrows():
                sym_val = closed_row.get("symbol")
                oid_val = closed_row.get("order_id")
                if not sym_val or oid_val is None:
                    continue
                processed_close_keys.add((str(sym_val), str(oid_val)))
        except Exception:
            pass
    history_cache: Dict[str, List[Dict[str, Any]]] = {}
    used_history_ids: set[str] = set()

    symbol_since_map: Dict[str, Optional[int]] = {}
    for _, opened_row in opened_df.iterrows():
        sym_val = opened_row.get("symbol")
        if not sym_val:
            continue
        ms_val = _ts_to_ms(opened_row.get("ts"))
        if sym_val not in symbol_since_map or (
            ms_val is not None
            and (
                symbol_since_map[sym_val] is None or ms_val < symbol_since_map[sym_val]
            )
        ):
            symbol_since_map[sym_val] = ms_val

    def _fetch_history(sym: str) -> List[Dict[str, Any]]:
        if sym in history_cache:
            return history_cache[sym]
        since_ms = symbol_since_map.get(sym)
        if since_ms is not None:
            since_ms = max(0, since_ms - 60 * 60 * 1000)
        settle_coin = _infer_settle_coin(sym)
        category = _infer_category(sym)
        try:
            history = (
                bybit.get_position_history(
                    sym,
                    since_ms=since_ms,
                    limit=200,
                    category=category,
                    settle_coin=settle_coin,
                    max_pages=5,
                )
                or []
            )
        except Exception:
            history = []
        history_cache[sym] = history
        return history

    def _match_history_entry(
        sym: str, side_val: str, amount_val: float, opened_ms: Optional[int]
    ) -> Optional[Dict[str, Any]]:
        history_items = _fetch_history(sym)
        if not history_items:
            return None
        candidates = sorted(
            history_items,
            key=lambda item: item.get("updated_time") or item.get("created_time") or 0,
        )
        tolerance = max(1e-8, abs(amount_val) * 0.05) if amount_val else 1e-8
        for candidate in candidates:
            unique_id = str(
                candidate.get("order_id")
                or candidate.get("unique_id")
                or f"{candidate.get('symbol_id') or sym}:{candidate.get('updated_time') or candidate.get('created_time')}"
            )
            if unique_id and (
                unique_id in existing_close_ids or unique_id in used_history_ids
            ):
                continue
            entry_side = (candidate.get("entry_side") or "").lower()
            if entry_side and side_val and entry_side != side_val:
                continue
            updated_ms = candidate.get("updated_time") or candidate.get("created_time")
            if updated_ms and opened_ms and updated_ms + 1000 < opened_ms:
                continue
            closed_size = candidate.get("closed_size")
            if closed_size is not None and amount_val > 0:
                if abs(closed_size - amount_val) > tolerance:
                    continue
            return candidate
        return None

    try:
        opened_iter_df = opened_df.copy()
        tp_series = opened_iter_df.get("tp")
        sl_series = opened_iter_df.get("sl")
        opened_iter_df["_tp_missing"] = (
            tp_series.isna() if tp_series is not None else True
        )
        opened_iter_df["_sl_missing"] = (
            sl_series.isna() if sl_series is not None else True
        )
        sort_by: list[str] = ["symbol", "_tp_missing", "_sl_missing"]
        ascending_flags: list[bool] = [True, True, True]
        if "ts" in opened_iter_df.columns:
            sort_by.append("ts")
            ascending_flags.append(True)
        opened_iter_df = opened_iter_df.sort_values(
            by=sort_by,
            ascending=ascending_flags,
            kind="mergesort",
        )
        opened_iter_df = opened_iter_df.drop(
            columns=["_tp_missing", "_sl_missing"], errors="ignore"
        )
    except Exception:
        opened_iter_df = opened_df

    for _, row in opened_iter_df.iterrows():
        symbol = row.get("symbol")
        side = (row.get("side") or "").lower()
        if not symbol or side not in ("buy", "sell"):
            continue

        try:
            entry_price = float(row.get("price") or 0)
            amount = float(row.get("quantity") or 0)
            tp = row.get("tp")
            sl = row.get("sl")
            tp_f = float(tp) if tp is not None and tp == tp else None
            sl_f = float(sl) if sl is not None and sl == sl else None
        except Exception:
            continue
        if amount <= 0 or entry_price <= 0:
            continue

        ts_open = row.get("ts")
        try:
            has_closed = False
            if not closed_df.empty:
                mask = closed_df["symbol"].astype(str).eq(str(symbol))
                try:
                    if ts_open is not None:
                        mask = mask & (closed_df["ts"] >= ts_open)
                except Exception:
                    pass
                has_closed = bool(mask.any())
            if has_closed:
                continue
        except Exception:
            pass

        try:
            open_positions = bybit.get_positions_by_symbol(symbol) or []
        except Exception:
            open_positions = []
        if open_positions:
            continue

        vwap_close = None
        order_id = None
        closed_by = CLOSED_BY_UNKNOWN
        realized_pnl = None
        open_fee = None
        close_fee = None
        funding_fee = None
        close_ts_dt: Optional[datetime] = None

        opened_ms = _ts_to_ms(ts_open)
        candidate = _match_history_entry(symbol, side, amount, opened_ms)
        if candidate:
            unique_id = candidate.get("order_id") or candidate.get("unique_id")
            if unique_id:
                unique_str = str(unique_id)
                used_history_ids.add(unique_str)
                existing_close_ids.add(unique_str)
                order_id = unique_str
            vwap_close = candidate.get("avg_exit_price") or candidate.get(
                "avg_entry_price"
            )
            realized_pnl = candidate.get("closed_pnl")
            open_fee = candidate.get("open_fee")
            close_fee = candidate.get("close_fee")
            funding_fee = candidate.get("funding_fee")
            close_ts_dt = _ms_to_naive_utc(
                candidate.get("updated_time") or candidate.get("created_time")
            )
            exec_type = (candidate.get("exec_type") or "").lower()
            if closed_by == CLOSED_BY_UNKNOWN and exec_type:
                if "take" in exec_type and "profit" in exec_type:
                    closed_by = CLOSED_BY_TARGET_PROFIT
                elif "stop" in exec_type and "loss" in exec_type:
                    closed_by = CLOSED_BY_STOP_LOSS

        reduce_side = "sell" if side == "buy" else "buy"
        since_ms = opened_ms

        if vwap_close is None:
            try:
                trades = bybit.get_my_trades(symbol, since_ms, 1000) or []
                trades = sorted(trades, key=lambda t: t.get("timestamp") or 0)
                total_value = 0.0
                total_amount = 0.0
                for t in trades:
                    try:
                        if (t.get("side") or "").lower() != reduce_side:
                            continue
                        price = t.get("price") or (t.get("info", {}) or {}).get("price")
                        amt = t.get("amount")
                        if price is None or amt is None:
                            continue
                        price_f = float(price)
                        amt_f = float(amt)
                        if price_f <= 0 or amt_f <= 0:
                            continue
                        total_value += price_f * amt_f
                        total_amount += amt_f
                        order_id = (
                            order_id
                            or t.get("order")
                            or (t.get("info", {}) or {}).get("orderId")
                        )
                        if amount > 0 and total_amount >= amount:
                            break
                    except Exception:
                        continue
                if total_amount > 0:
                    vwap_close = total_value / total_amount
            except Exception:
                pass

        if vwap_close is None:
            try:
                closed_orders = bybit.get_closed_orders(symbol, since_ms, 100) or []
                for o in sorted(
                    closed_orders, key=lambda x: x.get("timestamp") or 0, reverse=True
                ):
                    if (o.get("side") or "").lower() != reduce_side:
                        continue
                    st = (o.get("status") or "").lower()
                    if st not in ("closed", "filled"):
                        continue
                    avg = o.get("average") or o.get("price")
                    if avg is None:
                        continue
                    vwap_close = float(avg)
                    if order_id is None:
                        order_id = o.get("id")
                    info_str = (str(o.get("type")) + " " + str(o.get("info"))).lower()
                    if "take" in info_str and "profit" in info_str:
                        closed_by = CLOSED_BY_TARGET_PROFIT
                    elif "stop" in info_str and "loss" in info_str:
                        closed_by = CLOSED_BY_STOP_LOSS
                    break
            except Exception:
                pass

        if vwap_close is None:
            try:
                last = bybit.get_last_price(symbol)
                if last is not None:
                    vwap_close = float(last)
            except Exception:
                pass
        if vwap_close is None:
            continue

        computed_pnl = (
            (vwap_close - entry_price) * amount
            if side == "buy"
            else (entry_price - vwap_close) * amount
        )
        pnl = realized_pnl if realized_pnl is not None else computed_pnl

        if closed_by == CLOSED_BY_UNKNOWN:
            is_profit = computed_pnl > PNL_EPSILON
            is_loss = computed_pnl < -PNL_EPSILON
            close_to_tp = False
            close_to_sl = False

            if tp_f is not None:
                tp_denominator = max(abs(tp_f), PNL_EPSILON)
                close_to_tp = (
                    abs(vwap_close - tp_f) / tp_denominator <= TP_TOLERANCE_RATIO
                )

            if sl_f is not None:
                sl_denominator = max(abs(sl_f), PNL_EPSILON)
                close_to_sl = (
                    abs(vwap_close - sl_f) / sl_denominator <= SL_TOLERANCE_RATIO
                )

            if is_profit and close_to_tp:
                closed_by = CLOSED_BY_TARGET_PROFIT
            elif is_loss and close_to_sl:
                closed_by = CLOSED_BY_STOP_LOSS

        if close_ts_dt is None:
            close_ts_dt = datetime.utcnow()

        if order_id is None:
            synthetic_id = f"{symbol}:{int(_ts_to_ms(close_ts_dt) or 0)}"
            order_id = synthetic_id
            existing_close_ids.add(synthetic_id)

        order_id_str: Optional[str] = str(order_id) if order_id is not None else None
        already_processed = False
        order_key: Optional[tuple[str, str]] = None
        if symbol and order_id_str:
            order_key = (str(symbol), order_id_str)
            if order_key in processed_close_keys:
                already_processed = True
            else:
                processed_close_keys.add(order_key)

        if already_processed:
            continue

        try:
            store.record_trade(
                {
                    "ts": close_ts_dt,
                    "symbol": symbol,
                    "side": side,
                    "type": row.get("type") or "market",
                    "price": float(vwap_close),
                    "quantity": amount,
                    "tp": tp_f,
                    "sl": sl_f,
                    "leverage": row.get("leverage"),
                    "status": "closed",
                    "order_id": order_id_str,
                    "pnl": float(pnl),
                }
            )
        except Exception:
            pass

        close_str = f"{vwap_close:.6f}" if vwap_close is not None else str(vwap_close)
        qty_str = f"{amount:.6f}"
        pnl_str = f"{pnl:.6f}"

        meta_payload = {
            "auto_close": True,
            "closed_by": closed_by,
            "entry_price": entry_price,
            "close_price": vwap_close,
            "tp": tp_f,
            "sl": sl_f,
            "side": side,
            "pnl": float(pnl),
        }
        if order_id_str:
            meta_payload["order_id"] = order_id_str
        if open_fee is not None:
            meta_payload["open_fee"] = open_fee
        if close_fee is not None:
            meta_payload["close_fee"] = close_fee
        if funding_fee is not None:
            meta_payload["funding_fee"] = funding_fee

        try:
            store.record_journal(
                {
                    "symbol": symbol,
                    "entry_type": "action",
                    "content": f"auto_close ({closed_by}) price={close_str} qty={qty_str} realized={pnl_str}",
                    "reason": f"Position closed by exchange due to {closed_by} or manual without signal.",
                    "meta": dict(meta_payload),
                }
            )
        except Exception:
            pass

        try:
            review = _try_ai_review(
                symbol, side, entry_price, float(vwap_close), tp_f, sl_f, float(pnl)
            )
            if review:
                review_meta = dict(meta_payload)
                review_meta["reason"] = closed_by
                store.record_journal(
                    {
                        "symbol": symbol,
                        "entry_type": "review",
                        "content": review,
                        "reason": "auto_close_review",
                        "meta": review_meta,
                    }
                )
        except Exception:
            pass


def _try_ai_review(
    symbol: str,
    side: str,
    entry_price: float,
    close_price: float,
    tp: Optional[float],
    sl: Optional[float],
    pnl: float,
) -> Optional[str]:
    """간단한 AI 리뷰 텍스트 생성. 설정이 없으면 None."""
    try:
        if not os.environ.get("OPENAI_API_KEY"):
            return None
        ai = AIProvider()
        direction = "LONG" if side == "buy" else "SHORT"
        if tp is not None and (
            (side == "buy" and close_price >= tp)
            or (side == "sell" and close_price <= tp)
        ):
            reason = "TP reached"
        elif sl is not None and (
            (side == "buy" and close_price <= sl)
            or (side == "sell" and close_price >= sl)
        ):
            reason = "SL reached"
        else:
            reason = "Unknown trigger"
        prompt = (
            "다음 자동 청산 결과를 간단히 리뷰하고 개선점을 bullet로 3개 이내로 제안해줘.\n"
            f"심볼: {symbol}\n사이드: {direction}\n진입가: {entry_price}\n청산가: {close_price}\nTP: {tp}\nSL: {sl}\n실현손익: {pnl}\n사유: {reason}\n"
        )
        text = ai.decide(prompt)
        return (text or "").strip()[:2000]
    except Exception:
        return None


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _first_float_with_key(
    candidates: list[tuple[str, Any]],
) -> tuple[Optional[float], Optional[str]]:
    for key, val in candidates:
        result = _safe_float(val)
        if result is not None:
            return result, key
    return None, None


def _compute_position_summary_item(
    bybit: BybitUtils,
    position: Dict[str, Any],
    *,
    use_mark_price: bool,
    force_exchange_pnl: bool,
    force_roe: bool,
) -> Optional[Dict[str, Any]]:
    if not isinstance(position, dict):
        return None

    info_raw = position.get("info")
    info = info_raw if isinstance(info_raw, dict) else {}

    sym = position.get("symbol") or info.get("symbol")
    if not sym:
        return None

    side = position.get("side") or info.get("side")
    entry = position.get("entryPrice") or info.get("avgPrice")
    contract_size = position.get("contractSize") or info.get("contractSize")

    size_candidates: list[tuple[str, Any]] = [
        ("size", position.get("size")),
        ("contracts", position.get("contracts")),
        ("amount", position.get("amount")),
    ]
    if info:
        size_candidates.extend(
            [
                ("info.contracts", info.get("contracts")),
                ("info.amount", info.get("amount")),
                ("info.size", info.get("size")),
                ("info.positionAmt", info.get("positionAmt")),
            ]
        )

    size_f, size_key = _first_float_with_key(size_candidates)
    if (
        size_f is not None
        and size_key
        and "contracts" in size_key
        and contract_size is not None
    ):
        contract_size_f = _safe_float(contract_size)
        if contract_size_f is not None:
            size_f = size_f * contract_size_f

    tp = position.get("takeProfit") or info.get("takeProfit")
    sl = position.get("stopLoss") or info.get("stopLoss")
    lev = _safe_float(position.get("leverage") or info.get("leverage"))
    mark_price = position.get("markPrice") or info.get("markPrice")

    entry_f = _safe_float(entry)

    if use_mark_price and mark_price is not None:
        last_source = mark_price
    else:
        last_source = (
            mark_price if mark_price is not None else bybit.get_last_price(sym)
        )
    last_f = _safe_float(last_source)

    unreal = position.get("unrealizedPnl") or info.get("unrealisedPnl")
    unreal_f = _safe_float(unreal)

    pct = position.get("percentage") or info.get("unrealisedPnlPcnt")
    pct_f = _safe_float(pct)

    notional = position.get("notional") or info.get("positionValue")
    notional_f = _safe_float(notional)
    if notional_f is None and entry_f is not None and size_f is not None:
        notional_f = abs(entry_f * size_f)

    initial_margin = (
        position.get("initialMargin")
        or position.get("margin")
        or info.get("positionIM")
        or info.get("positionInitialMargin")
        or info.get("positionMargin")
    )
    initial_margin_f = _safe_float(initial_margin)
    if initial_margin_f is None and notional_f is not None:
        if lev:
            try:
                initial_margin_f = notional_f / lev
            except Exception:
                initial_margin_f = None

    side_norm = (side or "").lower() if isinstance(side, str) else None

    pnl = None
    if force_exchange_pnl and unreal_f is not None:
        pnl = unreal_f
    else:
        if unreal_f is not None:
            pnl = unreal_f
        elif (
            entry_f is not None
            and size_f is not None
            and last_f is not None
            and side_norm
        ):
            if side_norm in ("long", "buy"):
                pnl = (last_f - entry_f) * size_f
            else:
                pnl = (entry_f - last_f) * size_f

    pnl_pct = None
    if force_roe:
        if pnl is not None and initial_margin_f and initial_margin_f > 0:
            pnl_pct = (float(pnl) / initial_margin_f) * 100.0
        elif (
            pnl is not None and entry_f is not None and last_f is not None and side_norm
        ):
            if side_norm in ("long", "buy"):
                pnl_pct = ((last_f - entry_f) / entry_f) * 100.0
            else:
                pnl_pct = ((entry_f - last_f) / entry_f) * 100.0
    else:
        if pct_f is not None:
            pnl_pct = pct_f
        elif pnl is not None:
            if initial_margin_f and initial_margin_f > 0:
                pnl_pct = (float(pnl) / initial_margin_f) * 100.0
            elif entry_f is not None and last_f is not None and side_norm:
                if side_norm in ("long", "buy"):
                    pnl_pct = ((last_f - entry_f) / entry_f) * 100.0
                else:
                    pnl_pct = ((entry_f - last_f) / entry_f) * 100.0

    return {
        "symbol": sym,
        "side": side,
        "entryPrice": entry_f,
        "lastPrice": last_f,
        "size": size_f,
        "tp": _safe_float(tp),
        "sl": _safe_float(sl),
        "leverage": lev,
        "pnl": pnl,
        "pnlPct": pnl_pct,
    }


def _summarize_positions(
    bybit: BybitUtils,
    positions: Any,
    *,
    symbol: Optional[str],
    use_mark_price: bool,
    force_exchange_pnl: bool,
    force_roe: bool,
) -> List[Dict[str, Any]]:
    if not positions:
        return []

    items: List[Dict[str, Any]] = []
    for position in positions:
        if not isinstance(position, dict):
            continue
        info_raw = position.get("info")
        info = info_raw if isinstance(info_raw, dict) else {}
        sym = position.get("symbol") or info.get("symbol")
        if not sym:
            continue
        if symbol and sym != symbol:
            continue
        item = _compute_position_summary_item(
            bybit,
            position,
            use_mark_price=use_mark_price,
            force_exchange_pnl=force_exchange_pnl,
            force_roe=force_roe,
        )
        if item:
            items.append(item)
    return items


@app.get("/overlay_positions", response_class=HTMLResponse)
def overlay_positions(
    request: Request,
    symbol: Optional[str] = None,
    refresh: int = 5,
):
    return templates.TemplateResponse(
        "overlay_positions.html",
        {"request": request, "symbol": symbol, "refresh": refresh},
    )
