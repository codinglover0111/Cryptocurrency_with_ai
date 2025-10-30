from __future__ import annotations

import os
import json
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

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
    bybit = BybitUtils(is_testnet=bool(int(os.getenv("TESTNET", "1"))))
    data = bybit.get_account_overview()
    try:
        if isinstance(data, dict):
            bal = data.get("balance")
            if isinstance(bal, dict) and "raw" in bal:
                bal.pop("raw", None)
    except Exception:
        pass
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
                from datetime import datetime

                return datetime.fromisoformat(ts + "T00:00:00+00:00")
            from datetime import datetime

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
        # 문자열이면 JSON 파싱 시도 후 민감 정보 마스킹
        try:
            if isinstance(meta_val, str):
                try:
                    meta_val = json.loads(meta_val)
                except Exception:
                    pass
            meta_val = _redact_sensitive(meta_val)
        except Exception:
            pass
        items.append(
            {
                "ts": ts_iso,
                "symbol": row.get("symbol"),
                "entry_type": row.get("entry_type"),
                "content": row.get("content"),
                "reason": row.get("reason"),
                "meta": meta_val,
                "ref_order_id": row.get("ref_order_id"),
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
):
    """안전 공개용 저널 API: types는 SQL에 직접 반영하지 않고 서버에서 필터링.

    - 허용 타입만 필터: thought | decision | action | review
    - today_only, limit, ascending 등은 DB 쿼리 파라미터로 전달
    - meta는 문자열일 경우 JSON 파싱 후 마스킹
    """
    store = TradeStore(
        StorageConfig(
            mysql_url=os.getenv("MYSQL_URL"),
            sqlite_path=os.getenv("SQLITE_PATH"),
        )
    )

    allowed_types = {"thought", "decision", "action", "review"}
    req_types: Optional[List[str]] = (
        [t.strip() for t in types.split(",") if t.strip()] if types else None
    )
    if req_types:
        req_types = [t for t in req_types if t in allowed_types]

    # DB에서는 타입 미적용으로 조회 (SQL에 types 직접 반영하지 않음)
    df = store.fetch_journals(
        symbol=symbol,
        types=None,
        today_only=bool(today_only),
        limit=max(1, min(int(limit), 200)),
        ascending=bool(ascending),
    )

    if df.empty:
        return {"items": []}

    try:
        if req_types:
            df = df[df["entry_type"].astype(str).isin(req_types)]
    except Exception:
        pass

    items = []
    for _, row in df.iterrows():
        ts_iso = _to_utc_iso(row.get("ts"))
        meta_val = row.get("meta")
        try:
            if isinstance(meta_val, str):
                try:
                    meta_val = json.loads(meta_val)
                except Exception:
                    pass
            meta_val = _redact_sensitive(meta_val)
        except Exception:
            pass
        items.append(
            {
                "ts": ts_iso,
                "symbol": row.get("symbol"),
                "entry_type": row.get("entry_type"),
                "content": row.get("content"),
                "reason": row.get("reason"),
                "meta": meta_val,
                "ref_order_id": row.get("ref_order_id"),
            }
        )
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
    items = []
    for p in positions:
        try:
            sym = p.get("symbol") or (p.get("info", {}) or {}).get("symbol")
            if not sym:
                continue
            if symbol and sym != symbol:
                continue
            side = p.get("side") or (p.get("info", {}) or {}).get("side")
            entry = p.get("entryPrice") or (p.get("info", {}) or {}).get("avgPrice")
            # 수량: size(기본 단위) 우선, 없으면 contracts * contractSize 로 환산
            contract_size = p.get("contractSize") or (p.get("info", {}) or {}).get(
                "contractSize"
            )
            size_raw = p.get("size") or p.get("contracts") or p.get("amount")
            try:
                size_f = float(size_raw) if size_raw is not None else None
            except Exception:
                size_f = None
            if size_f is None and size_raw is not None:
                # 방어적 캐스팅 실패 대비
                try:
                    size_f = float(size_raw)
                except Exception:
                    size_f = None
            # contracts만 있는 경우 contractSize 곱해 기본 단위로 변환
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
            tp = (
                p.get("takeProfit")
                or (p.get("info", {}) or {}).get("takeProfit")
                or None
            )
            sl = p.get("stopLoss") or (p.get("info", {}) or {}).get("stopLoss") or None
            lev = p.get("leverage") or (p.get("info", {}) or {}).get("leverage")
            try:
                lev = float(lev) if lev is not None else None
            except Exception:
                lev = None

            # 가격 소스: 강제 옵션(force_mark) 시 마크 프라이스 고정, 아니면 마크 우선
            mark = p.get("markPrice") or (p.get("info", {}) or {}).get("markPrice")
            if force_mark:
                last = mark if mark is not None else bybit.get_last_price(sym)
            else:
                last = mark if mark is not None else bybit.get_last_price(sym)

            entry_f = float(entry) if entry is not None else None
            last_f = float(last) if last is not None else None

            pnl = None
            pnl_pct = None

            # 거래소 제공 미실현손익(가능 시 우선 사용)
            unreal = p.get("unrealizedPnl") or (p.get("info", {}) or {}).get(
                "unrealisedPnl"
            )
            try:
                unreal_f = float(unreal) if unreal is not None else None
            except Exception:
                unreal_f = None

            # 거래소 제공 퍼센트(가능 시)
            pct = p.get("percentage") or (p.get("info", {}) or {}).get(
                "unrealisedPnlPcnt"
            )
            try:
                pct_f = float(pct) if pct is not None else None
            except Exception:
                pct_f = None

            # ROE 계산을 위한 명목가치/초기증거금 산출
            notional = p.get("notional") or (p.get("info", {}) or {}).get(
                "positionValue"
            )
            try:
                notional_f = float(notional) if notional is not None else None
            except Exception:
                notional_f = None
            if notional_f is None and entry_f is not None and size_f is not None:
                notional_f = abs(entry_f * size_f)

            # 초기증거금: 격리/크로스 모두 커버하도록 다양한 필드 참조
            initial_margin = (
                p.get("initialMargin")
                or p.get("margin")
                or (p.get("info", {}) or {}).get("positionIM")
                or (p.get("info", {}) or {}).get("positionInitialMargin")
                or (p.get("info", {}) or {}).get("positionMargin")
            )
            try:
                initial_margin_f = (
                    float(initial_margin) if initial_margin is not None else None
                )
            except Exception:
                initial_margin_f = None
            if (initial_margin_f is None) and (notional_f is not None):
                try:
                    initial_margin_f = notional_f / float(lev) if lev else None
                except Exception:
                    initial_margin_f = None

            # PnL 값 결정: 거래소 제공치 → 자체 계산(마크/라스트)
            if force_exchange_pnl:
                pnl = unreal_f  # 폴백 금지
            else:
                if unreal_f is not None:
                    pnl = unreal_f
                elif (
                    entry_f is not None
                    and size_f is not None
                    and last_f is not None
                    and side
                ):
                    if (side or "").lower() in ("long", "buy"):
                        pnl = (last_f - entry_f) * size_f
                    else:
                        pnl = (entry_f - last_f) * size_f

            # 퍼센트: 강제 ROE면 ROE만, 아니면 거래소 percentage 우선 → ROE → 가격 변화율
            if force_roe:
                if pnl is not None and initial_margin_f and initial_margin_f > 0:
                    pnl_pct = (float(pnl) / initial_margin_f) * 100.0
                elif pnl is not None and entry_f and last_f is not None:
                    if (side or "").lower() in ("long", "buy"):
                        pnl_pct = ((last_f - entry_f) / entry_f) * 100.0
                    else:
                        pnl_pct = ((entry_f - last_f) / entry_f) * 100.0
            else:
                if pct_f is not None:
                    pnl_pct = pct_f
                elif pnl is not None:
                    if initial_margin_f and initial_margin_f > 0:
                        pnl_pct = (float(pnl) / initial_margin_f) * 100.0
                    elif entry_f and last_f is not None:
                        if (side or "").lower() in ("long", "buy"):
                            pnl_pct = ((last_f - entry_f) / entry_f) * 100.0
                        else:
                            pnl_pct = ((entry_f - last_f) / entry_f) * 100.0
            items.append(
                {
                    "symbol": sym,
                    "side": side,
                    "entryPrice": entry_f,
                    "lastPrice": last_f,
                    "size": size_f,
                    "tp": float(tp) if tp is not None else None,
                    "sl": float(sl) if sl is not None else None,
                    "leverage": lev,
                    "pnl": pnl,
                    "pnlPct": pnl_pct,
                }
            )
        except Exception:
            continue
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

    for _, row in opened_df.iterrows():
        symbol = row.get("symbol")
        side = (row.get("side") or "").lower()
        if not symbol or side not in ("buy", "sell"):
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

        # 아직 open인지 확인
        try:
            open_positions = bybit.get_positions_by_symbol(symbol) or []
        except Exception:
            open_positions = []
        if open_positions:
            continue

        # 거래/주문 히스토리 기반으로 청산가(VWAP) 산출
        vwap_close = None
        order_id = None
        closed_by = CLOSED_BY_UNKNOWN
        reduce_side = "sell" if side == "buy" else "buy"
        since_ms = None
        try:
            if hasattr(ts_open, "timestamp"):
                since_ms = int(ts_open.timestamp() * 1000)
        except Exception:
            pass
        try:
            trades = bybit.get_my_trades(symbol, since_ms, 1000) or []
            trades = sorted(trades, key=lambda t: t.get("timestamp") or 0)
            total_value = 0.0
            total_amount = 0.0
            target_qty = float(row.get("quantity") or 0)
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
                    if target_qty > 0 and total_amount >= target_qty:
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
                    order_id = order_id or o.get("id")
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

        if side == "buy":
            pnl = (vwap_close - entry_price) * amount
        else:
            pnl = (entry_price - vwap_close) * amount

        if closed_by == CLOSED_BY_UNKNOWN:
            is_profit = pnl > PNL_EPSILON
            is_loss = pnl < -PNL_EPSILON
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

        # closed 레코드 기록
        try:
            store.record_trade(
                {
                    "ts": datetime.utcnow(),
                    "symbol": symbol,
                    "side": side,
                    "type": row.get("type") or "market",
                    "price": float(vwap_close),
                    "quantity": amount,
                    "tp": tp_f,
                    "sl": sl_f,
                    "leverage": row.get("leverage"),
                    "status": "closed",
                    "order_id": order_id,
                    "pnl": float(pnl),
                }
            )
        except Exception:
            pass

        # 저널: action
        try:
            store.record_journal(
                {
                    "symbol": symbol,
                    "entry_type": "action",
                    "content": f"auto_close ({closed_by}) price={vwap_close} qty={amount}",
                    "reason": f"Position closed by exchange due to {closed_by} or manual without signal.",
                    "meta": {
                        "auto_close": True,
                        "closed_by": closed_by,
                        "entry_price": entry_price,
                        "close_price": vwap_close,
                        "tp": tp_f,
                        "sl": sl_f,
                        "side": side,
                        "pnl": float(pnl),
                    },
                }
            )
        except Exception:
            pass

        # 저널: AI 리뷰
        try:
            review = _try_ai_review(
                symbol, side, entry_price, float(vwap_close), tp_f, sl_f, float(pnl)
            )
            if review:
                store.record_journal(
                    {
                        "symbol": symbol,
                        "entry_type": "review",
                        "content": review,
                        "reason": "auto_close_review",
                        "meta": {
                            "auto_close": True,
                            "pnl": float(pnl),
                            "closed_by": closed_by,
                        },
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
        provider = os.getenv("AI_PROVIDER", "gemini").lower()
        if provider == "gemini" and not os.environ.get("GEMINI_API_KEY"):
            return None
        if provider != "gemini" and not os.environ.get("OPENAI_API_KEY"):
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
