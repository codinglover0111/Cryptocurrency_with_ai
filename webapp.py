from __future__ import annotations

import os
from datetime import datetime
from typing import Optional, List

from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from utils.bybit_utils import BybitUtils
from utils.storage import TradeStore, StorageConfig
from utils.ai_provider import AIProvider


app = FastAPI(title="Crypto Bot UI")


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
    return {"status": "ok", "ts": datetime.utcnow().isoformat()}


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


@app.post("/close_all")
def close_all():
    bybit = BybitUtils(is_testnet=bool(int(os.getenv("TESTNET", "1"))))
    res = bybit.close_all_positions()
    return {"ok": True, "result": res}


def _parse_symbols():
    raw = os.getenv(
        "TRADING_SYMBOLS",
        "XRPUSDT,WLDUSDT,ETHUSDT,BTCUSDT,SOLUSDT,DOGEUSDT",
    )
    return [s.strip().upper() for s in raw.split(",") if s.strip()]


def _to_ccxt_symbols(symbol_usdt: str):
    s = symbol_usdt.upper().replace(":USDT", "").replace("/", "")
    base = s[:-4] if s.endswith("USDT") else s
    spot = f"{base}/USDT"
    contract = f"{base}/USDT:USDT"
    return spot, contract


@app.get("/symbols")
def symbols():
    codes = _parse_symbols()
    items = []
    for c in codes:
        spot, contract = _to_ccxt_symbols(c)
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
    ascending: int = 1,
):
    store = TradeStore(
        StorageConfig(
            mysql_url=os.getenv("MYSQL_URL"),
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
        ts = row.get("ts")
        ts_iso = ts.isoformat() if hasattr(ts, "isoformat") else str(ts)
        items.append(
            {
                "ts": ts_iso,
                "symbol": row.get("symbol"),
                "entry_type": row.get("entry_type"),
                "content": row.get("content"),
                "reason": row.get("reason"),
                "meta": _redact_sensitive(row.get("meta")),
                "ref_order_id": row.get("ref_order_id"),
            }
        )
    return {"items": items}


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
            ts = row.get("ts")
            ts_iso = ts.isoformat() if hasattr(ts, "isoformat") else str(ts)
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
def positions_summary(symbol: Optional[str] = None):
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
            size = p.get("contracts") or p.get("amount") or p.get("size")
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

            # 거래소 UI와 일치하도록 마크 프라이스 우선 사용, 불가 시 최근 체결가 사용
            mark = p.get("markPrice") or (p.get("info", {}) or {}).get("markPrice")
            last = mark if mark is not None else bybit.get_last_price(sym)

            entry_f = float(entry) if entry is not None else None
            size_f = float(size) if size is not None else None
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

            initial_margin = p.get("initialMargin") or (p.get("info", {}) or {}).get(
                "positionInitialMargin"
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

            # 퍼센트: ROE(초기증거금 대비)가 우선, 불가 시 가격 변화율로 폴백
            if pnl is not None:
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
        closed_by = "unknown"
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
                        closed_by = "tp"
                    elif "stop" in info_str and "loss" in info_str:
                        closed_by = "sl"
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
            if closed_by == "unknown":
                closed_by = (
                    "tp"
                    if (tp_f is not None and vwap_close >= tp_f)
                    else (
                        "sl" if (sl_f is not None and vwap_close <= sl_f) else "unknown"
                    )
                )
        else:
            pnl = (entry_price - vwap_close) * amount
            if closed_by == "unknown":
                closed_by = (
                    "tp"
                    if (tp_f is not None and vwap_close <= tp_f)
                    else (
                        "sl" if (sl_f is not None and vwap_close >= sl_f) else "unknown"
                    )
                )

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
