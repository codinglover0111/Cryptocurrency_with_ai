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


app = FastAPI(title="Crypto Bot UI")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


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
    return bybit.get_account_overview()


@app.post("/leverage")
def set_leverage(body: LeverageBody):
    bybit = BybitUtils(is_testnet=bool(int(os.getenv("TESTNET", "1"))))
    res = bybit.set_leverage(body.symbol, body.leverage, body.margin_mode or "cross")
    return {"ok": True, "result": res}


@app.get("/stats")
def stats():
    store = TradeStore(
        StorageConfig(
            xlsx_path=os.getenv("TRADES_XLSX", "trades.xlsx"),
            mysql_url=os.getenv("MYSQL_URL"),
        )
    )
    return store.compute_stats()


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
            xlsx_path=os.getenv("TRADES_XLSX", "trades.xlsx"),
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
            xlsx_path=os.getenv("TRADES_XLSX", "trades.xlsx"),
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
                "meta": row.get("meta"),
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
):
    store = TradeStore(
        StorageConfig(
            xlsx_path=os.getenv("TRADES_XLSX", "trades.xlsx"),
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

            last = bybit.get_last_price(sym)
            entry_f = float(entry) if entry is not None else None
            size_f = float(size) if size is not None else None
            last_f = float(last) if last is not None else None
            pnl = None
            pnl_pct = None
            if (
                entry_f is not None
                and size_f is not None
                and last_f is not None
                and side
            ):
                if (side or "").lower() in ("long", "buy"):
                    pnl = (last_f - entry_f) * size_f
                    pnl_pct = ((last_f - entry_f) / entry_f) * 100.0
                else:
                    pnl = (entry_f - last_f) * size_f
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
