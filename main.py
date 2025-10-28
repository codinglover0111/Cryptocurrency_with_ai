import os
import json
from utils import bybit_utils, BybitUtils, Open_Position, make_to_object
from utils.risk import calculate_position_size, enforce_max_loss_sl
from utils.storage import TradeStore, StorageConfig
from utils.ai_provider import AIProvider
from dotenv import load_dotenv
import schedule
import ccxt
import pandas as pd
import time
from datetime import datetime
import pytz
import logging

# TODO: 추후 AI 관련 부분을 클래스로 묶어 리팩토링해야함

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("trading.log"), logging.StreamHandler()],
)

load_dotenv()


def round_price(price):
    """XRP/USDT 가격을 최소 정밀도(0.0001)에 맞게 반올림"""
    return round(float(price), 4)


def _parse_symbols():
    """환경변수 TRADING_SYMBOLS에서 심볼 리스트를 파싱 (기본: 주요 6종)."""
    raw = os.getenv(
        "TRADING_SYMBOLS",
        "XRPUSDT,WLDUSDT,ETHUSDT,BTCUSDT,SOLUSDT,DOGEUSDT",
    )
    syms = [s.strip().upper() for s in raw.split(",") if s.strip()]
    return syms or [
        "XRPUSDT",
        "WLDUSDT",
        "ETHUSDT",
        "BTCUSDT",
        "SOLUSDT",
        "DOGEUSDT",
    ]


def _to_ccxt_symbols(symbol_usdt: str):
    """BTCUSDT -> (spot_symbol, contract_symbol) = (BTC/USDT, BTC/USDT:USDT)"""
    s = symbol_usdt.upper().replace(":USDT", "").replace("/", "")
    if s.endswith("USDT"):
        base = s[:-4]
    else:
        base = s
    spot = f"{base}/USDT"
    contract = f"{base}/USDT:USDT"
    return spot, contract


def _contract_to_spot_symbol(contract_symbol: str) -> str:
    """e.g. BTC/USDT:USDT -> BTC/USDT"""
    try:
        return str(contract_symbol).replace(":USDT", "")
    except Exception:
        return contract_symbol


def _ohlcv_csv_between(
    spot_symbol: str, timeframe: str, since_ms: int, until_ms: int
) -> str:
    """지정 기간의 OHLCV를 CSV 문자열로 반환 (인덱스: timestamp)."""
    try:
        tf_to_ms = {
            "1m": 60_000,
            "3m": 180_000,
            "5m": 300_000,
            "15m": 900_000,
            "1h": 3_600_000,
            "4h": 14_400_000,
            "1d": 86_400_000,
        }
        step = tf_to_ms.get(timeframe, 60_000)
        est_bars = max(1, int((until_ms - since_ms) / step) + 5)
        limit = min(1000, est_bars)
        ex = ccxt.bybit()
        rows = ex.fetch_ohlcv(spot_symbol, timeframe, since=since_ms, limit=limit) or []
        # 기간 내로 제한
        rows = [r for r in rows if len(r) >= 6 and r[0] <= until_ms]
        if not rows:
            return ""
        df = pd.DataFrame(
            rows, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )  # type: ignore
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        return df.to_csv()
    except Exception:
        return ""


def _format_trade_reviews_for_prompt(store: TradeStore, contract_symbol: str) -> str:
    """해당 심볼의 최신 리뷰 5개를 프롬프트 텍스트로 구성"""
    try:
        df = store.fetch_journals(
            symbol=contract_symbol, types=["review"], limit=5, ascending=False
        )
        if df is None or getattr(df, "empty", True):
            return ""
        lines = []
        for _, row in df.iterrows():
            ts = row.get("ts")
            ts_str = (
                ts.strftime("%Y-%m-%d %H:%M:%S") if hasattr(ts, "strftime") else str(ts)
            )
            reason = row.get("reason") or ""
            content = row.get("content") or ""
            if len(content) > 500:
                content = content[:500]
            lines.append(f"[{ts_str}] {reason} | {content}")
        return "\n".join(lines)
    except Exception:
        return ""


def _review_losing_trades(store: TradeStore, since_minutes: int = 600) -> None:
    """최근 N분 내 실현 손익이 발생한(trades.status=closed, pnl!=0) 주문을 리뷰하여 journals(review)로 저장"""
    try:
        trades_df = store.load_trades()
        if trades_df is None or getattr(trades_df, "empty", True):
            return
        # 시각 정규화
        try:
            trades_df["ts"] = pd.to_datetime(trades_df["ts"], errors="coerce", utc=True)
        except Exception:
            trades_df["ts"] = pd.to_datetime(trades_df["ts"], errors="coerce", utc=True)

        now_utc = (
            pd.Timestamp.utcnow().tz_localize("UTC")
            if getattr(pd.Timestamp.utcnow(), "tz", None) is None
            else pd.Timestamp.utcnow()
        )
        since_ts = now_utc - pd.Timedelta(minutes=int(since_minutes))

        closed_recent = trades_df[
            (trades_df["status"] == "closed")
            & (trades_df["pnl"].notna())
            & (trades_df["pnl"] <= 0)
            & (trades_df["ts"] >= since_ts)
        ].copy()
        if getattr(closed_recent, "empty", True):
            return

        # 중복 리뷰 방지용 최근 리뷰 읽기
        recent_reviews = {}
        try:
            # 최근 100개 리뷰 메모리에 적재 (심볼별)
            review_df = store.fetch_journals(
                types=["review"], limit=100, ascending=False
            )
            if review_df is not None and not getattr(review_df, "empty", True):
                for _, r in review_df.iterrows():
                    sym = r.get("symbol")
                    meta = r.get("meta")
                    try:
                        if isinstance(meta, str):
                            meta = json.loads(meta)
                    except Exception:
                        meta = None
                    if not isinstance(meta, dict):
                        continue
                    key = (sym, meta.get("closed_ts"))
                    recent_reviews[key] = True
        except Exception:
            pass

        ai = AIProvider()
        for _, row in closed_recent.sort_values("ts").iterrows():
            contract_symbol = str(row.get("symbol"))
            side = row.get("side")
            close_ts = row.get("ts")
            pnl = float(row.get("pnl") or 0.0)

            # 중복 체크
            closed_key = (contract_symbol, pd.Timestamp(close_ts).isoformat())
            if recent_reviews.get(closed_key):
                continue

            # 오픈 시각 추정: 동일 심볼의 가장 최근 opened
            opened_df = trades_df[
                (trades_df["symbol"].astype(str) == contract_symbol)
                & (trades_df["status"] == "opened")
                & (trades_df["ts"] <= close_ts)
            ]
            if not getattr(opened_df, "empty", True):
                open_ts = opened_df["ts"].max()
                entry_price = float(
                    opened_df.sort_values("ts").iloc[-1].get("price") or 0
                )
            else:
                # 폴백: 종료 2시간 전으로 가정
                open_ts = pd.Timestamp(close_ts) - pd.Timedelta(hours=2)
                entry_price = None

            spot_symbol = _contract_to_spot_symbol(contract_symbol)
            since_ms = int(pd.Timestamp(open_ts).timestamp() * 1000)
            until_ms = int(pd.Timestamp(close_ts).timestamp() * 1000)

            csv_1m = _ohlcv_csv_between(spot_symbol, "1m", since_ms, until_ms)

            # 리뷰 프롬프트 구성
            open_ts_str = pd.Timestamp(open_ts).strftime("%Y-%m-%d %H:%M:%S UTC")
            close_ts_str = pd.Timestamp(close_ts).strftime("%Y-%m-%d %H:%M:%S UTC")
            is_loss = pnl < 0
            role_line = "손실 원인 분석가" if is_loss else "수익 요인 분석가"
            task_line = (
                "아래 CSV_1m 구간의 가격 흐름을 참고하여, 손실 발생의 핵심 원인과 재발 방지를 위한 교훈/체크리스트를 3~5개 불릿으로 제시하세요. 600자 이내.\n"
                if is_loss
                else "아래 CSV_1m 구간의 가격 흐름을 참고하여, 수익 발생의 핵심 요인과 재현 방법, 리스크 관리/익절·손절 개선 포인트를 3~5개 불릿으로 제시하세요. 600자 이내.\n"
            )
            prompt = (
                f"당신은 암호화폐 트레이딩 {role_line}입니다. 한국어로 간결히 답하세요.\n"
                f"심볼: {contract_symbol} (spot={spot_symbol})\n"
                f"포지션: {side}, 손익: {pnl} USDT\n"
                f"기간: {open_ts_str} ~ {close_ts_str}\n"
                + (f"진입가(추정): {entry_price}\n" if entry_price else "")
                + task_line
                + "[CSV_1m_BETWEEN]\n"
                + (csv_1m or "(no data)")
            )

            try:
                review_text = ai.decide(prompt)
            except Exception as _e:
                logging.error(f"Trade review LLM failed: {_e}")
                review_text = None

            try:
                store.record_journal(
                    {
                        "symbol": contract_symbol,
                        "entry_type": "review",
                        "content": review_text or "리뷰 생성 실패",
                        "reason": ("loss_review" if pnl < 0 else "win_review"),
                        "meta": {
                            "closed_ts": pd.Timestamp(close_ts).isoformat(),
                            "opened_ts": pd.Timestamp(open_ts).isoformat(),
                            "pnl": pnl,
                            "side": side,
                            "spot_symbol": spot_symbol,
                            "timeframe": "1m",
                        },
                    }
                )
            except Exception as _e:
                logging.error(f"Journal review write failed: {_e}")
    except Exception as e:
        logging.error(f"review_losing_trades failed: {e}")


def automation_for_symbol(symbol_usdt: str):
    try:
        is_testnet = bool(int(os.getenv("TESTNET", "1")))
        bybit = BybitUtils(is_testnet)
        store = TradeStore(
            StorageConfig(
                mysql_url=os.getenv("MYSQL_URL"),
                sqlite_path=os.getenv("SQLITE_PATH"),
            )
        )

        spot_symbol, contract_symbol = _to_ccxt_symbols(symbol_usdt)

        # 심볼 수 기준 균등 배분 비율(%) 계산
        try:
            num_symbols = max(1, len(_parse_symbols()))
        except Exception:
            num_symbols = 1
        per_symbol_alloc_pct = 100.0 / float(num_symbols)

        # 현재 포지션 체크(심볼 한정)
        current_position = bybit.get_positions_by_symbol(contract_symbol)
        # TODO 리팩토링 진행
        # if len(current_position)>0:
        #     logging.info("Active position exists, skipping trading cycle")
        #     return

        # 현재 오픈오더 확인(미사용 시 주석 처리)
        # current_orders = bybit.get_orders()
        # if current_orders is not None and len(current_position)==0:
        # # 포지션이 없으며 현재 오더가 있으면 동작
        #   logging.info("Active position exists, cancle current orders")
        #   bybit.cancle_orders()

        # AI 설정 (Gemini 또는 OpenAI 호환)
        ai_provider = AIProvider()

        # 캔들 데이터(이미지 대신 CSV 텍스트 전송)

        price_utils = bybit_utils(spot_symbol, "4h", 100)
        df_4h = price_utils.get_ohlcv()
        csv_4h = df_4h.to_csv()
        price_utils.set_timeframe("1h")
        df_1h = price_utils.get_ohlcv()
        csv_1h = df_1h.to_csv()
        price_utils.set_timeframe("15m")
        df_15m = price_utils.get_ohlcv()
        csv_15m = df_15m.to_csv()
        current_price = df_15m["close"].iloc[-1]

        if len(current_position) > 0:
            pos = current_position[0]
            pos_side = (
                pos.get("info", {}).get("side")
                if isinstance(pos.get("info"), dict)
                else pos.get("side")
            )
        else:
            pos_side = None

        # 현재 심볼 포지션 상세를 LLM에 전달하기 위한 요약 문자열 구성
        current_positions_lines = []
        try:
            last_fallback = float(current_price or 0)
            for p in current_position or []:
                try:
                    sym = p.get("symbol") or (p.get("info", {}) or {}).get("symbol")
                    if sym != contract_symbol:
                        continue
                    sidep = p.get("side") or (p.get("info", {}) or {}).get("side")
                    entry = p.get("entryPrice") or (p.get("info", {}) or {}).get(
                        "avgPrice"
                    )
                    contract_size = p.get("contractSize") or (
                        p.get("info", {}) or {}
                    ).get("contractSize")
                    size_raw = p.get("size") or p.get("contracts") or p.get("amount")
                    try:
                        size_f = float(size_raw) if size_raw is not None else None
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
                    mark = p.get("markPrice") or (p.get("info", {}) or {}).get(
                        "markPrice"
                    )
                    try:
                        last = float(mark) if mark is not None else float(last_fallback)
                    except Exception:
                        last = last_fallback
                    try:
                        entry_f = float(entry) if entry is not None else None
                    except Exception:
                        entry_f = None
                    unreal = p.get("unrealizedPnl") or (p.get("info", {}) or {}).get(
                        "unrealisedPnl"
                    )
                    pct = p.get("percentage") or (p.get("info", {}) or {}).get(
                        "unrealisedPnlPcnt"
                    )
                    tp = p.get("takeProfit") or (p.get("info", {}) or {}).get(
                        "takeProfit"
                    )
                    sl = p.get("stopLoss") or (p.get("info", {}) or {}).get("stopLoss")
                    lev = p.get("leverage") or (p.get("info", {}) or {}).get("leverage")

                    def _fmt(v):
                        try:
                            return f"{float(v):.6f}"
                        except Exception:
                            return str(v)

                    line = (
                        f"side={sidep}, "
                        f"size={_fmt(size_f) if size_f is not None else 'n/a'}, "
                        f"entry={_fmt(entry_f) if entry_f is not None else 'n/a'}, "
                        f"last={_fmt(last)}, "
                        f"tp={_fmt(tp) if tp is not None else 'n/a'}, "
                        f"sl={_fmt(sl) if sl is not None else 'n/a'}, "
                        f"lev={_fmt(lev) if lev is not None else 'n/a'}, "
                        f"unreal={_fmt(unreal) if unreal is not None else 'n/a'} ({_fmt(pct)}%)"
                    )
                    current_positions_lines.append(line)
                except Exception:
                    continue
        except Exception:
            current_positions_lines = []

        # 오늘 저널(같은 심볼 기준) 조회 후 프롬프트에 반영
        try:
            journals_today_df = store.fetch_journals(
                symbol=contract_symbol, today_only=True, limit=50, ascending=True
            )
            journal_today_lines = []
            if not journals_today_df.empty:
                for _, row in journals_today_df.iterrows():
                    ts = row.get("ts")
                    ts_str = (
                        ts.strftime("%H:%M:%S") if hasattr(ts, "strftime") else str(ts)
                    )
                    journal_today_lines.append(
                        f"[{ts_str}] ({row.get('entry_type')}) {row.get('reason') or ''} | {row.get('content') or ''}"
                    )
            journal_today_text = "\n".join(journal_today_lines)
        except Exception:
            journal_today_text = ""

        # 최근 리포트(결정/리뷰 위주, hold 포함) 10개 수집
        recent_reports_text = ""
        try:
            recent_df = store.fetch_journals(
                symbol=contract_symbol,
                types=["decision", "review"],
                limit=10,
                ascending=False,
            )
            if not recent_df.empty:
                lines = []
                # 최신순으로 가져왔으니 시간 흐름대로 보기 좋게 역순 정렬
                for _, row in recent_df.iloc[::-1].iterrows():
                    ts = row.get("ts")
                    ts_str = (
                        ts.strftime("%Y-%m-%d %H:%M:%S")
                        if hasattr(ts, "strftime")
                        else str(ts)
                    )
                    et = row.get("entry_type") or ""
                    reason = row.get("reason") or ""
                    content = row.get("content") or ""
                    # decision 내용이 JSON 문자열인 경우 간단 요약 시도
                    try:
                        obj = json.loads(content)
                        status = obj.get("status") or obj.get("Status")
                        price = obj.get("price")
                        tp_v = obj.get("tp")
                        sl_v = obj.get("sl")
                        lev = obj.get("leverage")
                        brief = (
                            f"status={status} price={price} tp={tp_v} sl={sl_v} lev={lev}"
                            if status is not None
                            else content
                        )
                    except Exception:
                        brief = content
                    line = f"[{ts_str}] ({et}) {reason} | {brief}"
                    # 길이 과다 방지
                    if len(line) > 300:
                        line = line[:300]
                    lines.append(line)
                recent_reports_text = "\n".join(lines)
        except Exception:
            recent_reports_text = ""

        # 최신 리뷰 5개 수집(해당 심볼)
        trade_reviews_text = _format_trade_reviews_for_prompt(store, contract_symbol)

        # 포지션이 존재하면, 최근 오픈 이후 기록도 수집
        since_open_text = ""
        try:
            # 최근 오픈 기록 기준으로 저널을 다시 끌어오기 위해 트레이드 DB 조회
            df_trades = store.load_trades()
            if not df_trades.empty:
                df_sym = df_trades[(df_trades["symbol"] == contract_symbol)]
                df_opened = df_sym[df_sym["status"] == "opened"]
                if not df_opened.empty:
                    last_open_ts = df_opened["ts"].max()
                    journals_since_df = store.fetch_journals(
                        symbol=contract_symbol,
                        today_only=True,
                        since_ts=last_open_ts,
                        limit=50,
                        ascending=True,
                    )
                    lines = []
                    if not journals_since_df.empty:
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

        now_utc = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
        prompt = (
            "You are a brilliant cryptocurrency trader.\n"
            "You can freely choose between short, long, or holding positions.\n"
            "You always trade rationally and never emotionally.\n"
            "You analyze the current situation and make the optimal trading decision at all times.\n"
            "You typically use high leverage between 5x and 50x.\n"
            "Decide TP and SL values considering leverage.\n"
            "If the SL (stop loss) value implies a loss over 100%, it will be limited to 80%.\n"
            "It's important to not just always watch, sometimes take quick profits or cut losses to maximize returns.\n"
            f"Current UTC time: {now_utc}\n"
            f"The following is OHLCV CSV data for {spot_symbol}.\n"
            "[CSV_4h]\n" + csv_4h + "\n"
            "[CSV_1h]\n" + csv_1h + "\n"
            "[CSV_15m]\n" + csv_15m + "\n"
            f"Current price: {current_price}\n"
            + (
                "[CURRENT_POSITIONS]\n" + "\n".join(current_positions_lines) + "\n"
                if current_positions_lines
                else "Current position: None\n"
            )
            + (
                "[RECENT_REPORTS_10]\n" + recent_reports_text + "\n"
                if recent_reports_text
                else ""
            )
            + (
                "[TRADE_REVIEWS_5]\n" + trade_reviews_text + "\n"
                if trade_reviews_text
                else ""
            )
            + (
                "[JOURNALS_TODAY]\n" + journal_today_text + "\n"
                if journal_today_text
                else ""
            )
            + (
                "[SINCE_LAST_OPEN]\n" + since_open_text + "\n"
                if pos_side is not None and since_open_text
                else ""
            )
            + "Choose one of: watch/hold, short, long, or stop.\n"
            + "Return your decision as JSON with the following fields: order type (market/limit), price, stop loss (sl), take profit (tp), buy_now (boolean), leverage (number).\n"
            + "If you want to immediately take profit or cut loss on an existing position, set close_now=true and optionally close_percent (1~100).\n"
            + "When close_now is true, we will reduceOnly market-close the current position(s) without opening a new one in this cycle.\n"
            + f"Per-symbol max allocation is {per_symbol_alloc_pct:.2f}% of account equity (for your reference).\n"
            + "Include your reasoning for the decision in the 'explain' field. Output JSON, Korean only."
        )
        # 우선 JSON 구조화 응답 시도(도구호출/response_format)
        try:
            value = ai_provider.decide_json(prompt)
            logging.info(
                json.dumps(
                    {
                        "event": "llm_response_parsed",
                        "provider": os.getenv("AI_PROVIDER", "gemini").lower(),
                        "parsed": value,
                    },
                    ensure_ascii=False,
                )
            )
        except Exception:
            # 폴백: 일반 텍스트 응답 후 파싱
            response = ai_provider.decide(prompt)
            try:
                logging.info(
                    json.dumps(
                        {
                            "event": "llm_response_raw",
                            "provider": os.getenv("AI_PROVIDER", "gemini").lower(),
                            "response": response,
                        },
                        ensure_ascii=False,
                    )
                )
            except Exception as _e:
                logging.error("LLM raw logging failed: %s", str(_e))
            parser = make_to_object()
            value = parser.make_it_object(response)
            try:
                logging.info(
                    json.dumps(
                        {
                            "event": "llm_response_parsed",
                            "provider": os.getenv("AI_PROVIDER", "gemini").lower(),
                            "parsed": value,
                        },
                        ensure_ascii=False,
                    )
                )
            except Exception as _e:
                logging.error("LLM parsed logging failed: %s", str(_e))

        # close_now 우선 처리: 기존 포지션 즉시 익절/손절(전체/부분)
        if bool(value.get("close_now")):
            try:
                percent = float(value.get("close_percent") or 100.0)
            except Exception:
                percent = 100.0
            percent = 100.0 if percent <= 0 else (percent if percent <= 100 else 100.0)
            positions = bybit.get_positions_by_symbol(contract_symbol) or []
            if positions:
                try:
                    if percent >= 99.999:
                        res = bybit.close_symbol_positions(contract_symbol)
                    else:
                        res = bybit.reduce_symbol_positions_percent(
                            contract_symbol, percent
                        )
                    # 간단 실현손익 추정 기록
                    try:
                        last = bybit.get_last_price(contract_symbol) or current_price
                        for p in positions:
                            entry = float(
                                p.get("entryPrice")
                                or p.get("info", {}).get("avgPrice")
                                or 0
                            )
                            amount = float(
                                p.get("contracts")
                                or p.get("amount")
                                or p.get("size")
                                or 0
                            )
                            sidep = p.get("side") or p.get("info", {}).get("side")
                            if amount <= 0 or entry <= 0 or not sidep:
                                continue
                            closed_qty = (
                                amount
                                if percent >= 99.999
                                else amount * (percent / 100.0)
                            )
                            pnl = (
                                (last - entry) * closed_qty
                                if sidep == "long"
                                else (entry - last) * closed_qty
                            )
                            store.record_trade(
                                {
                                    "ts": datetime.utcnow(),
                                    "symbol": contract_symbol,
                                    "side": sidep,
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
                        store.record_journal(
                            {
                                "symbol": contract_symbol,
                                "entry_type": "action",
                                "content": f"close_now percent={float(percent)}",
                                "reason": value.get("explain"),
                                "meta": {"result": str(res)[:500]},
                            }
                        )
                    except Exception:
                        pass
                except Exception as e:
                    logging.error("close_now failed: %s", str(e))
            else:
                logging.info("close_now ignored: no active position for symbol")
            return

        # 의사결정/기록 + 트레이딩 실행
        ai_status = str(value.get("Status") or value.get("status") or "").lower()
        if ai_status in ["long", "short"]:
            # 내부 CCXT 주문은 buy/sell을 사용하므로 매핑
            side = "buy" if ai_status == "long" else "sell"
            if value.get("buy_now") is True:
                typevalue = "market"
            else:
                typevalue = "limit"

            # 리스크 기반 수량 계산
            balance_info = bybit.get_balance("USDT") or {}
            balance_total = balance_info.get("total") or 0
            # 리스크는 고정(코드 내부), 최대 배분은 심볼 균등(환경변수로 오버라이드 가능)
            risk_percent = 20.0
            max_alloc = float(os.getenv("MAX_ALLOC_PERCENT", str(per_symbol_alloc_pct)))
            # AI가 제안한 레버리지 우선 사용, 없으면 기본값
            leverage = float(
                value.get("leverage") or os.getenv("DEFAULT_LEVERAGE", "5")
            )
            min_qty = float(os.getenv("MIN_QTY", "1"))

            entry_price = (
                current_price
                if typevalue == "market"
                else value.get("price") or current_price
            )
            # 1) TP/SL 확인 루프: AI가 제안한 TP/SL이 있으면 수익률/손실률 알려주고 확정받기
            orig_tp = value.get("tp")
            orig_sl = value.get("sl")
            use_tp = orig_tp
            use_sl = orig_sl
            if (
                isinstance(orig_tp, (int, float))
                and isinstance(orig_sl, (int, float))
                and float(orig_tp) > 0
                and float(orig_sl) > 0
            ):
                e = float(entry_price)
                tp_v = float(orig_tp)
                sl_v = float(orig_sl)
                if ai_status == "long":
                    tp_pct = (tp_v - e) / e * 100.0
                    sl_pct = (e - sl_v) / e * 100.0
                else:
                    tp_pct = (e - tp_v) / e * 100.0
                    sl_pct = (sl_v - e) / e * 100.0
                confirm_prompt = (
                    "당신이 제안한 주문 파라미터를 최종 확인하세요. JSON만 응답. 한국어로.\n"
                    f"심볼: {contract_symbol}\n"
                    f"포지션: {ai_status} (내부 side={side})\n"
                    f"진입가(entry): {float(e)}\n"
                    f"TP: {float(tp_v)} (예상 수익률: {tp_pct:.4f}%)\n"
                    f"SL: {float(sl_v)} (예상 손실률: {sl_pct:.4f}%)\n"
                    f"레버리지: {float(leverage)}x\n"
                    "필수: confirm(boolean). 선택: tp, sl, price, buy_now, leverage, explain.\n"
                    "확신하면 confirm=true. 수정이 필요하면 값을 조정해 응답하세요."
                )
                try:
                    confirm = ai_provider.confirm_trade_json(confirm_prompt)
                    logging.info(
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
                        # 주문 중단 기록
                        try:
                            store.record_journal(
                                {
                                    "symbol": contract_symbol,
                                    "entry_type": "decision",
                                    "content": json.dumps(
                                        {
                                            "status": "skip_after_confirm",
                                            "ai_status": ai_status,
                                        },
                                        ensure_ascii=False,
                                    ),
                                    "reason": (
                                        confirm.get("explain") or value.get("explain")
                                    ),
                                    "meta": {"first": value, "confirm": confirm},
                                }
                            )
                        except Exception:
                            pass
                        return
                    # 확정: 조정값 반영
                    use_tp = (
                        float(confirm.get("tp"))
                        if confirm.get("tp") is not None
                        else use_tp
                    )
                    use_sl = (
                        float(confirm.get("sl"))
                        if confirm.get("sl") is not None
                        else use_sl
                    )
                    if confirm.get("price") is not None:
                        entry_price = float(confirm.get("price"))
                    if confirm.get("buy_now") is not None:
                        typevalue = (
                            "market" if bool(confirm.get("buy_now")) else "limit"
                        )
                    if confirm.get("leverage") is not None:
                        leverage = float(confirm.get("leverage"))
                except Exception as _e:
                    logging.error("AI confirm failed: %s", str(_e))

            # 2) SL 강제 제한(손실 100% 초과 시 80%로 제한)
            try:
                if isinstance(use_sl, (int, float)) and float(use_sl) > 0:
                    use_sl = enforce_max_loss_sl(
                        entry_price=float(entry_price),
                        proposed_sl=float(use_sl),
                        position=ai_status,
                        max_loss_percent=80.0,
                    )
            except Exception:
                pass

            # 3) 수량 재계산(확정/보정된 SL 반영)
            stop_price = use_sl or (entry_price * (0.99 if side == "buy" else 1.01))
            quantity = calculate_position_size(
                balance_usdt=float(balance_total or 0),
                entry_price=float(entry_price),
                stop_price=float(stop_price),
                risk_percent=risk_percent,
                max_allocation_percent=max_alloc,
                leverage=leverage,
                min_quantity=min_qty,
            )

            # 3-1) 심볼별 누적 노출 상한 적용: 이미 보유 중인 해당 심볼 노출을 고려해 남은 여유치만 진입
            try:
                positions_same_symbol = current_position or []
                # 현재가 기준 노출 계산(보수적): |수량| * 현재가
                last_price_for_notional = float(current_price or entry_price)
                if last_price_for_notional <= 0:
                    last_price_for_notional = float(entry_price)
                existing_notional = 0.0
                for p in positions_same_symbol:
                    try:
                        amount_raw = (
                            p.get("contracts") or p.get("amount") or p.get("size")
                        )
                        if amount_raw is None:
                            continue
                        amount = abs(float(amount_raw))
                        existing_notional += amount * last_price_for_notional
                    except Exception:
                        continue
                # 이 심볼의 최대 노출 한도(리버리지 반영): balance * (max_alloc/100) * leverage
                max_notional_for_symbol = (
                    float(balance_total or 0)
                    * (float(max_alloc) / 100.0)
                    * max(1.0, float(leverage))
                )
                remaining_notional = max(
                    0.0, max_notional_for_symbol - existing_notional
                )
                if remaining_notional <= 0:
                    # 상한 도달: 주문 스킵
                    try:
                        store.record_journal(
                            {
                                "symbol": contract_symbol,
                                "entry_type": "decision",
                                "content": json.dumps(
                                    {
                                        "status": "skip",
                                        "reason": "per_symbol_cap_reached",
                                    },
                                    ensure_ascii=False,
                                ),
                                "reason": value.get("explain"),
                                "meta": {
                                    "existing_notional": float(existing_notional),
                                    "max_notional_for_symbol": float(
                                        max_notional_for_symbol
                                    ),
                                },
                            }
                        )
                    except Exception:
                        pass
                    return
                # 남은 여유치로 제한된 최대 수량
                max_qty_by_remaining = (
                    remaining_notional / float(entry_price)
                    if float(entry_price) > 0
                    else quantity
                )
                if max_qty_by_remaining <= 0:
                    try:
                        store.record_journal(
                            {
                                "symbol": contract_symbol,
                                "entry_type": "decision",
                                "content": json.dumps(
                                    {
                                        "status": "skip",
                                        "reason": "no_remaining_capacity",
                                    },
                                    ensure_ascii=False,
                                ),
                                "reason": value.get("explain"),
                                "meta": {
                                    "existing_notional": float(existing_notional),
                                    "max_notional_for_symbol": float(
                                        max_notional_for_symbol
                                    ),
                                },
                            }
                        )
                    except Exception:
                        pass
                    return
                # 최종 수량을 여유치 기준으로 클램프(최소 수량 보장)
                quantity = max(
                    min_qty, min(float(quantity), float(max_qty_by_remaining))
                )
            except Exception:
                # 실패 시 기존 계산값으로 진행(안전)
                pass

            position_params = Open_Position(
                symbol=contract_symbol,
                type=typevalue,
                price=value.get("price") or float(entry_price),
                side=side,
                tp=use_tp,
                sl=use_sl,
                quantity=quantity,
                leverage=leverage,
            )
            order = bybit.open_position(position_params)
            if order is None:
                logging.error("Failed to open position (order is None)")
                try:
                    store.record_journal(
                        {
                            "symbol": contract_symbol,
                            "entry_type": "action",
                            "content": f"open_failed {side} {typevalue} price={float(entry_price)} qty={float(quantity)}",
                            "reason": value.get("explain") or "order rejected",
                            "meta": {"leverage": leverage},
                        }
                    )
                except Exception:
                    pass
                return
            logging.info(f"Position opened: {position_params}")

            try:
                order_id = None
                if isinstance(order, dict):
                    order_id = order.get("id") or order.get("info", {}).get("orderId")
                store.record_trade(
                    {
                        "ts": datetime.utcnow(),
                        "symbol": position_params.symbol,
                        "side": position_params.side,
                        "type": position_params.type,
                        "price": float(entry_price),
                        "quantity": float(quantity),
                        "tp": position_params.tp,
                        "sl": position_params.sl,
                        "leverage": leverage,
                        "status": "opened",
                        "order_id": order_id,
                        "pnl": None,
                    }
                )
            except Exception as e:
                logging.error(f"Trade store write failed: {str(e)}")

            # 기록: decision + action
            try:
                store.record_journal(
                    {
                        "symbol": contract_symbol,
                        "entry_type": "decision",
                        "content": json.dumps(
                            {
                                "status": side,
                                "ai_status": ai_status,
                                "type": typevalue,
                                "price": float(entry_price),
                                "tp": use_tp,
                                "sl": use_sl,
                                "leverage": leverage,
                            },
                            ensure_ascii=False,
                        ),
                        "reason": value.get("explain"),
                        "meta": value,
                    }
                )
                store.record_journal(
                    {
                        "symbol": contract_symbol,
                        "entry_type": "action",
                        "content": f"open {side} {typevalue} price={float(entry_price)} qty={float(quantity)}",
                        "reason": value.get("explain"),
                        "meta": {"order_id": order_id},
                        "ref_order_id": order_id,
                    }
                )
            except Exception as _e:
                logging.error(f"Journal write failed: {_e}")

        elif ai_status == "hold":
            logging.info("No trading signal generated")
            try:
                store.record_journal(
                    {
                        "symbol": contract_symbol,
                        "entry_type": "decision",
                        "content": json.dumps({"status": "hold"}, ensure_ascii=False),
                        "reason": value.get("explain"),
                        "meta": value,
                    }
                )
            except Exception as _e:
                logging.error(f"Journal write failed: {_e}")

        elif ai_status == "stop":
            logging.info("stop position(close_all)")
            try:
                # 포지션별 대략적 실현 손익 기록 후 청산
                positions = bybit.get_positions_by_symbol(contract_symbol) or []
                last = bybit.get_last_price(contract_symbol) or current_price
                for p in positions:
                    try:
                        entry = float(
                            p.get("entryPrice")
                            or p.get("info", {}).get("avgPrice")
                            or 0
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
                        store.record_trade(
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
            except Exception as e:
                logging.error(f"PNL calc failed: {str(e)}")
            bybit.close_symbol_positions(contract_symbol)
            try:
                store.record_journal(
                    {
                        "symbol": contract_symbol,
                        "entry_type": "action",
                        "content": "close_all",
                        "reason": value.get("explain") or "stop signal",
                        "meta": value,
                    }
                )
            except Exception as _e:
                logging.error(f"Journal write failed: {_e}")

    except Exception as e:
        logging.error(f"Error in automation: {str(e)}")


def run_scheduler():
    # 서울 시간대 설정
    seoul_tz = pytz.timezone("Asia/Seoul")
    current_time = datetime.now(seoul_tz)
    logging.info(f"Scheduler started at {current_time}")

    # 매 30분 마다 실행: 설정된 모든 심볼을 순회
    def job():
        symbols = _parse_symbols()
        for s in symbols:
            try:
                logging.info(f"Run automation for {s}")
                automation_for_symbol(s)
            except Exception as e:
                logging.error(f"Automation error for {s}: {e}")

    schedule.every().hour.at(":30").do(job)

    # 손실 리뷰 작업: 5분마다 실행
    def review_job():
        try:
            store = TradeStore(
                StorageConfig(
                    mysql_url=os.getenv("MYSQL_URL"),
                    sqlite_path=os.getenv("SQLITE_PATH"),
                )
            )
            _review_losing_trades(store, since_minutes=600)
        except Exception as e:
            logging.error(f"Loss review job error: {e}")

    schedule.every(5).minutes.do(review_job)

    # 매 15분마다 실행
    # schedule.every(15).minutes.do(automation)

    # 초기 실행
    review_job()
    job()

    while True:
        try:
            schedule.run_pending()
            time.sleep(1)
        except Exception as e:
            logging.error(f"Scheduler error: {str(e)}")
            time.sleep(60)  # 오류 발생시 1분 대기


if __name__ == "__main__":
    run_scheduler()
