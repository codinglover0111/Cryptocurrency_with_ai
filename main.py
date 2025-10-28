import os
import json
from utils import bybit_utils, BybitUtils, Open_Position, make_to_object
from utils.risk import calculate_position_size
from utils.storage import TradeStore, StorageConfig
from utils.ai_provider import AIProvider
from dotenv import load_dotenv
import schedule
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


def automation_for_symbol(symbol_usdt: str):
    try:
        is_testnet = bool(int(os.getenv("TESTNET", "1")))
        bybit = BybitUtils(is_testnet)
        store = TradeStore(StorageConfig(mysql_url=os.getenv("MYSQL_URL")))

        spot_symbol, contract_symbol = _to_ccxt_symbols(symbol_usdt)

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
        # TODO 비동기 최적화 가능
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
            pos_entry = pos.get("entryPrice") or pos.get("info", {}).get("avgPrice")
        else:
            pos_side, pos_entry = None, None

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

        # 포지션이 존재하면, 최근 오픈 이후 기록도 수집
        since_open_text = ""
        try:
            # trades.xlsx에서 해당 심볼의 마지막 opened ts 조회
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
            f"Current UTC time: {now_utc}\n"
            f"The following is OHLCV CSV data for {spot_symbol}.\n"
            "[CSV_4h]\n" + csv_4h + "\n"
            "[CSV_1h]\n" + csv_1h + "\n"
            "[CSV_15m]\n" + csv_15m + "\n"
            f"Current price: {current_price}\n"
            + (
                f"Current position: side={pos_side}, entry={pos_entry}\n"
                if pos_side is not None
                else "Current position: None\n"
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
            + "Choose one of: watch/hold, sell, buy, or stop.\n"
            + "Return your decision as JSON with the following fields: order type (market/limit), price, stop loss (sl), take profit (tp), buy_now (boolean), leverage (number).\n"
            + "Account risk is limited to 20% (for your reference).\n"
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

        # 의사결정/기록 + 트레이딩 실행
        if value["Status"] in ["buy", "sell"]:
            side = value["Status"]
            if value.get("buy_now") is True:
                typevalue = "market"
            else:
                typevalue = "limit"

            # 리스크 기반 수량 계산
            balance_info = bybit.get_balance("USDT") or {}
            balance_total = balance_info.get("total") or 0
            # 리스크는 무조건 20% 리미트
            risk_percent = 20.0
            max_alloc = float(os.getenv("MAX_ALLOC_PERCENT", "20.0"))
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
            stop_price = value.get("sl") or (
                entry_price * (0.99 if side == "buy" else 1.01)
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

            position_params = Open_Position(
                symbol=contract_symbol,
                type=typevalue,
                price=value.get("price") or float(entry_price),
                side=side,
                tp=value.get("tp"),
                sl=value.get("sl"),
                quantity=quantity,
                leverage=leverage,
            )
            order = bybit.open_position(position_params)
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
                                "type": typevalue,
                                "price": float(entry_price),
                                "tp": value.get("tp"),
                                "sl": value.get("sl"),
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

        elif value["Status"] == "hold":
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

        elif value["Status"] == "stop":
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

    # 매 15분마다 실행
    # schedule.every(15).minutes.do(automation)

    # 초기 실행
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
