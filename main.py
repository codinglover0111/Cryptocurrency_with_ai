# pylint: disable=broad-except
# ruff: noqa: E722, BLE001
from __future__ import annotations

import logging
import time
from datetime import datetime
import os

import pytz
import schedule
from dotenv import load_dotenv

from app import setup_logging
from app.core.symbols import parse_trading_symbols
from app.workflows.trading import automation_for_symbol, run_loss_review


def _ensure_logging() -> None:
    if not logging.getLogger().handlers:
        setup_logging()


def run_scheduler() -> None:
    load_dotenv()
    _ensure_logging()

    seoul_tz = pytz.timezone("Asia/Seoul")
    current_time = datetime.now(seoul_tz)
    logging.info("Scheduler started at %s", current_time)

    def job() -> None:
        symbols = parse_trading_symbols()
        for symbol in symbols:
            try:
                logging.info("Run automation for %s", symbol)
                automation_for_symbol(symbol, symbols=symbols)
            except Exception:
                logging.exception("Automation error for %s", symbol)

    def review_job() -> None:
        try:
            logging.info("Run loss review job")
            symbols = parse_trading_symbols()
            run_loss_review(symbols=symbols)

        except Exception:
            logging.exception("Loss review job error")

    # 매시각 58분 13분 28분 43분에 실행
    schedule.every().hour.at(":58").do(job)
    schedule.every().hour.at(":13").do(job)
    schedule.every().hour.at(":28").do(job)
    schedule.every().hour.at(":43").do(job)
    # 5분 주기로 실행
    schedule.every(5).minutes.do(review_job)

    if os.getenv("COLD_START") == "1":
        job()
        review_job()

    while True:
        try:
            schedule.run_pending()
            time.sleep(1)
        except Exception:
            logging.exception("Scheduler error")
            time.sleep(60)


if __name__ == "__main__":
    run_scheduler()
