# pylint: disable=broad-except
# ruff: noqa: E722, BLE001
from __future__ import annotations

import logging
import time
from datetime import datetime
import os
from typing import Any, Dict

import pytz
import schedule
from dotenv import load_dotenv

from app import setup_logging
from app.config import SCHEDULER_CONFIG, load_runtime_config
from app.core.symbols import parse_trading_symbols
from app.workflows.trading import automation_for_symbol, run_loss_review


try:
    CONFIG_REFRESH_SECONDS = max(5, int(os.getenv("SCHEDULER_REFRESH_SECONDS", "30")))
except Exception:
    CONFIG_REFRESH_SECONDS = 30


def _ensure_logging() -> None:
    if not logging.getLogger().handlers:
        setup_logging()


def _normalize_minutes(value: Any, *, default: int, minimum: int, maximum: int) -> int:
    try:
        minutes = int(value)
    except (TypeError, ValueError):
        return default
    if minutes <= 0:
        return default
    return max(minimum, min(minutes, maximum))


def _load_scheduler_settings() -> Dict[str, Any]:
    runtime = load_runtime_config()
    raw = runtime.get("scheduler")
    config = dict(SCHEDULER_CONFIG)
    if isinstance(raw, dict):
        config.update(raw)
    return {
        "automation_minutes": _normalize_minutes(
            config.get("automation_minutes"),
            default=SCHEDULER_CONFIG["automation_minutes"],
            minimum=1,
            maximum=180,
        ),
        "loss_review_minutes": _normalize_minutes(
            config.get("loss_review_minutes"),
            default=SCHEDULER_CONFIG["loss_review_minutes"],
            minimum=1,
            maximum=720,
        ),
        "cold_start": bool(config.get("cold_start")),
    }


def _apply_scheduler_config(
    scheduler_obj: schedule.Scheduler,
    config: Dict[str, Any],
    job_callable,
    review_callable,
) -> None:
    scheduler_obj.clear()
    scheduler_obj.every(config["automation_minutes"]).minutes.do(job_callable)
    scheduler_obj.every(config["loss_review_minutes"]).minutes.do(review_callable)
    logging.info(
        "스케줄러 주기 적용: automation=%s분, loss_review=%s분",
        config["automation_minutes"],
        config["loss_review_minutes"],
    )


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

    scheduler_obj = schedule.Scheduler()
    scheduler_config = _load_scheduler_settings()
    _apply_scheduler_config(scheduler_obj, scheduler_config, job, review_job)
    last_reload = time.monotonic()

    cold_start_flag = scheduler_config["cold_start"] or os.getenv("COLD_START") == "1"
    if cold_start_flag:
        job()
        review_job()

    while True:
        try:
            scheduler_obj.run_pending()
            now = time.monotonic()
            if now - last_reload >= CONFIG_REFRESH_SECONDS:
                latest = _load_scheduler_settings()
                if latest != scheduler_config:
                    logging.info("스케줄러 설정 변경 감지: %s", latest)
                    scheduler_config = latest
                    _apply_scheduler_config(
                        scheduler_obj, scheduler_config, job, review_job
                    )
                last_reload = now
            time.sleep(1)
        except Exception:
            logging.exception("Scheduler error")
            time.sleep(60)


if __name__ == "__main__":
    run_scheduler()
