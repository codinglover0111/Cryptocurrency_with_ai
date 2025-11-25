# pylint: disable=broad-except
# ruff: noqa: E722, BLE001
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
import os
from typing import Any, Dict, Optional

import pytz
import schedule
from dotenv import load_dotenv

from app import setup_logging
from app.config import SCHEDULER_CONFIG, load_runtime_config
from app.core.symbols import parse_trading_symbols
from app.workflows.trading import automation_for_symbol, run_loss_review
from utils.storage import TradeStore, StorageConfig


try:
    CONFIG_REFRESH_SECONDS = max(5, int(os.getenv("SCHEDULER_REFRESH_SECONDS", "30")))
except Exception:
    CONFIG_REFRESH_SECONDS = 30


def _get_store() -> TradeStore:
    """스토어 인스턴스를 생성합니다."""
    return TradeStore(
        StorageConfig(
            mysql_url=os.getenv("MYSQL_URL"),
            sqlite_path=os.getenv("SQLITE_PATH"),
        )
    )


def _save_scheduler_state(
    last_automation_run: Optional[datetime] = None,
    last_review_run: Optional[datetime] = None,
    automation_minutes: Optional[int] = None,
    loss_review_minutes: Optional[int] = None,
    is_running: bool = True,
) -> None:
    """스케줄러 상태를 DB에 저장합니다."""
    try:
        store = _get_store()

        if last_automation_run is not None:
            store.set_scheduler_state(
                "last_automation_run", last_automation_run.isoformat()
            )
        if last_review_run is not None:
            store.set_scheduler_state("last_review_run", last_review_run.isoformat())
        if automation_minutes is not None:
            store.set_scheduler_state("automation_minutes", str(automation_minutes))
        if loss_review_minutes is not None:
            store.set_scheduler_state("loss_review_minutes", str(loss_review_minutes))
        store.set_scheduler_state("is_running", "1" if is_running else "0")
    except Exception as e:
        logging.warning("스케줄러 상태 저장 실패: %s", e)


def load_scheduler_state() -> Dict[str, Any]:
    """스케줄러 상태를 DB에서 읽어옵니다. 외부에서 호출 가능."""
    try:
        store = _get_store()
        states = store.get_all_scheduler_states()

        result = {}
        for key, data in states.items():
            value = data.get("value")
            if key == "is_running":
                result[key] = value == "1"
            elif key in ("automation_minutes", "loss_review_minutes"):
                try:
                    result[key] = int(value) if value else None
                except (TypeError, ValueError):
                    result[key] = None
            else:
                result[key] = value

            # updated_at은 가장 최근 것으로 유지
            if data.get("updated_at"):
                result["updated_at"] = data["updated_at"]

        return result
    except Exception:
        pass
    return {}


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


def _is_scheduler_paused() -> bool:
    """스케줄러가 일시 중단 상태인지 확인합니다."""
    try:
        store = _get_store()
        paused = store.get_scheduler_state("paused")
        return paused == "1"
    except Exception:
        return False


def _sort_symbols_btc_first(symbols: list) -> list:
    """BTCUSDT를 첫 번째로 정렬합니다."""
    btc_symbols = [s for s in symbols if s.upper().startswith("BTC")]
    other_symbols = [s for s in symbols if not s.upper().startswith("BTC")]
    return btc_symbols + other_symbols


def run_scheduler() -> None:
    load_dotenv()
    _ensure_logging()

    seoul_tz = pytz.timezone("Asia/Seoul")
    current_time = datetime.now(seoul_tz)
    logging.info("Scheduler started at %s", current_time)

    def job() -> None:
        # 일시 중단 상태 확인
        if _is_scheduler_paused():
            logging.info("Scheduler is paused, skipping automation")
            return

        symbols = parse_trading_symbols()
        # BTC를 먼저 분석하도록 정렬
        symbols = _sort_symbols_btc_first(symbols)
        logging.info("Symbol order (BTC first): %s", symbols)

        _save_scheduler_state(last_automation_run=datetime.now(timezone.utc))
        for symbol in symbols:
            try:
                # 각 심볼 실행 전 일시 중단 상태 재확인
                if _is_scheduler_paused():
                    logging.info("Scheduler paused during execution, stopping")
                    break
                logging.info("Run automation for %s", symbol)
                automation_for_symbol(symbol, symbols=symbols)
            except Exception:
                logging.exception("Automation error for %s", symbol)

    def review_job() -> None:
        # 일시 중단 상태 확인
        if _is_scheduler_paused():
            logging.info("Scheduler is paused, skipping loss review")
            return

        try:
            logging.info("Run loss review job")
            symbols = parse_trading_symbols()
            _save_scheduler_state(last_review_run=datetime.now(timezone.utc))
            run_loss_review(symbols=symbols)

        except Exception:
            logging.exception("Loss review job error")

    scheduler_obj = schedule.Scheduler()
    scheduler_config = _load_scheduler_settings()
    _apply_scheduler_config(scheduler_obj, scheduler_config, job, review_job)
    last_reload = time.monotonic()

    # 스케줄러 시작 시 상태 저장
    _save_scheduler_state(
        automation_minutes=scheduler_config["automation_minutes"],
        loss_review_minutes=scheduler_config["loss_review_minutes"],
        is_running=True,
    )

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
                    _save_scheduler_state(
                        automation_minutes=scheduler_config["automation_minutes"],
                        loss_review_minutes=scheduler_config["loss_review_minutes"],
                    )
                last_reload = now
            time.sleep(1)
        except Exception:
            logging.exception("Scheduler error")
            time.sleep(60)


if __name__ == "__main__":
    run_scheduler()
