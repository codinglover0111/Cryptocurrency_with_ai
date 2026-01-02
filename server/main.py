# pylint: disable=broad-except
# ruff: noqa: E722, BLE001
from __future__ import annotations

import logging
import os
import socket
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

import pytz
import schedule
from dotenv import load_dotenv

from app import setup_logging
from app.config import SCHEDULER_CONFIG, load_runtime_config
from app.core.symbols import parse_trading_symbols
from app.workflows.trading import automation_for_symbol, run_loss_review
from app.services import supabase_repo
from utils.storage import TradeStore, StorageConfig


try:
    CONFIG_REFRESH_SECONDS = max(5, int(os.getenv("SCHEDULER_REFRESH_SECONDS", "30")))
except Exception:
    CONFIG_REFRESH_SECONDS = 30


VT_SECONDS = 3600
MAX_RETRY = 3
POLL_INTERVAL_SECONDS = 5
LEASE_DURATION_SECONDS = 3600


def _should_use_queue() -> bool:
    """Supabase Queues(PGMQ)를 사용할지 여부를 판단합니다."""

    if os.getenv("USE_LEGACY_EXECUTION") == "1":
        return False

    return bool(supabase_repo.get_client())


def _bot_instance_id() -> str:
    """런너/스케줄러 인스턴스 식별자를 생성합니다."""

    host = "unknown"
    try:
        host = socket.gethostname()
    except Exception:
        host = "unknown"

    return f"{host}:{os.getpid()}:{uuid.uuid4().hex[:8]}"


def _is_run_lock_active(lock: Optional[Dict[str, Any]]) -> bool:
    """전역 락이 현재 유효하게 'running' 상태인지 판별합니다."""

    if not lock or not isinstance(lock, dict):
        return False

    if str(lock.get("status") or "").lower() != "running":
        return False

    # lease_expires_at이 없으면(구버전/초기 상태) running을 그대로 신뢰한다.
    lease_expires_at = lock.get("lease_expires_at")
    if not lease_expires_at:
        return True

    try:
        lease_dt = datetime.fromisoformat(str(lease_expires_at).replace("Z", "+00:00"))
        return lease_dt > datetime.now(timezone.utc)
    except Exception:
        return True


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
    """스케줄러가 일시 중단 상태인지 확인합니다.

    - Supabase가 설정되어 있으면 Supabase 상태를 우선합니다.
    - Supabase가 없거나 실패하면 로컬 DB 상태로 폴백합니다.
    """

    try:
        states = supabase_repo.get_scheduler_state_all()
        paused_row = states.get("paused") if isinstance(states, dict) else None
        if isinstance(paused_row, dict) and paused_row.get("value") is not None:
            return str(paused_row.get("value")) == "1"
    except Exception:
        pass

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
        """스케줄러 자동 실행(job)을 수행합니다.

        - 큐 모드: Supabase Queues에 enqueue만 수행합니다.
        - 레거시 모드: 기존처럼 즉시 실행합니다.
        """

        # 일시 중단 상태 확인
        if _is_scheduler_paused():
            logging.info("Scheduler is paused, skipping automation")
            return

        symbols = parse_trading_symbols()
        # BTC를 먼저 분석하도록 정렬
        symbols = _sort_symbols_btc_first(symbols)
        logging.info("Symbol order (BTC first): %s", symbols)

        _save_scheduler_state(last_automation_run=datetime.now(timezone.utc))

        # 큐 모드: 실행하지 않고 enqueue만 수행
        if _should_use_queue():
            lock = supabase_repo.get_run_lock()
            if _is_run_lock_active(lock):
                # 타입 체커/런타임 방어를 위해 dict 여부를 한 번 더 확인한다.
                lock_run_id = lock.get("run_id") if isinstance(lock, dict) else None
                logging.info(
                    "Scheduler enqueue skipped: already running (run_id=%s)",
                    lock_run_id,
                )
                return

            dedupe_key = "automation_all"
            dedupe_row = supabase_repo.get_job_dedupe(dedupe_key)
            if dedupe_row and str(dedupe_row.get("status") or "").lower() == "queued":
                logging.info(
                    "Scheduler enqueue skipped: already queued (run_id=%s)",
                    dedupe_row.get("run_id"),
                )
                return

            run_id = str(uuid.uuid4())
            requested_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            message = {
                "job_type": "automation_all",
                "dedupe_key": dedupe_key,
                "symbols": symbols,
                "requested_by": "scheduler",
                "requested_at": requested_at,
                "run_id": run_id,
            }

            msg_id = supabase_repo.pgmq_send(
                supabase_repo.QUEUE_SCHEDULED, message, sleep_seconds=0
            )
            if msg_id is None:
                logging.error(
                    "Scheduler enqueue failed: queue=%s", supabase_repo.QUEUE_SCHEDULED
                )
                return

            supabase_repo.upsert_job_dedupe(
                dedupe_key=dedupe_key,
                queue_name=supabase_repo.QUEUE_SCHEDULED,
                msg_id=msg_id,
                status="queued",
                run_id=run_id,
                requested_by="scheduler",
                requested_at=requested_at,
            )

            logging.info(
                "Scheduler queued: queue=%s run_id=%s msg_id=%s symbols=%s",
                supabase_repo.QUEUE_SCHEDULED,
                run_id,
                msg_id,
                len(symbols),
            )
            return

        # 레거시 모드: 기존처럼 즉시 실행
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
        """손실 리뷰 작업(review_job)을 수행합니다.

        - 큐 모드: Supabase Queues에 enqueue만 수행합니다.
        - 레거시 모드: 기존처럼 즉시 실행합니다.
        """

        # 일시 중단 상태 확인
        if _is_scheduler_paused():
            logging.info("Scheduler is paused, skipping loss review")
            return

        try:
            logging.info("Run loss review job")
            symbols = parse_trading_symbols()
            _save_scheduler_state(last_review_run=datetime.now(timezone.utc))

            # 큐 모드: 실행하지 않고 enqueue만 수행
            if _should_use_queue():
                lock = supabase_repo.get_run_lock()
                if _is_run_lock_active(lock):
                    # 타입 체커/런타임 방어를 위해 dict 여부를 한 번 더 확인한다.
                    lock_run_id = lock.get("run_id") if isinstance(lock, dict) else None
                    logging.info(
                        "Review enqueue skipped: already running (run_id=%s)",
                        lock_run_id,
                    )
                    return

                dedupe_key = "loss_review"
                dedupe_row = supabase_repo.get_job_dedupe(dedupe_key)
                if (
                    dedupe_row
                    and str(dedupe_row.get("status") or "").lower() == "queued"
                ):
                    logging.info(
                        "Review enqueue skipped: already queued (run_id=%s)",
                        dedupe_row.get("run_id"),
                    )
                    return

                run_id = str(uuid.uuid4())
                requested_at = (
                    datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
                )
                message = {
                    "job_type": "loss_review",
                    "dedupe_key": dedupe_key,
                    "symbols": symbols,
                    "requested_by": "scheduler",
                    "requested_at": requested_at,
                    "run_id": run_id,
                }

                msg_id = supabase_repo.pgmq_send(
                    supabase_repo.QUEUE_REVIEW, message, sleep_seconds=0
                )
                if msg_id is None:
                    logging.error(
                        "Review enqueue failed: queue=%s", supabase_repo.QUEUE_REVIEW
                    )
                    return

                supabase_repo.upsert_job_dedupe(
                    dedupe_key=dedupe_key,
                    queue_name=supabase_repo.QUEUE_REVIEW,
                    msg_id=msg_id,
                    status="queued",
                    run_id=run_id,
                    requested_by="scheduler",
                    requested_at=requested_at,
                )

                logging.info(
                    "Review queued: queue=%s run_id=%s msg_id=%s",
                    supabase_repo.QUEUE_REVIEW,
                    run_id,
                    msg_id,
                )
                return

            # 레거시 모드: 기존처럼 즉시 실행
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


def _safe_int(value: Any, *, default: int = 0) -> int:
    """값을 int로 안전하게 변환합니다."""

    try:
        return int(value)
    except Exception:
        return int(default)


def _record_runner_failure_journal(
    store: TradeStore,
    *,
    queue_name: str,
    msg_id: int,
    job_type: str,
    dedupe_key: str,
    run_id: str,
    error: str,
    payload: Dict[str, Any],
) -> None:
    """런너 실행 실패를 저널(로컬 DB/스토리지)에 기록합니다."""

    try:
        symbol = payload.get("symbol")
        symbol = str(symbol).strip().upper() if symbol else None

        store.record_journal(
            {
                "symbol": symbol,
                "entry_type": "error",
                "content": (
                    f"runner_job_failed job_type={job_type} queue={queue_name} msg_id={msg_id} "
                    f"run_id={run_id} dedupe_key={dedupe_key}"
                ),
                "reason": error[:500],
                "meta": {
                    "job_type": job_type,
                    "queue": queue_name,
                    "msg_id": msg_id,
                    "run_id": run_id,
                    "dedupe_key": dedupe_key,
                    "payload": payload,
                },
            }
        )
    except Exception:
        logging.exception("Runner failure journal write error")


def _execute_job_from_message(job_type: str, payload: Dict[str, Any]) -> None:
    """큐 메시지(job_type)에 따라 실제 실행을 수행합니다."""

    if job_type == "automation_all":
        raw_symbols = payload.get("symbols")
        if isinstance(raw_symbols, list):
            symbols = [str(s).strip().upper() for s in raw_symbols if str(s).strip()]
        else:
            symbols = parse_trading_symbols()

        symbols = _sort_symbols_btc_first(symbols)
        for symbol in symbols:
            if not symbol:
                continue
            logging.info("Runner executes automation_for_symbol: %s", symbol)
            automation_for_symbol(symbol, symbols=symbols)
        return

    if job_type == "automation_symbol":
        symbol = str(payload.get("symbol") or "").strip().upper()
        if not symbol:
            raise ValueError("symbol_missing")
        logging.info("Runner executes automation_for_symbol: %s", symbol)
        automation_for_symbol(symbol)
        return

    if job_type == "loss_review":
        raw_symbols = payload.get("symbols")
        if isinstance(raw_symbols, list):
            symbols = [str(s).strip().upper() for s in raw_symbols if str(s).strip()]
        else:
            symbols = parse_trading_symbols()
        logging.info("Runner executes run_loss_review")
        run_loss_review(symbols=symbols)
        return

    raise ValueError(f"unknown_job_type:{job_type}")


def _requeue_with_backoff(
    *,
    queue_name: str,
    msg_id: int,
    msg_payload: Dict[str, Any],
    read_ct: int,
) -> None:
    """락 미획득 시 메시지를 백오프로 재적재합니다."""

    backoff_seconds = min(60, max(POLL_INTERVAL_SECONDS, (int(read_ct) + 1) * 5))

    # 메시지 유실을 막기 위해 '재적재 성공' 시에만 기존 메시지를 삭제한다.
    new_msg_id = supabase_repo.pgmq_send(
        queue_name, msg_payload, sleep_seconds=int(backoff_seconds)
    )
    if new_msg_id is None:
        logging.warning(
            "Re-enqueue failed; keep invisible until VT. queue=%s msg_id=%s",
            queue_name,
            msg_id,
        )
        return

    try:
        supabase_repo.pgmq_delete(queue_name, message_id=msg_id)
    except Exception:
        logging.exception("Failed to delete old message after re-enqueue")

    # 디듀프 테이블의 msg_id도 최신으로 맞춰둔다(베스트 에포트)
    try:
        dedupe_key = str(msg_payload.get("dedupe_key") or "").strip()
        run_id = str(msg_payload.get("run_id") or "").strip()
        requested_by = str(msg_payload.get("requested_by") or "runner").strip()
        requested_at = str(msg_payload.get("requested_at") or "").strip()

        if dedupe_key and run_id and requested_at:
            supabase_repo.upsert_job_dedupe(
                dedupe_key=dedupe_key,
                queue_name=queue_name,
                msg_id=int(new_msg_id),
                status="queued",
                run_id=run_id,
                requested_by=requested_by,
                requested_at=requested_at,
            )
    except Exception:
        pass


def _process_pgmq_message(
    *,
    queue_name: str,
    msg_row: Dict[str, Any],
    holder: str,
    store: TradeStore,
) -> None:
    """PGMQ 메시지 1건을 처리(락/디듀프/실행/재시도)합니다."""

    msg_id = _safe_int(msg_row.get("msg_id") or msg_row.get("message_id"))
    if msg_id <= 0:
        logging.warning("Invalid message id: %s", msg_row)
        return

    read_ct = _safe_int(msg_row.get("read_ct"), default=0)

    raw_payload = msg_row.get("message")
    msg_payload: Dict[str, Any] = raw_payload if isinstance(raw_payload, dict) else {}

    job_type = str(msg_payload.get("job_type") or "").strip()
    dedupe_key = str(msg_payload.get("dedupe_key") or "").strip()
    run_id = str(msg_payload.get("run_id") or "").strip()

    if not job_type:
        logging.warning(
            "Missing job_type; archive message. queue=%s msg_id=%s", queue_name, msg_id
        )
        try:
            supabase_repo.pgmq_archive(queue_name, message_id=msg_id)
        except Exception:
            pass
        return

    effective_run_id = run_id or str(uuid.uuid4())

    acquired = supabase_repo.try_acquire_run_lock(
        holder=holder,
        run_id=effective_run_id,
        lease_seconds=LEASE_DURATION_SECONDS,
    )
    if not acquired:
        _requeue_with_backoff(
            queue_name=queue_name,
            msg_id=msg_id,
            msg_payload=msg_payload,
            read_ct=read_ct,
        )
        return

    try:
        if dedupe_key:
            supabase_repo.update_job_dedupe_status(
                dedupe_key=dedupe_key,
                status="running",
                run_id=effective_run_id,
            )

        _execute_job_from_message(job_type, msg_payload)

        # 성공 처리: 메시지 삭제 + 디듀프 done
        try:
            supabase_repo.pgmq_delete(queue_name, message_id=msg_id)
        except Exception:
            logging.exception(
                "PGMQ delete failed: queue=%s msg_id=%s", queue_name, msg_id
            )

        if dedupe_key:
            supabase_repo.update_job_dedupe_status(
                dedupe_key=dedupe_key,
                status="done",
                run_id=effective_run_id,
            )

    except Exception as exc:
        logging.exception(
            "Runner job failed: queue=%s msg_id=%s job_type=%s read_ct=%s",
            queue_name,
            msg_id,
            job_type,
            read_ct,
        )

        if read_ct >= MAX_RETRY:
            # 최대 재시도 초과: archive + failed + 저널
            try:
                supabase_repo.pgmq_archive(queue_name, message_id=msg_id)
            except Exception:
                logging.exception(
                    "PGMQ archive failed: queue=%s msg_id=%s", queue_name, msg_id
                )

            if dedupe_key:
                supabase_repo.update_job_dedupe_status(
                    dedupe_key=dedupe_key,
                    status="failed",
                    run_id=effective_run_id,
                )

            _record_runner_failure_journal(
                store,
                queue_name=queue_name,
                msg_id=msg_id,
                job_type=job_type,
                dedupe_key=dedupe_key,
                run_id=effective_run_id,
                error=str(exc),
                payload=msg_payload,
            )

        # read_ct < MAX_RETRY: vt 만료 후 자동 재시도 (메시지를 남겨둔다)

    finally:
        try:
            supabase_repo.release_run_lock(holder=holder)
        except Exception:
            logging.exception("Failed to release run lock")


def run_runner() -> None:
    """PGMQ 큐를 폴링하며 잡을 소비/실행하는 Runner를 시작합니다."""

    load_dotenv()
    _ensure_logging()

    if not _should_use_queue():
        logging.error("Runner requires Supabase configured and USE_LEGACY_EXECUTION!=1")
        return

    holder = _bot_instance_id()
    store = _get_store()

    logging.info("Runner started: holder=%s", holder)

    while True:
        try:
            processed = False
            for queue_name in [
                supabase_repo.QUEUE_MANUAL,
                supabase_repo.QUEUE_SCHEDULED,
                supabase_repo.QUEUE_REVIEW,
            ]:
                msgs = supabase_repo.pgmq_read(queue_name, vt_seconds=VT_SECONDS, n=1)
                if not msgs:
                    continue

                processed = True
                msg_row = msgs[0]
                _process_pgmq_message(
                    queue_name=queue_name,
                    msg_row=msg_row,
                    holder=holder,
                    store=store,
                )
                break

            if not processed:
                time.sleep(POLL_INTERVAL_SECONDS)

        except Exception:
            logging.exception("Runner loop error")
            time.sleep(POLL_INTERVAL_SECONDS)


def run_bot() -> None:
    """BOT_MODE에 따라 스케줄러/런너를 실행합니다."""

    mode = str(os.getenv("BOT_MODE") or "").strip().lower()

    if mode == "runner":
        run_runner()
        return

    if mode == "both" or (not mode and _should_use_queue()):
        threading.Thread(target=run_runner, daemon=True).start()
        run_scheduler()
        return

    # 기본값: 기존과 동일하게 스케줄러만 실행
    run_scheduler()


if __name__ == "__main__":
    run_bot()
