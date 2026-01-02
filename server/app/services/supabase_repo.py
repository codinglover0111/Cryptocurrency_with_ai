"""Supabase PostgREST helpers using supabase-py.

This module is opt-in: it activates when SUPABASE_URL and
SUPABASE_SERVICE_ROLE_KEY are present. Otherwise callers should
fallback to the existing TradeStore implementation.
"""

from __future__ import annotations

import datetime as dt
import os
from functools import lru_cache
from typing import Any, Dict, List, Optional

from supabase import Client, create_client

# Tables we touch:
# - agent_prompts(agent_type text unique, prompt_template text, updated_at timestamptz)
# - scheduler_state(key text unique, value text, updated_at timestamptz)
# - shared_analysis(symbol text, analysis_type text, content text, created_at timestamptz)
# - runtime_config(section text, config_data text, updated_at timestamptz)


def _has_env() -> bool:
    return bool(os.getenv("SUPABASE_URL") and os.getenv("SUPABASE_SERVICE_ROLE_KEY"))


@lru_cache(maxsize=1)
def get_client() -> Optional[Client]:
    """Return a cached supabase client or None if not configured."""
    if not _has_env():
        return None
    return create_client(
        os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"]
    )


# ---------- agent_prompts ----------
def get_agent_prompt(agent_type: str) -> Optional[str]:
    client = get_client()
    if not client:
        return None
    res = (
        client.table("agent_prompts")
        .select("prompt_template")
        .eq("agent_type", agent_type)
        .maybe_single()
        .execute()
    )
    if res.data and res.data.get("prompt_template"):
        return str(res.data["prompt_template"])
    return None


def get_all_agent_prompts() -> Dict[str, Dict[str, Any]]:
    client = get_client()
    if not client:
        return {}
    res = client.table("agent_prompts").select("*").execute()
    prompts: Dict[str, Dict[str, Any]] = {}
    for row in res.data or []:
        agent_type = row.get("agent_type")
        if not agent_type:
            continue
        prompts[str(agent_type)] = row
    return prompts


def upsert_agent_prompt(agent_type: str, prompt_template: str) -> bool:
    client = get_client()
    if not client:
        return False
    now_iso = dt.datetime.utcnow().isoformat() + "Z"
    client.table("agent_prompts").upsert(
        {
            "agent_type": agent_type,
            "prompt_template": prompt_template,
            "updated_at": now_iso,
        }
    ).execute()
    return True


def upsert_agent_prompts_bulk(prompts: Dict[str, str]) -> bool:
    client = get_client()
    if not client or not prompts:
        return False
    now_iso = dt.datetime.utcnow().isoformat() + "Z"
    payload = [
        {"agent_type": k, "prompt_template": v, "updated_at": now_iso}
        for k, v in prompts.items()
    ]
    client.table("agent_prompts").upsert(payload).execute()
    return True


def delete_agent_prompt(agent_type: str) -> bool:
    client = get_client()
    if not client:
        return False
    client.table("agent_prompts").delete().eq("agent_type", agent_type).execute()
    return True


# ---------- scheduler_state ----------
def get_scheduler_state_all() -> Dict[str, Dict[str, Any]]:
    client = get_client()
    if not client:
        return {}
    res = client.table("scheduler_state").select("*").execute()
    states: Dict[str, Dict[str, Any]] = {}
    for row in res.data or []:
        key = row.get("key")
        if not key:
            continue
        states[str(key)] = row
    return states


def set_scheduler_state(key: str, value: str) -> bool:
    client = get_client()
    if not client:
        return False
    now_iso = dt.datetime.utcnow().isoformat() + "Z"
    client.table("scheduler_state").upsert(
        {"key": key, "value": value, "updated_at": now_iso}
    ).execute()
    return True


# ---------- shared_analysis ----------
def save_shared_analysis(
    symbol: str, analysis_type: str, content: str, created_at: Optional[str] = None
) -> bool:
    client = get_client()
    if not client:
        return False
    ts = created_at or dt.datetime.utcnow().isoformat() + "Z"
    client.table("shared_analysis").upsert(
        {
            "symbol": symbol,
            "analysis_type": analysis_type,
            "content": content,
            "created_at": ts,
        }
    ).execute()
    return True


def get_shared_analysis(
    symbol: str, analysis_type: Optional[str] = None
) -> List[Dict[str, Any]]:
    client = get_client()
    if not client:
        return []
    query = client.table("shared_analysis").select("*").eq("symbol", symbol)
    if analysis_type:
        query = query.eq("analysis_type", analysis_type)
    res = query.order("created_at", desc=True).execute()
    return res.data or []


# ---------- runtime_config ----------
def get_runtime_config(section: str) -> Optional[str]:
    client = get_client()
    if not client:
        return None
    res = (
        client.table("runtime_config")
        .select("config_data")
        .eq("section", section)
        .maybe_single()
        .execute()
    )
    if res.data and res.data.get("config_data") is not None:
        return str(res.data["config_data"])
    return None


def set_runtime_config(section: str, config_data: str) -> bool:
    client = get_client()
    if not client:
        return False
    now_iso = dt.datetime.utcnow().isoformat() + "Z"
    client.table("runtime_config").upsert(
        {"section": section, "config_data": config_data, "updated_at": now_iso}
    ).execute()
    return True


# ---------- pgmq_public (Queues) ----------
DEFAULT_PGMQ_SCHEMA = "pgmq_public"

QUEUE_MANUAL = "cca_jobs_manual"
QUEUE_SCHEDULED = "cca_jobs_scheduled"
QUEUE_REVIEW = "cca_jobs_review"

DEFAULT_RUN_LOCK_KEY = "automation_global"


def _utc_now() -> dt.datetime:
    """UTC 기준 현재 시간을 반환합니다."""

    return dt.datetime.now(dt.timezone.utc)


def _to_utc_iso(ts: dt.datetime) -> str:
    """datetime을 UTC ISO8601(Z) 문자열로 변환합니다."""

    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=dt.timezone.utc)
    else:
        ts = ts.astimezone(dt.timezone.utc)
    return ts.isoformat().replace("+00:00", "Z")


def pgmq_send(
    queue_name: str,
    message: Dict[str, Any],
    *,
    sleep_seconds: int = 0,
) -> Optional[int]:
    """PGMQ 큐에 메시지를 적재합니다."""

    client = get_client()
    if not client:
        return None

    payload = {
        "queue_name": queue_name,
        "message": message,
        "sleep_seconds": int(sleep_seconds or 0),
    }
    res = client.schema(DEFAULT_PGMQ_SCHEMA).rpc("send", payload).execute()

    # Supabase RPC 응답이 숫자 또는 리스트/딕셔너리로 올 수 있어 방어적으로 파싱한다.
    try:
        if isinstance(res.data, (int, float, str)):
            return int(res.data)
        if isinstance(res.data, list) and res.data:
            return int(res.data[0])
        if isinstance(res.data, dict):
            for key in ("id", "message_id", "msg_id"):
                if key in res.data:
                    return int(res.data[key])
    except Exception:
        return None

    return None


def pgmq_read(queue_name: str, *, vt_seconds: int, n: int = 1) -> List[Dict[str, Any]]:
    """PGMQ 큐에서 메시지를 읽어옵니다.

    - vt_seconds: visibility timeout(메시지 재노출까지의 시간)
    - n: 읽을 메시지 최대 개수
    """

    client = get_client()
    if not client:
        return []

    payload = {
        "queue_name": queue_name,
        "sleep_seconds": int(vt_seconds),
        "n": int(n),
    }
    res = client.schema(DEFAULT_PGMQ_SCHEMA).rpc("read", payload).execute()

    if isinstance(res.data, list):
        return [row for row in res.data if isinstance(row, dict)]
    if isinstance(res.data, dict):
        return [res.data]
    return []


def pgmq_delete(queue_name: str, *, message_id: int) -> bool:
    """PGMQ 큐에서 메시지를 영구 삭제합니다."""

    client = get_client()
    if not client:
        return False

    payload = {"queue_name": queue_name, "message_id": int(message_id)}
    client.schema(DEFAULT_PGMQ_SCHEMA).rpc("delete", payload).execute()
    return True


def pgmq_archive(queue_name: str, *, message_id: int) -> bool:
    """PGMQ 큐의 메시지를 아카이브 테이블로 이동합니다."""

    client = get_client()
    if not client:
        return False

    payload = {"queue_name": queue_name, "message_id": int(message_id)}
    client.schema(DEFAULT_PGMQ_SCHEMA).rpc("archive", payload).execute()
    return True


# ---------- cca_run_lock ----------


def get_run_lock(lock_key: str = DEFAULT_RUN_LOCK_KEY) -> Optional[Dict[str, Any]]:
    """전역 실행 락(단일 행)을 조회합니다."""

    client = get_client()
    if not client:
        return None

    res = (
        client.table("cca_run_lock")
        .select("*")
        .eq("lock_key", lock_key)
        .maybe_single()
        .execute()
    )
    return res.data or None


def try_acquire_run_lock(
    *,
    holder: str,
    run_id: str,
    lease_seconds: int,
    lock_key: str = DEFAULT_RUN_LOCK_KEY,
) -> bool:
    """전역 실행 락 획득을 시도합니다.

    - status=idle 이거나 lease_expires_at이 만료된 경우에만 running으로 전환합니다.
    - PostgREST update 필터로 동시성(원자성)을 확보합니다.
    """

    client = get_client()
    if not client:
        return False

    now = _utc_now()
    now_iso = _to_utc_iso(now)
    lease_duration = max(1, int(lease_seconds))
    lease_expires_at = _to_utc_iso(now + dt.timedelta(seconds=lease_duration))

    payload = {
        "status": "running",
        "holder": holder,
        "run_id": run_id,
        "started_at": now_iso,
        "lease_expires_at": lease_expires_at,
        "updated_at": now_iso,
    }

    # NOTE: or_ 필터는 문자열 DSL을 사용하므로 ISO 문자열에 공백이 없어야 한다.
    res = (
        client.table("cca_run_lock")
        .update(payload)
        .eq("lock_key", lock_key)
        .or_(f"status.eq.idle,lease_expires_at.lt.{now_iso}")
        .execute()
    )

    return bool(res.data)


def release_run_lock(*, holder: str, lock_key: str = DEFAULT_RUN_LOCK_KEY) -> bool:
    """전역 실행 락을 해제합니다.

    - holder가 일치할 때만 해제하여 다른 인스턴스의 락을 실수로 풀지 않습니다.
    """

    client = get_client()
    if not client:
        return False

    now_iso = _to_utc_iso(_utc_now())

    payload = {
        "status": "idle",
        "holder": None,
        "run_id": None,
        "started_at": None,
        "lease_expires_at": None,
        "updated_at": now_iso,
    }

    res = (
        client.table("cca_run_lock")
        .update(payload)
        .eq("lock_key", lock_key)
        .eq("holder", holder)
        .execute()
    )

    return bool(res.data)


# ---------- cca_job_dedupe ----------


def get_job_dedupe(dedupe_key: str) -> Optional[Dict[str, Any]]:
    """디듀프 키로 잡 대기/실행 상태를 조회합니다."""

    client = get_client()
    if not client:
        return None

    res = (
        client.table("cca_job_dedupe")
        .select("*")
        .eq("dedupe_key", dedupe_key)
        .maybe_single()
        .execute()
    )
    return res.data or None


def upsert_job_dedupe(
    *,
    dedupe_key: str,
    queue_name: str,
    msg_id: Optional[int],
    status: str,
    run_id: str,
    requested_by: str,
    requested_at: str,
) -> bool:
    """디듀프 테이블을 upsert로 갱신합니다."""

    client = get_client()
    if not client:
        return False

    now_iso = _to_utc_iso(_utc_now())
    payload: Dict[str, Any] = {
        "dedupe_key": dedupe_key,
        "queue_name": queue_name,
        "msg_id": msg_id,
        "status": status,
        "run_id": run_id,
        "requested_by": requested_by,
        "requested_at": requested_at,
        "updated_at": now_iso,
    }

    client.table("cca_job_dedupe").upsert(payload).execute()
    return True


def update_job_dedupe_status(
    *,
    dedupe_key: str,
    status: str,
    run_id: Optional[str] = None,
) -> bool:
    """디듀프 상태를 업데이트합니다."""

    client = get_client()
    if not client:
        return False

    now_iso = _to_utc_iso(_utc_now())
    payload: Dict[str, Any] = {"status": status, "updated_at": now_iso}
    if run_id is not None:
        payload["run_id"] = run_id

    res = (
        client.table("cca_job_dedupe")
        .update(payload)
        .eq("dedupe_key", dedupe_key)
        .execute()
    )
    return bool(res.data)
