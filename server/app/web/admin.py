"""관리자 API."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.auth.deps import require_admin
from app.auth.api_keys import ApiKeyService
from app.config import (
    AGENT_CONFIG,
    SCHEDULER_CONFIG,
    load_runtime_config,
    save_runtime_config,
)
from utils.storage import get_trade_store
from app.services import supabase_repo


AVAILABLE_MODELS: Dict[str, list[str]] = {
    "openai": ["gpt-4o-mini", "gpt-4.1-mini"],
    "gemini": ["gemini-2.0-flash-exp", "gemini-1.5-pro"],
    "openrouter": ["deepseek/deepseek-chat", "qwen/qwen-2.5-72b-instruct"],
    "anthropic": ["claude-3-5-sonnet-latest"],
}


class AgentConfigPayload(BaseModel):
    indicator_agent: Dict[str, object] = Field(default_factory=dict)
    pattern_agent: Dict[str, object] = Field(default_factory=dict)
    trend_agent: Dict[str, object] = Field(default_factory=dict)
    decision_agent: Dict[str, object] = Field(default_factory=dict)


class SchedulerPayload(BaseModel):
    automation_minutes: int = Field(ge=1, le=180)
    loss_review_minutes: int = Field(ge=1, le=720)
    cold_start: bool = False


class ApiKeyPayload(BaseModel):
    """API 키 설정 요청."""

    provider: str
    key_type: str
    value: str
    environment: str = "default"


class ApiKeyDeletePayload(BaseModel):
    """API 키 삭제 요청."""

    provider: str
    key_type: str
    environment: str = "default"


class RiskConfigPayload(BaseModel):
    """리스크 설정 요청."""

    default_leverage: int = Field(ge=1, le=100, default=5)
    max_loss_percent: int = Field(ge=1, le=100, default=40)
    position_allocation_percent: int = Field(ge=1, le=100, default=20)


class TradingSymbolsPayload(BaseModel):
    """거래 심볼 설정 요청."""

    symbols: List[str] = Field(default_factory=list, min_length=1)


class RunSymbolPayload(BaseModel):
    """특정 심볼 즉시 실행 요청."""

    symbol: str


class ApiKeyBulkPayload(BaseModel):
    """여러 API 키 일괄 설정."""

    keys: List[ApiKeyPayload]


class PromptPayload(BaseModel):
    """단일 프롬프트 저장 요청."""

    agent_type: str
    prompt_template: str


class PromptsBulkPayload(BaseModel):
    """여러 프롬프트 일괄 저장 요청."""

    prompts: Dict[str, str]  # {agent_type: prompt_template, ...}


router = APIRouter(prefix="/admin", tags=["admin"])


@router.get("/models")
def list_models(_: str = Depends(require_admin)):
    return {"providers": AVAILABLE_MODELS}


@router.get("/agent-config")
def get_agent_config(_: str = Depends(require_admin)):
    runtime = load_runtime_config()
    return runtime.get("agents", AGENT_CONFIG)


@router.post("/agent-config")
def update_agent_config(payload: AgentConfigPayload, _: str = Depends(require_admin)):
    runtime = load_runtime_config()
    updated = {**AGENT_CONFIG}
    for key, value in payload.model_dump().items():
        if not value:
            continue
        base = dict(updated.get(key) or {})
        base.update(value)
        updated[key] = base
    runtime["agents"] = updated
    save_runtime_config(runtime)
    return {"ok": True, "agents": runtime["agents"]}


def _normalize_scheduler_states(states: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Normalize key/value rows into typed scheduler state."""
    result: Dict[str, Any] = {}
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
        if data.get("updated_at"):
            result["updated_at"] = data["updated_at"]
    return result


def _get_scheduler_state() -> Dict[str, Any]:
    """Return scheduler state (Supabase first, DB fallback)."""
    try:
        states = supabase_repo.get_scheduler_state_all()
        if states:
            return _normalize_scheduler_states(states)
    except Exception:
        pass
    try:
        store = get_trade_store()
        states = store.get_all_scheduler_states()
        return _normalize_scheduler_states(states)
    except Exception:
        return {}


def _calculate_next_run(
    last_run_iso: Optional[str], interval_minutes: int
) -> Optional[str]:
    """마지막 실행 시간과 주기를 기반으로 다음 실행 시간을 계산합니다."""
    if not last_run_iso:
        return None
    try:
        last_run = datetime.fromisoformat(last_run_iso.replace("Z", "+00:00"))
        next_run = last_run + timedelta(minutes=interval_minutes)
        return next_run.isoformat().replace("+00:00", "Z")
    except Exception:
        return None


@router.get("/scheduler")
def get_scheduler(_: str = Depends(require_admin)):
    runtime = load_runtime_config()
    config = runtime.get("scheduler", SCHEDULER_CONFIG)

    # 스케줄러 실행 상태 가져오기
    state = _get_scheduler_state()

    # paused 상태 확인
    paused = state.get("paused") == "1" if state.get("paused") else False

    # 설정과 상태 병합
    result = {
        **config,
        "is_running": state.get("is_running", False),
        "paused": paused,
        "last_automation_run": state.get("last_automation_run"),
        "last_review_run": state.get("last_review_run"),
        "updated_at": state.get("updated_at"),
    }

    # 다음 실행 시간 계산
    automation_minutes = config.get(
        "automation_minutes", SCHEDULER_CONFIG["automation_minutes"]
    )
    result["next_automation_run"] = _calculate_next_run(
        state.get("last_automation_run"), automation_minutes
    )

    return result


@router.post("/scheduler")
def update_scheduler(payload: SchedulerPayload, _: str = Depends(require_admin)):
    runtime = load_runtime_config()
    runtime["scheduler"] = payload.model_dump()
    save_runtime_config(runtime)
    return {"ok": True, "scheduler": runtime["scheduler"]}


# ===== API Keys Management =====


@router.get("/api-keys/providers")
def list_api_key_providers(_: str = Depends(require_admin)):
    """지원하는 API 키 Provider 목록 반환."""
    return {"providers": ApiKeyService.PROVIDERS}


@router.get("/api-keys/status")
def get_api_keys_status(_: str = Depends(require_admin)):
    """각 Provider별 API 키 설정 상태 반환."""
    return {"status": ApiKeyService.get_status()}


@router.get("/api-keys")
def list_api_keys(_: str = Depends(require_admin)):
    """저장된 API 키 목록 반환 (마스킹됨)."""
    return {"keys": ApiKeyService.list_keys()}


@router.post("/api-keys")
def set_api_key(payload: ApiKeyPayload, _: str = Depends(require_admin)):
    """API 키 설정."""
    # Provider 검증
    if payload.provider not in ApiKeyService.PROVIDERS:
        raise HTTPException(
            status_code=400, detail=f"Unknown provider: {payload.provider}"
        )

    provider_config = ApiKeyService.PROVIDERS[payload.provider]

    # key_type 검증
    if payload.key_type not in provider_config["key_types"]:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid key_type '{payload.key_type}' for provider '{payload.provider}'",
        )

    # environment 검증
    if payload.environment not in provider_config["environments"]:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid environment '{payload.environment}' for provider '{payload.provider}'",
        )

    success = ApiKeyService.set_key(
        provider=payload.provider,
        key_type=payload.key_type,
        value=payload.value,
        environment=payload.environment,
    )

    if not success:
        raise HTTPException(status_code=500, detail="Failed to save API key")

    return {"ok": True, "message": "API key saved successfully"}


@router.post("/api-keys/bulk")
def set_api_keys_bulk(payload: ApiKeyBulkPayload, _: str = Depends(require_admin)):
    """여러 API 키 일괄 설정."""
    results = []
    for key_data in payload.keys:
        # Provider 검증
        if key_data.provider not in ApiKeyService.PROVIDERS:
            results.append(
                {
                    "provider": key_data.provider,
                    "key_type": key_data.key_type,
                    "environment": key_data.environment,
                    "ok": False,
                    "error": f"Unknown provider: {key_data.provider}",
                }
            )
            continue

        provider_config = ApiKeyService.PROVIDERS[key_data.provider]

        # key_type 검증
        if key_data.key_type not in provider_config["key_types"]:
            results.append(
                {
                    "provider": key_data.provider,
                    "key_type": key_data.key_type,
                    "environment": key_data.environment,
                    "ok": False,
                    "error": f"Invalid key_type",
                }
            )
            continue

        # environment 검증
        if key_data.environment not in provider_config["environments"]:
            results.append(
                {
                    "provider": key_data.provider,
                    "key_type": key_data.key_type,
                    "environment": key_data.environment,
                    "ok": False,
                    "error": f"Invalid environment",
                }
            )
            continue

        success = ApiKeyService.set_key(
            provider=key_data.provider,
            key_type=key_data.key_type,
            value=key_data.value,
            environment=key_data.environment,
        )

        results.append(
            {
                "provider": key_data.provider,
                "key_type": key_data.key_type,
                "environment": key_data.environment,
                "ok": success,
            }
        )

    return {"results": results}


@router.delete("/api-keys")
def delete_api_key(payload: ApiKeyDeletePayload, _: str = Depends(require_admin)):
    """API 키 삭제."""
    success = ApiKeyService.delete_key(
        provider=payload.provider,
        key_type=payload.key_type,
        environment=payload.environment,
    )

    return {"ok": success}


# ===== Scheduler Pause/Resume =====


@router.post("/scheduler/pause")
def pause_scheduler(_: str = Depends(require_admin)):
    """스케줄러 일시 중단."""
    try:
        if not supabase_repo.set_scheduler_state("paused", "1"):
            store = get_trade_store()
            store.set_scheduler_state("paused", "1")
        return {"ok": True, "paused": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/scheduler/resume")
def resume_scheduler(_: str = Depends(require_admin)):
    """스케줄러 재개."""
    try:
        if not supabase_repo.set_scheduler_state("paused", "0"):
            store = get_trade_store()
            store.set_scheduler_state("paused", "0")
        return {"ok": True, "paused": False}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ===== Risk Config =====


@router.get("/risk-config")
def get_risk_config(_: str = Depends(require_admin)):
    """리스크 설정 조회."""
    runtime = load_runtime_config()
    risk_config = runtime.get(
        "risk",
        {
            "default_leverage": 5,
            "max_loss_percent": 40,
            "position_allocation_percent": 20,
        },
    )
    return risk_config


@router.post("/risk-config")
def update_risk_config(payload: RiskConfigPayload, _: str = Depends(require_admin)):
    """리스크 설정 업데이트."""
    runtime = load_runtime_config()
    runtime["risk"] = payload.model_dump()
    save_runtime_config(runtime)
    return {"ok": True, "risk": runtime["risk"]}


# ===== Trading Symbols Management =====


@router.get("/trading-symbols/available")
def get_available_symbols(_: str = Depends(require_admin)):
    """거래 가능한 심볼 목록 반환."""
    from app.core.symbols import AVAILABLE_SYMBOLS

    return {"symbols": list(AVAILABLE_SYMBOLS)}


@router.get("/trading-symbols")
def get_trading_symbols(_: str = Depends(require_admin)):
    """현재 설정된 거래 심볼 목록 반환."""
    from app.core.symbols import (
        parse_trading_symbols,
        get_trading_symbols_from_db,
        DEFAULT_SYMBOLS,
    )

    # DB에서 설정된 심볼 확인
    db_symbols = get_trading_symbols_from_db()

    # 현재 활성화된 심볼
    active_symbols = parse_trading_symbols()

    return {
        "symbols": active_symbols,
        "source": "db" if db_symbols else "env_or_default",
        "defaults": list(DEFAULT_SYMBOLS),
    }


@router.post("/trading-symbols")
def update_trading_symbols(
    payload: TradingSymbolsPayload, _: str = Depends(require_admin)
):
    """거래 심볼 목록 업데이트."""
    from app.core.symbols import save_trading_symbols_to_db, AVAILABLE_SYMBOLS

    # 심볼 검증 (선택적 - 알려진 심볼인지 확인)
    normalized_symbols = [s.strip().upper() for s in payload.symbols if s.strip()]

    # 유효하지 않은 심볼 경고 (저장은 진행)
    unknown_symbols = [s for s in normalized_symbols if s not in AVAILABLE_SYMBOLS]

    success = save_trading_symbols_to_db(normalized_symbols)

    if not success:
        raise HTTPException(status_code=500, detail="Failed to save trading symbols")

    return {
        "ok": True,
        "symbols": normalized_symbols,
        "warnings": (
            f"Unknown symbols (may still work): {unknown_symbols}"
            if unknown_symbols
            else None
        ),
    }


# ===== Immediate Execution =====


@router.post("/run-now", status_code=202)
def run_automation_now(_: str = Depends(require_admin)):
    """전체 심볼 즉시 분석을 요청합니다.

    - 기본 동작: Supabase Queues(PGMQ)에 enqueue만 수행하고 202를 반환합니다.
    - 롤백/호환: USE_LEGACY_EXECUTION=1 이면 기존처럼 스레드로 직접 실행합니다.
    """

    import logging
    import os
    import threading
    import uuid
    from datetime import datetime, timezone

    from app.core.symbols import parse_trading_symbols
    from app.workflows.trading import run_automation_for_all_symbols

    logger = logging.getLogger(__name__)
    symbols = parse_trading_symbols()
    symbol_count = len(symbols)

    logger.info(f"[즉시실행] 전체 심볼 분석 요청 - {symbol_count}개 심볼: {symbols}")

    # 레거시 실행(롤백 플랜): 로컬 스레드로 즉시 실행
    if os.getenv("USE_LEGACY_EXECUTION") == "1":

        def run_in_background():
            """스레드에서 전체 심볼 자동매매를 실행합니다(레거시)."""

            try:
                logger.info("[즉시실행] 백그라운드 스레드 시작(레거시)")
                run_automation_for_all_symbols()
                logger.info("[즉시실행] 백그라운드 스레드 완료(레거시)")
            except Exception as e:
                logger.exception(f"[즉시실행] 오류 발생(레거시): {e}")

        thread = threading.Thread(target=run_in_background, daemon=True)
        thread.start()

        return {
            "ok": True,
            "message": f"전체 {symbol_count}개 심볼 분석이 백그라운드에서 시작되었습니다.(레거시)",
            "symbols": symbols,
        }

    # 신규 실행: 큐 enqueue 전용
    client = supabase_repo.get_client()
    if not client:
        raise HTTPException(status_code=503, detail="supabase_not_configured")

    lock = supabase_repo.get_run_lock()
    if lock and str(lock.get("status") or "").lower() == "running":
        # lease_expires_at이 만료된 경우에는 "실행 중"으로 보지 않는다.
        lease_expires_at = lock.get("lease_expires_at")
        is_expired = False
        if lease_expires_at:
            try:
                lease_dt = datetime.fromisoformat(
                    str(lease_expires_at).replace("Z", "+00:00")
                )
                is_expired = lease_dt <= datetime.now(timezone.utc)
            except Exception:
                is_expired = False

        if not is_expired:
            raise HTTPException(
                status_code=409,
                detail={"error": "already_running", "run_id": lock.get("run_id")},
            )

    dedupe_key = "automation_all"

    dedupe_row = supabase_repo.get_job_dedupe(dedupe_key)
    if dedupe_row and str(dedupe_row.get("status") or "").lower() == "queued":
        return {
            "status": "already_queued",
            "run_id": dedupe_row.get("run_id"),
            "queue": dedupe_row.get("queue_name"),
        }

    run_id = str(uuid.uuid4())
    requested_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    message = {
        "job_type": "automation_all",
        "dedupe_key": dedupe_key,
        "symbols": symbols,
        "requested_by": "admin_api",
        "requested_at": requested_at,
        "run_id": run_id,
    }

    msg_id = supabase_repo.pgmq_send(
        supabase_repo.QUEUE_MANUAL, message, sleep_seconds=0
    )
    if msg_id is None:
        raise HTTPException(status_code=500, detail="enqueue_failed")

    supabase_repo.upsert_job_dedupe(
        dedupe_key=dedupe_key,
        queue_name=supabase_repo.QUEUE_MANUAL,
        msg_id=msg_id,
        status="queued",
        run_id=run_id,
        requested_by="admin_api",
        requested_at=requested_at,
    )

    logger.info(
        "[즉시실행] enqueue 완료: queue=%s run_id=%s msg_id=%s",
        supabase_repo.QUEUE_MANUAL,
        run_id,
        msg_id,
    )

    return {
        "status": "queued",
        "run_id": run_id,
        "queue": supabase_repo.QUEUE_MANUAL,
        "symbols": symbols,
    }


@router.post("/run-symbol", status_code=202)
def run_symbol_now(payload: RunSymbolPayload, _: str = Depends(require_admin)):
    """특정 심볼 즉시 분석을 요청합니다.

    - 기본 동작: Supabase Queues(PGMQ)에 enqueue만 수행하고 202를 반환합니다.
    - 롤백/호환: USE_LEGACY_EXECUTION=1 이면 기존처럼 스레드로 직접 실행합니다.
    """

    import logging
    import os
    import threading
    import uuid
    from datetime import datetime, timezone

    from app.workflows.trading import automation_for_symbol

    logger = logging.getLogger(__name__)
    symbol = payload.symbol.strip().upper()

    if not symbol:
        raise HTTPException(status_code=400, detail="심볼이 비어있습니다.")

    logger.info(f"[즉시실행] 특정 심볼 분석 요청: {symbol}")

    # 레거시 실행(롤백 플랜): 로컬 스레드로 즉시 실행
    if os.getenv("USE_LEGACY_EXECUTION") == "1":

        def run_in_background():
            """스레드에서 특정 심볼 자동매매를 실행합니다(레거시)."""

            try:
                logger.info(f"[즉시실행] {symbol} 백그라운드 분석 시작(레거시)")
                automation_for_symbol(symbol)
                logger.info(f"[즉시실행] {symbol} 백그라운드 분석 완료(레거시)")
            except Exception as e:
                logger.exception(f"[즉시실행] {symbol} 오류 발생(레거시): {e}")

        thread = threading.Thread(target=run_in_background, daemon=True)
        thread.start()

        return {
            "ok": True,
            "message": f"{symbol} 분석이 백그라운드에서 시작되었습니다.(레거시)",
            "symbol": symbol,
        }

    # 신규 실행: 큐 enqueue 전용
    client = supabase_repo.get_client()
    if not client:
        raise HTTPException(status_code=503, detail="supabase_not_configured")

    lock = supabase_repo.get_run_lock()
    if lock and str(lock.get("status") or "").lower() == "running":
        # lease_expires_at이 만료된 경우에는 "실행 중"으로 보지 않는다.
        lease_expires_at = lock.get("lease_expires_at")
        is_expired = False
        if lease_expires_at:
            try:
                lease_dt = datetime.fromisoformat(
                    str(lease_expires_at).replace("Z", "+00:00")
                )
                is_expired = lease_dt <= datetime.now(timezone.utc)
            except Exception:
                is_expired = False

        if not is_expired:
            raise HTTPException(
                status_code=409,
                detail={"error": "already_running", "run_id": lock.get("run_id")},
            )

    dedupe_key = f"automation_symbol:{symbol}"

    dedupe_row = supabase_repo.get_job_dedupe(dedupe_key)
    if dedupe_row and str(dedupe_row.get("status") or "").lower() == "queued":
        return {
            "status": "already_queued",
            "run_id": dedupe_row.get("run_id"),
            "queue": dedupe_row.get("queue_name"),
            "symbol": symbol,
        }

    run_id = str(uuid.uuid4())
    requested_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    message = {
        "job_type": "automation_symbol",
        "dedupe_key": dedupe_key,
        "symbol": symbol,
        "requested_by": "admin_api",
        "requested_at": requested_at,
        "run_id": run_id,
    }

    msg_id = supabase_repo.pgmq_send(
        supabase_repo.QUEUE_MANUAL, message, sleep_seconds=0
    )
    if msg_id is None:
        raise HTTPException(status_code=500, detail="enqueue_failed")

    supabase_repo.upsert_job_dedupe(
        dedupe_key=dedupe_key,
        queue_name=supabase_repo.QUEUE_MANUAL,
        msg_id=msg_id,
        status="queued",
        run_id=run_id,
        requested_by="admin_api",
        requested_at=requested_at,
    )

    logger.info(
        "[즉시실행] enqueue 완료: queue=%s run_id=%s msg_id=%s symbol=%s",
        supabase_repo.QUEUE_MANUAL,
        run_id,
        msg_id,
        symbol,
    )

    return {
        "status": "queued",
        "run_id": run_id,
        "queue": supabase_repo.QUEUE_MANUAL,
        "symbol": symbol,
    }


# ===== Agent Prompts Management =====


@router.get("/prompts")
def get_prompts(_: str = Depends(require_admin)):
    """모든 에이전트 프롬프트 조회."""
    from app.agents.prompt_service import get_all_prompts, get_prompt_variables

    prompts = get_all_prompts()

    # 각 프롬프트에 사용 가능한 변수 목록 추가
    for agent_type in prompts:
        prompts[agent_type]["variables"] = get_prompt_variables(agent_type)

    return {"prompts": prompts}


@router.post("/prompts")
def update_prompt(payload: PromptPayload, _: str = Depends(require_admin)):
    """단일 에이전트 프롬프트 저장."""
    from app.agents.prompt_service import save_prompt, DEFAULT_PROMPTS

    if payload.agent_type not in DEFAULT_PROMPTS:
        raise HTTPException(
            status_code=400,
            detail=f"알 수 없는 에이전트 타입: {payload.agent_type}",
        )

    if not payload.prompt_template.strip():
        raise HTTPException(
            status_code=400,
            detail="프롬프트 템플릿이 비어있습니다.",
        )

    success = save_prompt(payload.agent_type, payload.prompt_template)

    if not success:
        raise HTTPException(status_code=500, detail="프롬프트 저장 실패")

    return {"ok": True, "agent_type": payload.agent_type}


@router.post("/prompts/bulk")
def update_prompts_bulk(payload: PromptsBulkPayload, _: str = Depends(require_admin)):
    """여러 에이전트 프롬프트 일괄 저장."""
    from app.agents.prompt_service import save_prompts_bulk, DEFAULT_PROMPTS

    # 유효한 프롬프트만 필터링
    valid_prompts = {}
    invalid_types = []

    for agent_type, prompt_template in payload.prompts.items():
        if agent_type not in DEFAULT_PROMPTS:
            invalid_types.append(agent_type)
            continue
        if prompt_template and prompt_template.strip():
            valid_prompts[agent_type] = prompt_template

    if not valid_prompts:
        raise HTTPException(
            status_code=400,
            detail="저장할 유효한 프롬프트가 없습니다.",
        )

    success = save_prompts_bulk(valid_prompts)

    if not success:
        raise HTTPException(status_code=500, detail="프롬프트 저장 실패")

    result = {
        "ok": True,
        "saved": list(valid_prompts.keys()),
    }

    if invalid_types:
        result["warnings"] = f"알 수 없는 에이전트 타입: {invalid_types}"

    return result


@router.post("/prompts/reset/{agent_type}")
def reset_prompt(agent_type: str, _: str = Depends(require_admin)):
    """특정 에이전트 프롬프트를 기본값으로 초기화."""
    from app.agents.prompt_service import reset_prompt as do_reset, DEFAULT_PROMPTS

    if agent_type not in DEFAULT_PROMPTS:
        raise HTTPException(
            status_code=400,
            detail=f"알 수 없는 에이전트 타입: {agent_type}",
        )

    success = do_reset(agent_type)

    if not success:
        raise HTTPException(status_code=500, detail="프롬프트 초기화 실패")

    return {"ok": True, "agent_type": agent_type}


@router.post("/prompts/reset-all")
def reset_all_prompts(_: str = Depends(require_admin)):
    """모든 에이전트 프롬프트를 기본값으로 초기화."""
    from app.agents.prompt_service import reset_all_prompts as do_reset_all

    success = do_reset_all()

    if not success:
        raise HTTPException(status_code=500, detail="일부 프롬프트 초기화 실패")

    return {"ok": True}
