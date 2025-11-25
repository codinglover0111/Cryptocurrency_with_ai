"""관리자 API."""

from __future__ import annotations

import os
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.auth.middleware import require_admin
from app.auth.api_keys import ApiKeyService
from app.config import (
    AGENT_CONFIG,
    SCHEDULER_CONFIG,
    load_runtime_config,
    save_runtime_config,
)
from utils.storage import TradeStore, StorageConfig


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


class ApiKeyBulkPayload(BaseModel):
    """여러 API 키 일괄 설정."""

    keys: List[ApiKeyPayload]


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


def _get_scheduler_state() -> Dict[str, Any]:
    """스케줄러 실행 상태를 DB에서 읽어옵니다."""
    try:
        store = TradeStore(
            StorageConfig(
                mysql_url=os.getenv("MYSQL_URL"),
                sqlite_path=os.getenv("SQLITE_PATH"),
            )
        )
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
        store = TradeStore(
            StorageConfig(
                mysql_url=os.getenv("MYSQL_URL"),
                sqlite_path=os.getenv("SQLITE_PATH"),
            )
        )
        store.set_scheduler_state("paused", "1")
        return {"ok": True, "paused": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/scheduler/resume")
def resume_scheduler(_: str = Depends(require_admin)):
    """스케줄러 재개."""
    try:
        store = TradeStore(
            StorageConfig(
                mysql_url=os.getenv("MYSQL_URL"),
                sqlite_path=os.getenv("SQLITE_PATH"),
            )
        )
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
