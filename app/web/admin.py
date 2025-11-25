"""관리자 API."""

from __future__ import annotations

from typing import Dict

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from app.auth.middleware import require_admin
from app.config import (
    AGENT_CONFIG,
    SCHEDULER_CONFIG,
    load_runtime_config,
    save_runtime_config,
)


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


router = APIRouter(prefix="/admin", tags=["admin"])


@router.get("/models")
def list_models(_: str = Depends(require_admin)):
    return {"providers": AVAILABLE_MODELS}


@router.get("/agent-config")
def get_agent_config(_: str = Depends(require_admin)):
    runtime = load_runtime_config()
    return runtime.get("agents", AGENT_CONFIG)


@router.post("/agent-config")
def update_agent_config(
    payload: AgentConfigPayload, _: str = Depends(require_admin)
):
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


@router.get("/scheduler")
def get_scheduler(_: str = Depends(require_admin)):
    runtime = load_runtime_config()
    return runtime.get("scheduler", SCHEDULER_CONFIG)


@router.post("/scheduler")
def update_scheduler(payload: SchedulerPayload, _: str = Depends(require_admin)):
    runtime = load_runtime_config()
    runtime["scheduler"] = payload.model_dump()
    save_runtime_config(runtime)
    return {"ok": True, "scheduler": runtime["scheduler"]}

