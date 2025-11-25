"""일반 사용자 API."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.auth.middleware import SessionUser, require_user
from app.config import load_runtime_config


router = APIRouter(prefix="/user", tags=["user"])


@router.get("/settings")
def read_settings(user: SessionUser = Depends(require_user)):
    runtime = load_runtime_config()
    return {
        "user": {"username": user.username, "role": user.role},
        "agents": runtime.get("agents"),
        "scheduler": runtime.get("scheduler"),
    }

