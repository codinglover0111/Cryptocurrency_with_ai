"""일반 사용자 API."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.auth.deps import SupabaseUser, require_user
from app.config import load_runtime_config


router = APIRouter(prefix="/user", tags=["user"])


@router.get("/settings")
def read_settings(user: SupabaseUser = Depends(require_user)):
    runtime = load_runtime_config()
    username = user.email or user.id
    return {
        "user": {
            "id": user.id,
            "email": user.email,
            "username": username,
            "role": user.role,
        },
        "agents": runtime.get("agents"),
        "scheduler": runtime.get("scheduler"),
    }

