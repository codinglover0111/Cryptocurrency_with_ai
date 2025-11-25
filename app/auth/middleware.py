"""세션 헬퍼."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from fastapi import Depends, HTTPException, Request, status


@dataclass
class SessionUser:
    username: str
    role: str


def get_current_user(request: Request) -> Optional[SessionUser]:
    data = request.session.get("user") if hasattr(request, "session") else None
    if not data:
        return None
    return SessionUser(username=data.get("username", ""), role=data.get("role", "user"))


def require_user(user: SessionUser = Depends(get_current_user)) -> SessionUser:
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="로그인이 필요합니다.")
    return user


def require_admin(user: SessionUser = Depends(require_user)) -> SessionUser:
    if user.role != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="관리자 권한 필요")
    return user

