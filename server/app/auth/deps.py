"""FastAPI 의존성(Supabase JWT 검증)."""

from __future__ import annotations

from fastapi import Depends, Header, HTTPException, Request, status

from .supabase import SupabaseAuthError, SupabaseUser, verify_supabase_token


def _get_bearer_token(authorization: str | None = Header(None)) -> str:
    """Authorization 헤더에서 Bearer 토큰을 추출한다."""
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization 헤더가 필요합니다.",
        )
    if not authorization.lower().startswith("bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization 형식이 올바르지 않습니다.",
        )
    token = authorization.split(" ", 1)[1].strip()
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="토큰이 비어있습니다.",
        )
    return token


def get_current_user(
    request: Request, token: str = Depends(_get_bearer_token)
) -> SupabaseUser:
    """Supabase JWT를 검증하고 사용자 정보를 반환한다."""
    try:
        user = verify_supabase_token(token)
        # 요청 컨텍스트에 사용자 정보를 저장해 후속 의존성이 재사용할 수 있게 한다.
        request.state.user = user
        return user
    except SupabaseAuthError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(exc),
        ) from exc


def require_user(user: SupabaseUser = Depends(get_current_user)) -> SupabaseUser:
    """인증된 사용자만 허용."""
    return user


def require_admin(user: SupabaseUser = Depends(get_current_user)) -> SupabaseUser:
    """관리자만 허용."""
    if user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="관리자 권한 필요",
        )
    return user


__all__ = ["SupabaseUser", "require_user", "require_admin", "get_current_user"]
