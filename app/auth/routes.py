"""인증 엔드포인트."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel

from .middleware import SessionUser, get_current_user, require_admin
from .service import AuthService, auth_service


router = APIRouter(prefix="/auth", tags=["auth"])


class LoginBody(BaseModel):
    username: str
    password: str


class CreateUserBody(LoginBody):
    role: str = "user"


class UnblockIPBody(BaseModel):
    ip_address: str


def get_service() -> AuthService:
    return auth_service


def get_client_ip(request: Request) -> str:
    """클라이언트 IP 주소 추출 (프록시 지원)."""
    # X-Forwarded-For 헤더 확인 (프록시/로드밸런서 뒤)
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        # 첫 번째 IP가 실제 클라이언트 IP
        return forwarded.split(",")[0].strip()

    # X-Real-IP 헤더 확인
    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip.strip()

    # 직접 연결된 클라이언트 IP
    if request.client:
        return request.client.host

    return "unknown"


@router.post("/login")
def login(
    body: LoginBody,
    request: Request,
    service: AuthService = Depends(get_service),
):
    client_ip = get_client_ip(request)

    # IP 차단 여부 확인
    if service.is_ip_blocked(client_ip):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="해당 IP는 로그인 시도 횟수 초과로 차단되었습니다. 관리자에게 문의하세요.",
        )

    user = service.authenticate(body.username, body.password)

    if not user:
        # 실패 기록 및 차단 여부 확인
        was_blocked = service.record_login_attempt(
            client_ip, body.username, success=False
        )

        if was_blocked:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="로그인 시도 횟수를 초과하여 IP가 차단되었습니다.",
            )

        # 남은 시도 횟수 안내
        from .service import MAX_LOGIN_ATTEMPTS

        remaining = MAX_LOGIN_ATTEMPTS - service.get_recent_failed_attempts(client_ip)

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"로그인 실패 (남은 시도: {max(0, remaining)}회)",
        )

    # 성공 기록
    service.record_login_attempt(client_ip, body.username, success=True)

    request.session["user"] = {"username": user.username, "role": user.role}
    return {"ok": True, "user": {"username": user.username, "role": user.role}}


@router.post("/logout")
def logout(request: Request):
    request.session.pop("user", None)
    return {"ok": True}


@router.get("/me")
def me(user: SessionUser = Depends(get_current_user)):
    if user is None:
        return {"authenticated": False}
    return {
        "authenticated": True,
        "user": {"username": user.username, "role": user.role},
    }


@router.post("/users")
def create_user(
    body: CreateUserBody,
    _: SessionUser = Depends(require_admin),
    service: AuthService = Depends(get_service),
):
    user = service.create_user(body.username, body.password, role=body.role)
    return {"ok": True, "user": {"username": user.username, "role": user.role}}


@router.get("/users")
def list_users(
    _: SessionUser = Depends(require_admin),
    service: AuthService = Depends(get_service),
):
    return {"items": service.list_users()}


# ===== IP 차단 관리 (관리자 전용) =====


@router.get("/blocked-ips")
def list_blocked_ips(
    _: SessionUser = Depends(require_admin),
    service: AuthService = Depends(get_service),
):
    """차단된 IP 목록 조회."""
    return {"items": service.list_blocked_ips()}


@router.post("/unblock-ip")
def unblock_ip(
    body: UnblockIPBody,
    _: SessionUser = Depends(require_admin),
    service: AuthService = Depends(get_service),
):
    """IP 차단 해제."""
    success = service.unblock_ip(body.ip_address)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="해당 IP는 차단 목록에 없습니다.",
        )
    return {"ok": True, "message": f"{body.ip_address} 차단 해제됨"}
