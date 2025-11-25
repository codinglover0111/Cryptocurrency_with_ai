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


def get_service() -> AuthService:
    return auth_service


@router.post("/login")
def login(
    body: LoginBody,
    request: Request,
    service: AuthService = Depends(get_service),
):
    user = service.authenticate(body.username, body.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="로그인 실패"
        )
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
    return {"authenticated": True, "user": {"username": user.username, "role": user.role}}


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

