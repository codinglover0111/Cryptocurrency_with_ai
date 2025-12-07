"""Supabase JWT 검증 유틸리티."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import requests
from jose import jwt
from jose.exceptions import JWTError


class SupabaseAuthError(Exception):
    """Supabase 인증/검증 오류."""


@dataclass
class SupabaseUser:
    """검증된 Supabase 사용자 정보."""

    id: str
    email: Optional[str]
    role: str
    claims: Dict[str, Any]


_JWKS_CACHE: Dict[str, Any] = {"expires_at": 0.0, "keys": None}
_JWKS_CACHE_TTL = float(os.getenv("SUPABASE_JWKS_CACHE_SECONDS", "300"))


def _resolve_jwks_url() -> str:
    """JWKS URL을 반환한다. SUPABASE_JWKS_URL 없을 경우 SUPABASE_URL로 유도."""
    if os.getenv("SUPABASE_JWKS_URL"):
        return os.getenv("SUPABASE_JWKS_URL")  # type: ignore[arg-type]

    base_url = os.getenv("SUPABASE_URL")
    if not base_url:
        raise SupabaseAuthError(
            "SUPABASE_JWKS_URL 또는 SUPABASE_URL 환경변수가 필요합니다."
        )
    return base_url.rstrip("/") + "/auth/v1/jwks"


def _fetch_jwks() -> Dict[str, Any]:
    """Supabase JWKS를 가져온다."""
    url = _resolve_jwks_url()
    try:
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        if "keys" not in data:
            raise SupabaseAuthError("JWKS 응답에 keys 필드가 없습니다.")
        return data
    except Exception as exc:  # pylint: disable=broad-except
        raise SupabaseAuthError(f"Supabase JWKS를 가져오지 못했습니다: {exc}") from exc


def get_supabase_jwks() -> Dict[str, Any]:
    """캐싱된 Supabase JWKS를 반환한다."""
    now = time.time()
    if _JWKS_CACHE.get("keys") and now < float(_JWKS_CACHE.get("expires_at", 0)):
        return _JWKS_CACHE["keys"]

    jwks = _fetch_jwks()
    _JWKS_CACHE["keys"] = jwks
    _JWKS_CACHE["expires_at"] = now + _JWKS_CACHE_TTL
    return jwks


def _extract_role(claims: Dict[str, Any]) -> str:
    """app_metadata.role → user_metadata.role → role → 기본 user 순서로 역할을 추출."""
    app_meta = claims.get("app_metadata") or {}
    user_meta = claims.get("user_metadata") or {}

    role = app_meta.get("role") or user_meta.get("role") or claims.get("role")
    if not role:
        return "user"
    try:
        return str(role)
    except Exception:  # pylint: disable=broad-except
        return "user"


def verify_supabase_token(token: str) -> SupabaseUser:
    """Supabase JWT를 검증하고 사용자 정보를 반환한다."""
    try:
        header = jwt.get_unverified_header(token)
        kid = header.get("kid")
    except JWTError as exc:
        raise SupabaseAuthError("JWT 헤더 파싱에 실패했습니다.") from exc

    jwks = get_supabase_jwks()
    keys = jwks.get("keys", [])
    key = next((k for k in keys if k.get("kid") == kid), None)
    if not key:
        raise SupabaseAuthError("JWKS에 일치하는 키가 없습니다.")

    try:
        claims = jwt.decode(
            token,
            key,
            algorithms=[key.get("alg", "RS256")],
            options={"verify_aud": False},
        )
    except JWTError as exc:
        raise SupabaseAuthError(f"JWT 검증에 실패했습니다: {exc}") from exc

    user_id = claims.get("sub")
    if not user_id:
        raise SupabaseAuthError("JWT에 sub 클레임이 없습니다.")

    role = _extract_role(claims)
    email = claims.get("email")

    return SupabaseUser(id=str(user_id), email=email, role=role, claims=claims)
