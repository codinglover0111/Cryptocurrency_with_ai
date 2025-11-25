"""인증 서비스 레이어."""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Optional

from passlib.context import CryptContext
from sqlalchemy.orm import Session

from .models import BlockedIP, LoginAttempt, SessionLocal, User


pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# 로그인 실패 허용 횟수 및 시간 창
MAX_LOGIN_ATTEMPTS = int(os.getenv("MAX_LOGIN_ATTEMPTS", "10"))
LOGIN_ATTEMPT_WINDOW_MINUTES = int(os.getenv("LOGIN_ATTEMPT_WINDOW_MINUTES", "30"))


class AuthService:
    """세션 기반 인증 도우미."""

    def __init__(self) -> None:
        self._ensure_default_admin()

    def _ensure_default_admin(self) -> None:
        username = os.getenv("ADMIN_USERNAME", "admin")
        password = os.getenv("ADMIN_PASSWORD", "admin123")
        with SessionLocal() as session:
            existing = (
                session.query(User).filter(User.username == username).one_or_none()
            )
            if existing is None:
                session.add(
                    User(
                        username=username,
                        password_hash=pwd_context.hash(password),
                        role="admin",
                        is_active=True,
                    )
                )
                session.commit()

    def _get_user(self, session: Session, username: str) -> Optional[User]:
        return session.query(User).filter(User.username == username).one_or_none()

    def is_ip_blocked(self, ip_address: str) -> bool:
        """해당 IP가 차단되었는지 확인."""
        with SessionLocal() as session:
            blocked = (
                session.query(BlockedIP)
                .filter(BlockedIP.ip_address == ip_address)
                .one_or_none()
            )
            return blocked is not None

    def get_recent_failed_attempts(self, ip_address: str) -> int:
        """최근 시간 창 내의 실패 횟수 반환."""
        cutoff = datetime.now(timezone.utc) - timedelta(
            minutes=LOGIN_ATTEMPT_WINDOW_MINUTES
        )
        with SessionLocal() as session:
            count = (
                session.query(LoginAttempt)
                .filter(
                    LoginAttempt.ip_address == ip_address,
                    LoginAttempt.success == False,
                    LoginAttempt.attempted_at >= cutoff,
                )
                .count()
            )
            return count

    def record_login_attempt(
        self, ip_address: str, username: str, success: bool
    ) -> bool:
        """로그인 시도를 기록하고, 실패 횟수 초과 시 IP를 차단.

        Returns:
            bool: IP가 차단되면 True, 아니면 False
        """
        with SessionLocal() as session:
            # 로그인 시도 기록
            attempt = LoginAttempt(
                ip_address=ip_address,
                username=username,
                success=success,
            )
            session.add(attempt)
            session.commit()

            # 성공한 경우 차단하지 않음
            if success:
                return False

            # 실패 횟수 확인
            cutoff = datetime.now(timezone.utc) - timedelta(
                minutes=LOGIN_ATTEMPT_WINDOW_MINUTES
            )
            fail_count = (
                session.query(LoginAttempt)
                .filter(
                    LoginAttempt.ip_address == ip_address,
                    LoginAttempt.success == False,
                    LoginAttempt.attempted_at >= cutoff,
                )
                .count()
            )

            # 허용 횟수 초과 시 IP 차단
            if fail_count >= MAX_LOGIN_ATTEMPTS:
                existing_block = (
                    session.query(BlockedIP)
                    .filter(BlockedIP.ip_address == ip_address)
                    .one_or_none()
                )
                if existing_block is None:
                    block = BlockedIP(
                        ip_address=ip_address,
                        reason=f"로그인 {fail_count}회 실패로 자동 차단",
                    )
                    session.add(block)
                    session.commit()
                return True

            return False

    def unblock_ip(self, ip_address: str) -> bool:
        """IP 차단 해제."""
        with SessionLocal() as session:
            blocked = (
                session.query(BlockedIP)
                .filter(BlockedIP.ip_address == ip_address)
                .one_or_none()
            )
            if blocked:
                session.delete(blocked)
                session.commit()
                return True
            return False

    def list_blocked_ips(self) -> list[dict]:
        """차단된 IP 목록 반환."""
        with SessionLocal() as session:
            blocked_list = session.query(BlockedIP).all()
            return [
                {
                    "id": b.id,
                    "ip_address": b.ip_address,
                    "reason": b.reason,
                    "blocked_at": b.blocked_at.isoformat() if b.blocked_at else None,
                    "is_permanent": b.is_permanent,
                }
                for b in blocked_list
            ]

    def authenticate(self, username: str, password: str) -> Optional[User]:
        with SessionLocal() as session:
            user = self._get_user(session, username)
            if not user or not user.is_active:
                return None
            if not pwd_context.verify(password, user.password_hash):
                return None
            return user

    def create_user(self, username: str, password: str, role: str = "user") -> User:
        with SessionLocal() as session:
            if self._get_user(session, username):
                raise ValueError("이미 존재하는 사용자입니다.")
            user = User(
                username=username,
                password_hash=pwd_context.hash(password),
                role=role,
                is_active=True,
            )
            session.add(user)
            session.commit()
            session.refresh(user)
            return user

    def list_users(self) -> list[dict]:
        with SessionLocal() as session:
            users = session.query(User).all()
            return [
                {
                    "id": u.id,
                    "username": u.username,
                    "role": u.role,
                    "active": u.is_active,
                }
                for u in users
            ]


auth_service = AuthService()
