"""인증 서비스 레이어."""

from __future__ import annotations

import os
from typing import Optional

from passlib.context import CryptContext
from sqlalchemy.orm import Session

from .models import SessionLocal, User


pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


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
                {"id": u.id, "username": u.username, "role": u.role, "active": u.is_active}
                for u in users
            ]


auth_service = AuthService()

