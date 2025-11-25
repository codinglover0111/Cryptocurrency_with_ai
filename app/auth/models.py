"""인증용 SQLAlchemy 모델."""

from __future__ import annotations

import os
from pathlib import Path

from sqlalchemy import Boolean, Column, Integer, String, create_engine
from sqlalchemy.orm import declarative_base, scoped_session, sessionmaker


def _default_auth_db() -> str:
    base_dir = Path(os.getenv("APP_BASE_DIR") or Path(__file__).resolve().parents[2])
    db_path = base_dir / "data" / "auth.sqlite"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{db_path.as_posix()}"


AUTH_DATABASE_URL = os.getenv("AUTH_DATABASE_URL", _default_auth_db())
ENGINE = create_engine(
    AUTH_DATABASE_URL,
    connect_args={"check_same_thread": False}
    if AUTH_DATABASE_URL.startswith("sqlite")
    else {},
)
SessionLocal = scoped_session(sessionmaker(bind=ENGINE, autoflush=False, autocommit=False))

Base = declarative_base()


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    username = Column(String(64), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    role = Column(String(16), default="user", nullable=False)
    is_active = Column(Boolean, default=True)


Base.metadata.create_all(ENGINE)

