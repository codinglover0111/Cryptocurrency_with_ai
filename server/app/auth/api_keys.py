"""API 키 저장 및 관리 모듈."""

from __future__ import annotations

import base64
import hashlib
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from cryptography.fernet import Fernet
from sqlalchemy import Column, DateTime, Integer, String, Text, Boolean

from app.auth.models import Base, ENGINE, SessionLocal


# 암호화 키 생성 (환경변수 또는 자동 생성)
def _get_encryption_key() -> bytes:
    """암호화 키를 가져오거나 생성합니다."""
    key_env = os.getenv("API_KEY_ENCRYPTION_KEY")
    if key_env:
        # 환경변수에서 키 로드
        try:
            return base64.urlsafe_b64decode(key_env)
        except Exception:
            pass

    # Supabase 서비스 롤 키 또는 기본값 기반 파생
    secret = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv(
        "API_KEY_FALLBACK_SECRET", "change-me"
    )
    # 32바이트 키 생성 (Fernet 요구사항)
    key_hash = hashlib.sha256(secret.encode()).digest()
    return base64.urlsafe_b64encode(key_hash)


def _get_cipher() -> Fernet:
    """Fernet 암호화 객체를 반환합니다."""
    return Fernet(_get_encryption_key())


def encrypt_value(value: str) -> str:
    """값을 암호화합니다."""
    if not value:
        return ""
    cipher = _get_cipher()
    encrypted = cipher.encrypt(value.encode("utf-8"))
    return base64.urlsafe_b64encode(encrypted).decode("utf-8")


def decrypt_value(encrypted_value: str) -> str:
    """암호화된 값을 복호화합니다."""
    if not encrypted_value:
        return ""
    try:
        cipher = _get_cipher()
        decoded = base64.urlsafe_b64decode(encrypted_value.encode("utf-8"))
        decrypted = cipher.decrypt(decoded)
        return decrypted.decode("utf-8")
    except Exception:
        return ""


class ApiKey(Base):
    """API 키 저장 테이블."""

    __tablename__ = "api_keys"

    id = Column(Integer, primary_key=True)
    provider = Column(String(64), nullable=False)  # bybit, openai, gemini, etc.
    key_type = Column(String(64), nullable=False)  # api_key, api_secret, etc.
    environment = Column(
        String(32), default="default"
    )  # default, testnet, demo, mainnet
    encrypted_value = Column(Text, nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )


# 테이블 생성
Base.metadata.create_all(ENGINE)


class ApiKeyService:
    """API 키 관리 서비스."""

    # 지원하는 Provider와 키 타입 정의
    PROVIDERS = {
        "bybit": {
            "name": "Bybit",
            "environments": ["demo", "testnet", "mainnet"],
            "key_types": ["api_key", "api_secret"],
        },
        "openai": {
            "name": "OpenAI",
            "environments": ["default"],
            "key_types": ["api_key"],
        },
        "openrouter": {
            "name": "OpenRouter",
            "environments": ["default"],
            "key_types": ["api_key"],
        },
        "gemini": {
            "name": "Gemini",
            "environments": ["default"],
            "key_types": ["api_key"],
        },
        "anthropic": {
            "name": "Anthropic",
            "environments": ["default"],
            "key_types": ["api_key"],
        },
    }

    @staticmethod
    def set_key(
        provider: str, key_type: str, value: str, environment: str = "default"
    ) -> bool:
        """API 키를 저장하거나 업데이트합니다."""
        session = SessionLocal()
        try:
            # 기존 키 검색
            existing = (
                session.query(ApiKey)
                .filter(
                    ApiKey.provider == provider,
                    ApiKey.key_type == key_type,
                    ApiKey.environment == environment,
                )
                .first()
            )

            encrypted = encrypt_value(value)

            if existing:
                existing.encrypted_value = encrypted
                existing.is_active = True
                existing.updated_at = datetime.now(timezone.utc)
            else:
                new_key = ApiKey(
                    provider=provider,
                    key_type=key_type,
                    environment=environment,
                    encrypted_value=encrypted,
                    is_active=True,
                )
                session.add(new_key)

            session.commit()
            return True
        except Exception as e:
            session.rollback()
            print(f"Error saving API key: {e}")
            return False
        finally:
            session.close()

    @staticmethod
    def get_key(
        provider: str, key_type: str, environment: str = "default"
    ) -> Optional[str]:
        """API 키를 가져옵니다. DB에 없으면 환경변수에서 폴백합니다."""
        session = SessionLocal()
        try:
            key_record = (
                session.query(ApiKey)
                .filter(
                    ApiKey.provider == provider,
                    ApiKey.key_type == key_type,
                    ApiKey.environment == environment,
                    ApiKey.is_active == True,
                )
                .first()
            )

            if key_record and key_record.encrypted_value:
                decrypted = decrypt_value(key_record.encrypted_value)
                if decrypted:
                    return decrypted

            # DB에 없으면 환경변수에서 폴백
            return ApiKeyService._get_env_fallback(provider, key_type, environment)
        except Exception as e:
            print(f"Error getting API key: {e}")
            return ApiKeyService._get_env_fallback(provider, key_type, environment)
        finally:
            session.close()

    @staticmethod
    def _get_env_fallback(
        provider: str, key_type: str, environment: str
    ) -> Optional[str]:
        """환경변수에서 API 키를 가져옵니다 (폴백)."""
        provider_upper = provider.upper()
        env_upper = environment.upper() if environment != "default" else ""
        key_type_upper = key_type.upper()

        # 환경변수 이름 패턴 시도
        patterns = []
        if env_upper:
            patterns.append(f"{provider_upper}_{env_upper}_{key_type_upper}")
            patterns.append(f"{env_upper}_{provider_upper}_{key_type_upper}")
        patterns.append(f"{provider_upper}_{key_type_upper}")

        for pattern in patterns:
            value = os.getenv(pattern)
            if value:
                return value

        return None

    @staticmethod
    def delete_key(provider: str, key_type: str, environment: str = "default") -> bool:
        """API 키를 삭제(비활성화)합니다."""
        session = SessionLocal()
        try:
            key_record = (
                session.query(ApiKey)
                .filter(
                    ApiKey.provider == provider,
                    ApiKey.key_type == key_type,
                    ApiKey.environment == environment,
                )
                .first()
            )

            if key_record:
                key_record.is_active = False
                key_record.encrypted_value = ""
                key_record.updated_at = datetime.now(timezone.utc)
                session.commit()
                return True
            return False
        except Exception as e:
            session.rollback()
            print(f"Error deleting API key: {e}")
            return False
        finally:
            session.close()

    @staticmethod
    def list_keys() -> List[Dict[str, Any]]:
        """저장된 모든 API 키 목록을 반환합니다 (값은 마스킹)."""
        session = SessionLocal()
        try:
            keys = session.query(ApiKey).filter(ApiKey.is_active == True).all()
            result = []
            for key in keys:
                # 값의 일부만 표시
                decrypted = decrypt_value(key.encrypted_value)
                masked = ""
                if decrypted:
                    if len(decrypted) > 8:
                        masked = decrypted[:4] + "****" + decrypted[-4:]
                    else:
                        masked = "****"

                result.append(
                    {
                        "provider": key.provider,
                        "key_type": key.key_type,
                        "environment": key.environment,
                        "masked_value": masked,
                        "has_value": bool(decrypted),
                        "updated_at": key.updated_at.isoformat()
                        if key.updated_at
                        else None,
                    }
                )
            return result
        except Exception as e:
            print(f"Error listing API keys: {e}")
            return []
        finally:
            session.close()

    @staticmethod
    def get_status() -> Dict[str, Any]:
        """각 Provider별 API 키 설정 상태를 반환합니다."""
        session = SessionLocal()
        try:
            status = {}
            for provider, config in ApiKeyService.PROVIDERS.items():
                provider_status = {
                    "name": config["name"],
                    "environments": {},
                }
                for env in config["environments"]:
                    env_status = {}
                    for key_type in config["key_types"]:
                        # DB에서 확인
                        key_record = (
                            session.query(ApiKey)
                            .filter(
                                ApiKey.provider == provider,
                                ApiKey.key_type == key_type,
                                ApiKey.environment == env,
                                ApiKey.is_active == True,
                            )
                            .first()
                        )

                        has_db_key = bool(key_record and key_record.encrypted_value)
                        has_env_key = bool(
                            ApiKeyService._get_env_fallback(provider, key_type, env)
                        )

                        env_status[key_type] = {
                            "configured": has_db_key or has_env_key,
                            "source": "db"
                            if has_db_key
                            else ("env" if has_env_key else None),
                        }
                    provider_status["environments"][env] = env_status
                status[provider] = provider_status
            return status
        except Exception as e:
            print(f"Error getting API key status: {e}")
            return {}
        finally:
            session.close()


# 편의 함수들
def get_bybit_keys(environment: str = "default") -> Dict[str, Optional[str]]:
    """Bybit API 키와 시크릿을 가져옵니다."""
    return {
        "api_key": ApiKeyService.get_key("bybit", "api_key", environment),
        "api_secret": ApiKeyService.get_key("bybit", "api_secret", environment),
    }


def get_llm_key(provider: str) -> Optional[str]:
    """LLM Provider의 API 키를 가져옵니다."""
    return ApiKeyService.get_key(provider.lower(), "api_key", "default")
