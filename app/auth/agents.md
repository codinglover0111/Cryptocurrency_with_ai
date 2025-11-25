# app/auth - 인증/권한

## 역할

- 세션 기반 인증과 역할(관리자/일반 사용자) 관리를 담당하며 FastAPI 라우터에서 재사용할 의존성을 제공합니다.

## 파일 가이드

- `middleware.py`: `SessionMiddleware` 초기화, `get_current_user`, `require_user`, `require_admin` 의존성 제공
- `models.py`: SQLAlchemy `User` 모델과 `SessionLocal` 설정
- `routes.py`: `/auth/login`, `/auth/logout`, `/auth/me`, `/auth/users` 등의 FastAPI 엔드포인트
- `service.py`: `AuthService`로 로그인 검증, 기본 관리자 자동 생성, 사용자 CRUD
- `__init__.py`: 익스포트 편의

## 유지보수 체크리스트

- 관리자 기본 계정은 `ADMIN_USERNAME`/`ADMIN_PASSWORD` 환경변수로 결정됩니다. 운영 환경에서는 강력한 값으로 교체하세요.
- 세션 암호화 키는 `.env`의 `WEB_SESSION_SECRET`을 사용하며, 값 변경 시 기존 세션이 무효화됩니다.
- DB 스키마를 바꿀 때 `models.py`와 마이그레이션(또는 초기화 스크립트)을 함께 조정하세요.
