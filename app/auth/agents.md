# app/auth - 인증/권한

## 역할

- 세션 기반 인증과 역할(관리자/일반 사용자) 관리를 담당하며 FastAPI 라우터에서 재사용할 의존성을 제공합니다.

## 파일 가이드

- `middleware.py`: `SessionMiddleware` 초기화, `get_current_user`, `require_user`, `require_admin` 의존성 제공
- `models.py`: SQLAlchemy 모델 정의
  - `User`: 사용자 계정
  - `LoginAttempt`: 로그인 시도 기록 (IP 차단용)
  - `BlockedIP`: 차단된 IP 주소 관리
- `routes.py`: FastAPI 엔드포인트
  - `/auth/login`: 로그인 (IP 차단 확인, 실패 횟수 추적)
  - `/auth/logout`: 로그아웃
  - `/auth/me`: 현재 사용자 정보
  - `/auth/users`: 사용자 목록/생성
  - `/auth/blocked-ips`: 차단 IP 목록 (관리자)
  - `/auth/unblock-ip`: IP 차단 해제 (관리자)
- `service.py`: `AuthService`로 로그인 검증, 기본 관리자 자동 생성, 사용자 CRUD, IP 차단 관리
- `__init__.py`: 익스포트 편의

## IP 차단 시스템

- **환경변수**
  - `MAX_LOGIN_ATTEMPTS`: 로그인 실패 허용 횟수 (기본 10)
  - `LOGIN_ATTEMPT_WINDOW_MINUTES`: 실패 횟수 계산 시간 창 (기본 30분)
- **동작 방식**
  1. 로그인 실패 시 `LoginAttempt` 테이블에 기록
  2. 시간 창 내 실패 횟수가 한계 초과 시 `BlockedIP`에 추가
  3. 차단된 IP는 로그인 시도 자체가 거부됨
  4. 관리자가 UI에서 수동 차단 해제 가능

## 유지보수 체크리스트

- 관리자 기본 계정은 `ADMIN_USERNAME`/`ADMIN_PASSWORD` 환경변수로 결정됩니다. 운영 환경에서는 강력한 값으로 교체하세요.
- 세션 암호화 키는 `.env`의 `WEB_SESSION_SECRET`을 사용하며, 값 변경 시 기존 세션이 무효화됩니다.
- DB 스키마를 바꿀 때 `models.py`와 마이그레이션(또는 초기화 스크립트)을 함께 조정하세요.
- 프록시/로드밸런서 환경에서는 `X-Forwarded-For` 또는 `X-Real-IP` 헤더가 올바르게 전달되는지 확인하세요.
