# 배포 가이드

## Docker 이미지

- `deploy/Dockerfile.backend`: FastAPI 백엔드(UI 포함) 컨테이너.
  - `uv pip install --system`으로 의존성을 설치하고 `uv run uvicorn ...`으로 구동합니다.
- `deploy/Dockerfile.scheduler`: `main.py` 스케줄러 전용 컨테이너.
- `deploy/Dockerfile.web`: Next.js `next export` 결과물을 FastAPI `StaticFiles`로 제공하는 경량 웹 컨테이너.
  - 기본 경로는 `/srv/app/web`이며, `NEXT_BUILD_DIR` 환경변수로 변경할 수 있습니다.

모든 Dockerfile은 `ghcr.io/astral-sh/uv:python3.11-bookworm-slim` 이미지를 사용하며, uv 실행 플로우를 유지합니다.

## Railway 설정

- `railway.toml`과 `Procfile`을 함께 사용합니다.
- 프로세스 분리
  - `backend`: `uv run uvicorn webapp:app --host 0.0.0.0 --port $PORT`
  - `scheduler`: `uv run python -m main`
- 빌드 설정은 `deploy/Dockerfile.backend`, `deploy/Dockerfile.scheduler`를 각각 사용합니다.
- 기본 환경변수는 `railway.toml`의 `[variables]` 섹션에 정리되어 있습니다. 반드시 실제 값으로 대체하세요.

### Supabase 초기화

- `deploy/supabase/init.sql`을 Supabase SQL 에디터에 실행하여 `trades`, `journals` 테이블 및 인덱스를 생성합니다.
- 필요한 환경변수
  - `SUPABASE_URL`
  - `SUPABASE_ANON_KEY`
  - `SUPABASE_SERVICE_ROLE_KEY`
  - `SUPABASE_JWT_SECRET`
  - `SUPABASE_PROJECT_ID`

## 로컬 개발

- `Makefile`
  - `make install` → uv를 통해 의존성 설치
  - `make backend` / `make scheduler` → 각각 FastAPI, 스케줄러 실행
  - `make frontend` → `web/out` 디렉터리에 있는 Next.js export 번들을 로컬에서 서빙
  - `make compose-up` / `make compose-down` → Docker Compose를 uv 베이스 이미지로 실행/중지
- Compose는 uv 기반 컨테이너만을 묶도록 재구성되어 있으며, 프론트엔드는 선택적으로 `web/out` 볼륨을 마운트합니다.

## 프론트엔드 분리 배포

- Next.js 프로젝트는 Vercel(또는 별도 CI/CD)에서 관리합니다.
- `next export` 결과(`out` 디렉터리)를 아티팩트로 보관한 뒤:
  - Vercel에서 정적 배포하거나,
  - `deploy/Dockerfile.web` 기반 컨테이너에 포함시켜 Railway 등에서 서빙할 수 있습니다.
