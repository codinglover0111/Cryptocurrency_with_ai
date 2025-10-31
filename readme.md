## 안내

- **이 프로젝트는 절대적인 수익을 보장하지 않습니다.** 모든 투자 의사결정은 본인의 책임입니다.
- 자동매매를 실행하기 전에 백테스트와 종합적인 리스크 검토를 반드시 수행하세요.

## 프로젝트 구조

```
app/
  core/          # 공용 유틸리티(심볼 파싱 등)
  services/      # 데이터 수집·저널 등 도메인 서비스
  workflows/     # 트레이딩 자동화 파이프라인
main.py          # 스케줄러 진입점
webapp.py        # FastAPI 기반 웹 UI
utils/           # 거래소/AI/저장소 등 기존 래퍼 모듈
```

- `app/workflows/trading.py`는 한 사이클의 자동매매 흐름을 담당합니다.
  - 시장 컨텍스트 수집 → 프롬프트 구성 → AI 결정 파싱 → 주문 실행 → 결과 기록까지 단계별 함수로 나뉘어 있습니다.
  - 확인(Confirm) 단계가 별도 함수로 분리되어 있어, LLM이 제시한 TP/SL/가격을 재검증하고 필요 시 스킵하도록 했습니다.
- `app/services/journal.py`는 거래 리뷰와 저널 포맷팅 등을 담당합니다.
- `main.py`는 로깅 설정과 작업 스케줄링만을 책임지며, 나머지 로직은 `app` 패키지로 이동했습니다.

## 실행 방법

1. `.env`에 필요한 API 키와 환경변수를 설정합니다. (예시는 `env.example` 또는 아래 환경변수 설명 참고)
2. `uv` 기반 로컬 개발 워크플로우:

   ```bash
   make install          # uv pip install --python python3.11 -r requirements.txt
   make backend          # uv run --python python3.11 uvicorn webapp:app --reload
   make scheduler        # uv run --python python3.11 python -m main
   ```

   - `Makefile`의 `UV`/`PYTHON` 변수를 원하는 버전으로 덮어쓸 수 있습니다.
   - Next.js 프론트를 `next export`로 빌드했다면 `web/out`에 결과물을 두고 `make frontend`로 정적 번들을 서빙할 수 있습니다.

3. Docker Compose(uv 베이스 이미지)로 백엔드/스케줄러/프론트를 묶을 수 있습니다.

   ```bash
   docker compose up -d --build
   ```

   - `deploy/Dockerfile.backend`, `deploy/Dockerfile.scheduler`, `deploy/Dockerfile.web`는 모두 `ghcr.io/astral-sh/uv` 이미지를 사용합니다.
   - 프론트 서비스는 `web/out` 디렉터리에 export된 정적 산출물이 있을 때만 기동됩니다.

4. 단일 컨테이너가 필요하다면 `deploy/Dockerfile.backend`를 단독으로 빌드해 FastAPI UI를 배포할 수 있습니다.

## 주요 환경변수

```text
TESTNET=1
TRADING_SYMBOLS=XRPUSDT,WLDUSDT,ETHUSDT,BTCUSDT,SOLUSDT,DOGEUSDT
MAX_ALLOC_PERCENT=20
DEFAULT_LEVERAGE=5
MAX_LOSS_PERCENT=80
AVAILABLE_NOTIONAL_SAFETY=0.95

# AI (Gemini 기본 / OpenAI 호환 선택)
AI_PROVIDER=gemini
GEMINI_API_KEY=...
OPENAI_BASE_URL=...
OPENAI_API_KEY=...
OPENAI_MODEL=...

# 데이터 저장소 (MySQL 또는 SQLite 폴백)
MYSQL_URL=mysql+pymysql://bot:botpass@db:3306/cryptobot
SQLITE_PATH=data/trading.sqlite

# Supabase (선택 구성)
SUPABASE_URL=https://<project>.supabase.co
SUPABASE_ANON_KEY=...
SUPABASE_SERVICE_ROLE_KEY=...
SUPABASE_JWT_SECRET=...
SUPABASE_PROJECT_ID=<project-id>
```

## 스케줄러 동작

- `main.py`는 `schedule` 패키지를 사용하여 5분 주기로 다음 작업을 수행합니다.
  1. `automation_for_symbol`을 심볼 리스트 순회 호출
  2. `run_loss_review`로 최근 손실 포지션 리뷰 생성
- 각 사이클의 디버그 정보와 예외는 `trading.log`에 기록되며, FastAPI UI에서도 실시간 모니터링이 가능합니다.

## 웹 UI (FastAPI)

- 루트(`GET /`)와 각종 JSON API(`/health`, `/status`, `/leverage`, `/stats`, `/stats_range`, `/close_all`, `/symbols`)를 제공합니다.
- 정적 파일은 `static/`, 템플릿은 `templates/`에 위치합니다.
- `parse_trading_symbols()`를 사용해 서버와 자동매매가 동일한 심볼 구성을 공유합니다.
- UI에서는 심볼 정보·계정 개요·통계·저널 기록을 확인할 수 있습니다.

## 참고 문서

- `docs/architecture.md`: 최신 모듈 구조와 데이터 흐름을 정리한 문서입니다.
- `docs/deployment.md`: uv 기반 Docker/배포 파이프라인과 Supabase 초기화 절차를 정리했습니다.
- `utils/` 디렉터리의 모듈은 기존 레거시 기능을 보존하기 위해 유지되며, 새로운 서비스 계층에서 랩핑하여 사용합니다.

## 다음 단계 제안

1. 스테이징 환경에서 `.env`를 구성한 뒤, `python3 main.py`로 한 사이클 이상 실행해 로그와 DB 기록을 확인하세요.
2. `docs/architecture.md`를 참고하여 추가 모듈 분리 또는 테스트 코드 도입 계획을 수립하세요.
