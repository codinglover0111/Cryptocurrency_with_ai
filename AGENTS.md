# Cryptocurrency_with_ai(CCA) - 프로젝트 가이드

> LangChain 1.0 / LangGraph 1.0 기반 멀티 에이전트 암호화폐 자동매매 시스템

## 프로젝트 구조

```text
├── app/                    # 핵심 애플리케이션 모듈
│   ├── agents/             # 멀티 에이전트 (Indicator/Pattern/Trend/Decision)
│   ├── auth/               # 세션 인증, IP 차단 시스템
│   ├── config/             # LLM/스케줄러/리스크 설정
│   ├── core/               # 심볼/숫자 포맷 유틸리티
│   ├── graph/              # LangGraph 워크플로 그래프
│   ├── opro/               # Adaptive-OPRO 프롬프트 최적화
│   ├── services/           # 마켓 데이터/저널링 서비스
│   ├── web/                # FastAPI 라우터 (admin/user)
│   └── workflows/          # 자동매매 워크플로 엔트리포인트
├── utils/                  # 공통 헬퍼 (Bybit, AI, 스토리지, 리스크)
├── static/                 # 프런트엔드 JS/CSS
├── templates/              # Jinja2 HTML 템플릿
├── docs/                   # 아키텍처 문서
├── main.py                 # 봇 엔트리포인트 (스케줄러)
└── webapp.py               # 웹 서버 엔트리포인트
```

## 에이전트 파이프라인

```text
Indicator Agent → Pattern Agent → Trend Agent → Decision Agent
                      │                │
                      └───────┬────────┘
                              ▼
                     Adaptive-OPRO Loop
```

| 에이전트  | 역할                                | LLM 유형 |
| --------- | ----------------------------------- | -------- |
| Indicator | RSI, MACD, 볼린저 등 기술 지표 분석 | 텍스트   |
| Pattern   | 캔들 패턴 감지 (비전 기반)          | 비전     |
| Trend     | 지지/저항, 추세 방향 판별           | 비전     |
| Decision  | LONG/SHORT/HOLD/STOP 최종 결정      | 텍스트   |

## 개발 규칙

### 1. 문서 우선 원칙

- **새 기능 추가 전 `agents.md` 먼저 갱신** → 구현 진행
- 각 폴더의 `agents.md`는 해당 모듈의 단일 진실 공급원(Single Source of Truth)

### 2. 설정 변경 시 동기화

- 에이전트 LLM 설정 변경 → `app/config/default_config.py` + 관리자 UI 함께 수정
- 스키마 변경 → `app/agents/schemas.py` + `app/graph/workflow.py` + `app/workflows/trading.py` 연동 확인
- API 응답 변경 → `static/*.js` + `templates/*.html` 동기화

### 3. LLM 호출 규칙

- 모든 LLM 호출은 **LangChain 1.0 인터페이스** 사용
- 구조화된 출력은 `with_structured_output` 활용 (JSON 파싱 로직 제거)
- 새 provider 추가 시 `app/graph/llm_factory.py` + `.env.sample` 업데이트

### 4. 인증/권한

- FastAPI 엔드포인트는 **role-based access control** 필수 적용
- `admin`: 설정 변경, 사용자 관리, 스케줄러 제어
- `user`: 대시보드/저널 조회만 가능

### 5. Bybit API 호출

- CCXT 통합 메서드 우선 사용
- 없을 경우 `privatePost` 폴백
- 에러 코드(110007, 110025, 110026, 110043) 별도 처리 로직 유지

### 6. 데이터 저장

- 런타임 설정: **DB 우선** → JSON 폴백 (`utils/storage.py`)
- 스케줄러 상태, 공유 분석 결과: `scheduler_state`, `shared_analysis` 테이블

## 주요 환경변수

| 변수                               | 설명                                                                |
| ---------------------------------- | ------------------------------------------------------------------- |
| `BYBIT_ENV`                        | demo / testnet / mainnet                                            |
| `BYBIT_API_KEY`, `BYBIT_SECRET`    | Bybit API 인증                                                      |
| `OPENAI_API_KEY`, `GEMINI_API_KEY` | LLM API 키                                                          |
| `AI_PROVIDER`                      | 미설정 시 OpenRouter → OpenAI → Anthropic → Gemini 순으로 자동 감지 |
| `ADMIN_USERNAME`, `ADMIN_PASSWORD` | 기본 관리자 계정                                                    |
| `WEB_SESSION_SECRET`               | 세션 암호화 키                                                      |
| `MAX_LOGIN_ATTEMPTS`               | IP 차단 임계값 (기본 10)                                            |
| `TRADING_SYMBOLS`                  | 거래 심볼 목록 (관리자 UI에서 DB 설정 우선)                         |
| `CORS_ALLOWED_ORIGINS`             | 쉼표 구분 허용 오리진(정확히 일치)                                  |
| `CORS_ALLOWED_ORIGIN_REGEX`        | 정규식 허용 오리진, 기본값 `https://.*\.up\.railway\.app`           |
| `PRODUCTION`                       | `1` 또는 `true` 설정 시 HTTPS 전용 세션 쿠키 활성화                 |
| `MYSQL_URL`                        | MySQL 연결 URL (예: `mysql+pymysql://user:pwd@host:3306/db`)        |
| `MYSQL_ROOT_PASSWORD`              | docker-compose MySQL root 비밀번호 (기본: `rootpass`)               |
| `MYSQL_DATABASE`                   | docker-compose MySQL 데이터베이스명 (기본: `crypto_trading`)        |
| `MYSQL_USER`                       | docker-compose MySQL 사용자 (기본: `crypto`)                        |
| `MYSQL_PASSWORD`                   | docker-compose MySQL 비밀번호 (기본: `cryptopass`)                  |
| `FORCE_SQLITE`                     | `1` 설정 시 MySQL 대신 SQLite 강제 사용                             |

## 리스크 설정 기본값

| 항목                          | 기본값 | 설명                 |
| ----------------------------- | ------ | -------------------- |
| `default_leverage`            | 5      | 기본 레버리지        |
| `max_loss_percent`            | 40     | 최대 손실 허용 %     |
| `position_allocation_percent` | 20     | 포지션당 최대 할당 % |

- confirm 단계 LLM 검증(`app/workflows/trading.py`)도 동일한 `max_loss_percent` 한도를 사용하므로 관리자 UI 또는 환경변수(`MAX_LOSS_PERCENT`, `MAX_LEVERAGED_LOSS_PERCENT`) 변경 시 즉시 반영됩니다.

## 컨테이너 빌드 가이드

- Docker 이미지는 `ghcr.io/astral-sh/uv:python3.12-alpine` 파생본을 기본으로 사용한다.
- uv CLI는 이미지에 포함되어 있으므로 별도 설치하지 말고 시스템 파이썬(`UV_SYSTEM_PYTHON=1`)에 직접 동기화한다.
- Alpine 환경 의존성은 `apk add --no-cache pkgconf python3-dev mariadb-dev build-base curl tzdata` 조합으로 통일한다.
- `requirements.txt`를 먼저 복사한 뒤 `uv pip install --system -r requirements.txt`로 의존성을 설치하고, 이후 애플리케이션 전체를 복사해 Docker 레이어 캐시를 유지한다.
- 런타임 명령은 `uv run ...` 형식으로 통일해 uv가 관리하는 환경을 항상 사용한다.
- **웹 서버(webapp.py)** 실행 시 Uvicorn에 `--proxy-headers --forwarded-allow-ips=*` 옵션을 추가하여 Railway 등 리버스 프록시 환경에서 HTTPS를 올바르게 감지하도록 한다.
- 자세한 권장 패턴은 uv 공식 가이드([docs.astral.sh](https://docs.astral.sh/uv/guides/integration/docker/#installing-a-project))를 따른다.

## 로컬 개발 환경 (docker-compose)

로컬에서 MySQL과 함께 전체 시스템을 테스트하려면:

```bash
# 전체 서비스 시작 (MySQL + Bot + Web)
docker-compose up -d

# 로그 확인
docker-compose logs -f

# 서비스 중지
docker-compose down

# 볼륨 포함 완전 삭제 (MySQL 데이터 포함)
docker-compose down -v
```

**기본 접속 정보:**

- 웹 UI: http://localhost:8000
- MySQL: `localhost:3306` (user: `crypto`, password: `cryptopass`, database: `crypto_trading`)

**MySQL 대신 SQLite 사용:**

`.env` 파일에 `FORCE_SQLITE=1`을 추가하면 MySQL 대신 SQLite를 사용합니다.

## 세션 및 쿠키 설정

- 프로덕션 환경(Railway 등)에서는 세션 쿠키가 **HTTPS 전용**(`https_only=True`)으로 설정됩니다.
- `RAILWAY_PUBLIC_DOMAIN`, `RAILWAY_ENVIRONMENT`, 또는 `PRODUCTION=1` 환경변수가 있으면 자동으로 프로덕션 모드로 인식합니다.
- 로컬 개발 시에는 HTTP에서도 쿠키가 작동합니다.
- 세션 쿠키의 `SameSite` 속성은 `lax`로 설정되어 동일 사이트 요청에서 쿠키가 전송됩니다.

## 의존성 주의사항

- `passlib[bcrypt]==1.7.4`는 최신 `bcrypt` 4.2.x에서 제거된 `__about__` 메타데이터에 의존한다. 런타임 오류를 방지하기 위해 `bcrypt==4.1.2`로 고정한다.

## 즉시 실행 기능

관리자 대시보드의 스케줄러 섹션에서 스케줄러 주기를 기다리지 않고 즉시 분석을 실행할 수 있습니다.

### 기능

- **전체 심볼 즉시 실행**: 설정된 모든 거래 심볼에 대해 분석 실행
- **특정 심볼 실행**: 선택한 심볼에 대해서만 분석 실행

### API 엔드포인트

- `POST /admin/run-now`: 전체 심볼 즉시 실행 (백그라운드 스레드)
- `POST /admin/run-symbol`: 특정 심볼 즉시 실행 (백그라운드 스레드)

### 관련 파일

- `app/web/admin.py` - 즉시 실행 API 엔드포인트
- `templates/admin.html` - 즉시 실행 버튼 및 심볼 선택 모달
- `static/admin.js` - 즉시 실행 함수 및 이벤트 핸들러

## 에이전트 분석 모달

공개/관리자 대시보드의 "최근 활동" 섹션에서 항목을 클릭하면 4개 에이전트(Indicator, Pattern, Trend, Decision)의 분석 보고서를 모달로 확인할 수 있습니다.

### 기능

- **탭 UI**: Indicator / Pattern / Trend / Decision 4개 탭으로 구분
- **마크다운 렌더링**: 에이전트 분석 텍스트가 마크다운으로 렌더링됨
- **전문 저장**: 모든 에이전트 분석 결과가 저널 `meta.agents` 필드에 저장됨
- **스킵 로그에도 노출**: confirm 단계 거부나 TP/SL 업데이트 실패와 같이 `_record_skip()`이 호출된 경우에도 `meta.agents`가 함께 기록되어 모달에서 최근 분석을 잃지 않습니다.
- **표시/숨김 규칙**: `static/style.css`의 `.modal-backdrop`은 `hidden` 속성 토글만으로 제어하므로 JS에서는 `.active` 클래스를 다루지 않습니다.
- **심볼 라벨**: 헤더에 `에이전트 분석 보고서(BTCUSDT)` 형식으로 선택한 심볼을 노출해 어떤 보고서를 보는지 즉시 식별합니다.
- **모바일 대응**: 640px 이하에서는 모달이 전체 화면을 차지하고 탭/카드가 줄바꿈되어 모바일에서도 동일 동작을 제공합니다.

### 저널 메타 데이터 구조

```json
{
  "decision": { ... },
  "agents": {
    "indicator": { "rsi": 45.2, "macd_signal": "bullish", "summary": "..." },
    "pattern": { "patterns_found": ["hammer"], "analysis": "..." },
    "trend": { "trend_direction": "uptrend", "analysis": "..." },
    "decision": { "status": "long", "explain": "..." }
  }
}
```

### 관련 파일

- `app/workflows/trading.py` - 에이전트 결과 수집 및 저널 저장
- `templates/index.html` - 공개 대시보드 모달 UI
- `static/admin.js` - 관리자 대시보드 모달 로직
- `static/admin.css` - 모달 스타일
- `static/app.js` - 마크다운 렌더링 함수

## 폴더별 상세 가이드

각 폴더의 세부 내용은 해당 폴더의 `agents.md` 참조:

- `app/agents.md` - 앱 전체 운영 가이드
- `app/agents/agents.md` - 멀티 에이전트 상세
- `app/auth/agents.md` - 인증/IP 차단 시스템
- `app/config/agents.md` - 설정 관리
- `app/graph/agents.md` - LangGraph 워크플로
- `app/opro/agents.md` - Adaptive-OPRO
- `app/workflows/agents.md` - 트레이딩 워크플로
- `utils/agents.md` - 공통 헬퍼
- `docs/agents.md` - 전체 아키텍처 문서

---

> 문서를 최신 상태로 유지하지 않으면 구현이 중단됩니다.
