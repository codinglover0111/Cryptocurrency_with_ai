# Cryptocurrency_with_ai (CCA) - 프로젝트 가이드

> LangChain 1.0 / LangGraph 1.0 기반 멀티 에이전트 암호화폐 자동매매 시스템  
> 프런트엔드: Next.js(App Router) + Supabase Auth, 백엔드: FastAPI/LangGraph

## 프로젝트 구조

```text
├── server/                         # FastAPI/LangGraph 백엔드
│   ├── app/                        # 에이전트·워크플로·서비스·라우터
│   ├── utils/                      # Bybit/AI/스토리지/리스크 헬퍼
│   ├── static/                     # JS/CSS (레거시 대시보드)
│   ├── templates/                  # Jinja2 템플릿 (레거시)
│   ├── docs/                       # 아키텍처 문서
│   ├── main.py                     # 스케줄러 엔트리포인트
│   └── webapp.py                   # FastAPI 엔트리포인트
├── frontend/                       # Next.js App Router UI (Supabase 인증, API 프록시)
│   ├── app/                        # 페이지·라우트 핸들러
│   ├── app/api/proxy/[...path]/    # FastAPI 프록시 (Authorization: Bearer <supabase_jwt>)
│   ├── middleware.ts               # /admin 보호 라우트 미들웨어
│   ├── lib/supabase*               # Supabase 클라이언트 헬퍼
│   ├── public/                     # 정적 에셋
│   └── agents.md                   # 프런트엔드 단일 진실 문서
├── Dockerfile.bot                  # bot 서비스 (server/main.py)
├── Dockerfile.web                  # web 서비스 (server/webapp.py)
├── docker-compose.yml              # 로컬 개발 (MySQL + bot + web)
└── AGENTS.md                       # 최상위 가이드 (본 문서)
```

## 에이전트 파이프라인

```
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

프롬프트는 DB(`agent_prompts`) 우선, 폴백은 `server/app/agents/prompts.py`.

## 개발 규칙

1) **문서 우선**: 새 기능 추가 전 해당 폴더의 `agents.md`를 먼저 업데이트.  
2) **설정 동기화**: LLM/스키마/API 응답 변경 시 백엔드·프런트엔드·UI를 함께 수정.  
3) **LLM 호출**: LangChain 1.0 인터페이스 사용, 구조화 출력은 `with_structured_output`.  
4) **권한**: 모든 FastAPI 라우터는 `require_user`/`require_admin`(Supabase JWT) 필수.  
5) **Bybit**: CCXT 우선, 오류 코드(110007, 110025, 110026, 110043)별 처리 유지.  
6) **데이터 저장**: DB 우선 → JSON 폴백 (`utils/storage.py`), `scheduler_state`, `shared_analysis` 테이블 유지.

## 인증/권한 (Supabase + Next.js 프록시)

- Next.js가 Supabase Auth(이메일/비밀번호 기본, OAuth 선택)로 로그인/세션을 관리.  
- Next.js API 라우트 `app/api/proxy/[...path]`가 세션을 검증하고 FastAPI로 프록시(Authorization: Bearer <supabase_jwt>).  
- FastAPI는 Supabase JWKS로 토큰을 재검증하고 `role` 클레임(`app_metadata.role` → `user_metadata.role` → 기본 user)으로 RBAC를 적용.  
- 브라우저는 FastAPI를 직접 호출하지 않는다.

## 주요 환경변수 (백엔드 server/.env.sample)

| 변수                        | 설명                                                         |
| --------------------------- | ------------------------------------------------------------ |
| `BYBIT_ENV`                 | demo / testnet / mainnet                                     |
| `BYBIT_API_KEY`, `BYBIT_SECRET` | Bybit API 인증                                           |
| `OPENAI_API_KEY`, `GEMINI_API_KEY`, `OPENROUTER_API_KEY` | LLM API 키                |
| `SUPABASE_URL`              | Supabase 프로젝트 URL                                        |
| `SUPABASE_ANON_KEY`         | Supabase anon 키 (프런트엔드)                               |
| `SUPABASE_SERVICE_ROLE_KEY` | Supabase 서비스 롤 키 (백엔드 검증/관리)                    |
| `SUPABASE_JWKS_URL`         | Supabase JWKS (`https://<project>.supabase.co/auth/v1/jwks`) |
| `API_KEY_ENCRYPTION_KEY`    | API 키 암호화용 키 (Base64 urlsafe 32바이트)                 |
| `TRADING_SYMBOLS`           | 거래 심볼 목록 (DB 설정 우선)                               |
| `CORS_ALLOWED_ORIGINS`      | 쉼표 구분 허용 오리진                                       |
| `CORS_ALLOWED_ORIGIN_REGEX` | 정규식 허용 오리진 (기본 `https://.*\.up\.railway\.app`)    |
| `MYSQL_URL` / `SQLITE_PATH` | DB 연결 (FORCE_SQLITE=1로 SQLite 강제)                      |

프런트엔드 예시(`frontend/.env.example`):  
`NEXT_PUBLIC_SUPABASE_URL`, `NEXT_PUBLIC_SUPABASE_ANON_KEY`, `SUPABASE_SERVICE_ROLE_KEY`, `SUPABASE_JWKS_URL`, `FASTAPI_BASE_URL`.

## 컨테이너 빌드 가이드

- 베이스 이미지: `ghcr.io/astral-sh/uv:python3.12-alpine` (uv 내장, `UV_SYSTEM_PYTHON=1`).  
- 의존성 설치: `apk add --no-cache pkgconf python3-dev mariadb-dev build-base curl tzdata`.  
- `server/requirements.txt` 복사 후 `uv pip install --system -r requirements.txt`, 이어서 `server/` 전체 복사.  
- web 실행: `uv run uvicorn webapp:app --proxy-headers --forwarded-allow-ips=*`.  
- bot 실행: `uv run python main.py`.

## 로컬 개발 (docker-compose)

```bash
docker-compose up -d      # MySQL + bot + web
docker-compose logs -f
docker-compose down       # 종료
docker-compose down -v    # 볼륨 포함 삭제
```

- 웹 UI: http://localhost:8000  
- MySQL: localhost:3306 (user: crypto / pass: cryptopass / db: crypto_trading)  
- SQLite 사용 시 `.env`에 `FORCE_SQLITE=1`.

## 주요 기능 메모

- 스케줄러 제어/일시중단 및 즉시 실행: `/admin/run-now`, `/admin/run-symbol`.  
- 리스크 설정/프롬프트 관리/모델 선택: `/admin/*` API.  
- 에이전트 분석 모달: Indicator/Pattern/Trend/Decision 탭, 마크다운 렌더링, 모바일 전체 화면 대응.  
- 런타임 설정/스케줄러 상태/공유 분석 결과: DB 우선 저장, 실패 시 JSON 폴백.
