# Cryptocurrency_with_ai(CCA) - 멀티 에이전트 암호화폐 트레이딩 시스템

> **경고**: 본 프로젝트는 교육/연구용 레퍼런스입니다. 실거래 손익은 보장되지 않습니다. 자동 매매를 실행하기 전에 백테스트와 리스크 검토를 반드시 수행하세요.

## 핵심 기술

- LangChain 1.0 + LangGraph 1.0: 멀티 에이전트 그래프 오케스트레이션
- Adaptive-OPRO: 실적 기반 프롬프트 최적화 루프
- FastAPI + APScheduler: API 및 스케줄러
- CCXT (Bybit): 선물/스팟 데이터와 주문 처리
- Vision LLM 지원: 패턴/추세 에이전트에서 캔들 이미지 분석

## 프로젝트 구조 (요약)

```text
app/
├─ agents/        # Indicator, Pattern, Trend, Decision 에이전트
├─ auth/          # 세션 기반 인증/권한
├─ config/        # LLM/스케줄러/OPRO 기본 설정
├─ core/          # 공통 심볼/포맷 유틸
├─ graph/         # LangGraph 워크플로 정의
├─ opro/          # Adaptive-OPRO 모듈
├─ services/      # 마켓/저널 서비스
├─ web/           # FastAPI 라우터 (admin/user)
└─ workflows/     # 트레이딩 자동화 워크플로
docs/            # 설계 문서
static/          # JS/CSS 에셋
templates/       # Jinja2 템플릿
utils/           # 거래/AI 헬퍼 모듈
```

- 각 폴더에 `agents.md`를 두어 책임과 유지보수 포인트를 기록했습니다.

## 빠른 시작

### 1) 환경 변수

- 루트에 `.env`를 생성하고 필요한 키를 채웁니다 (`.env.sample` 참고).

```bash
BYBIT_API_KEY=...
BYBIT_API_SECRET=...
BYBIT_ENV=demo               # demo | testnet | mainnet

GEMINI_API_KEY=...
OPENAI_API_KEY=...
OPENROUTER_API_KEY=...

TRADING_SYMBOLS=BTCUSDT,ETHUSDT,SOLUSDT
MAX_ALLOC_PERCENT=20
DEFAULT_LEVERAGE=5
MAX_LOSS_PERCENT=80

SQLITE_PATH=data/trading.sqlite
# MYSQL_URL=mysql+pymysql://user:pass@host:3306/db

WEB_SESSION_SECRET=change-me
ADMIN_USERNAME=admin
ADMIN_PASSWORD=admin123
```

- 단일 LLM 워크플로우는 더 이상 제공하지 않으며 멀티 에이전트 그래프만 실행됩니다.

### 2) 로컬 실행

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
pip install -r requirements.txt
python main.py
```

### 3) Docker

```bash
docker compose up -d --build
```

## 멀티 에이전트 파이프라인

```text
Indicator Agent -> Pattern Agent -> Trend Agent -> Decision Agent
                     |                |
                     +-----+----------+
                           v
                    Adaptive-OPRO Loop
```

- **Indicator**: RSI, MACD, Bollinger 등 기술 지표 요약 (LLM: 텍스트)
- **Pattern**: 캔들 차트 패턴/강도 분석, 비전 모델 필요
- **Trend**: 추세/지지·저항 추정, 비전 모델 필요
- **Decision**: LONG/SHORT/HOLD/STOP 판단, 포지션 크기·TP/SL 산출

## 웹 UI / API

- 기본 주소: `http://localhost:8000`
- 관리자: 에이전트 모델/파라미터, 스케줄러 주기 설정, OPRO 상태 확인
- 사용자: 계정/세션, 현재 전략 상태 조회
- 주요 엔드포인트
  - `POST /auth/login`, `GET /auth/me`
  - `GET|POST /admin/agent-config`, `GET|POST /admin/scheduler`
  - `GET /user/settings`

## 문서

- `docs/agents.md`: 멀티 에이전트 아키텍처 상세
- `docs/architecture.md`: 전체 시스템 구조
- 각 폴더의 `agents.md`: 폴더별 책임/파일 가이드

## 개발 규칙

1. 기능 추가 시 `docs/agents.md`와 관련 `agents.md`를 먼저 업데이트합니다.
2. LLM/스케줄러 변경 시 `app/config/default_config.py`와 관리자 UI가 같은 값을 읽도록 맞춥니다.
3. LangChain 1.0의 구조화 출력(`with_structured_output`)을 사용해 파싱 로직을 단순화합니다.
4. FastAPI 라우터는 역할 기반 접근 제어를 유지합니다.
5. Bybit 호출은 CCXT 래퍼를 우선 사용하고, 직접 호출 시 응답 코드를 체크합니다.

## 면책

이 저장소는 교육/연구용 레퍼런스입니다. 자동 매매나 실거래 적용 전 백테스트와 리스크 검토를 반드시 수행하고, 모든 손익 책임은 사용자에게 있습니다.
