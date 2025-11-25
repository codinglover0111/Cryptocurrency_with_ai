# QuantAgent 멀티 에이전트 아키텍처

이 문서는 LangChain 1.0과 LangGraph 1.0을 기반으로 한 QuantAgent 시스템의
전체 설계 문서입니다. 모든 구현 이전에 본 문서의 요구사항이 충족되어야 하며,
향후 리팩토링 및 기능 확장의 단일 출처(Single Source of Truth)로 사용됩니다.

---

## 1. 시스템 개요

- **목표**: 횡보장을 포함한 다양한 시장 레짐에서 안정적으로 작동하는
  멀티 에이전트 기반 암호화폐 매매 의사결정 시스템 구축
- **핵심 기술**
  - LangChain 1.0 / LangGraph 1.0
  - Adaptive-OPRO (Adaptive Prompt Optimization by OPRO)
  - FastAPI + APScheduler 백엔드
  - Bybit 거래 (CCXT 기반)
  - Vision LLM (패턴/추세 에이전트)
- **Python 버전**: 3.10 이상

---

## 2. 에이전트 파이프라인

```text
Indicator Agent → Pattern Agent → Trend Agent → Decision Agent
                     │                │
                     └───────┬────────┘
                             ▼
                    Adaptive-OPRO Loop
```

### 2.1 Indicator Agent (`app/agents/indicator_agent.py`)

- **입력**: OHLCV 데이터(4H/1H/15M), 포지션 요약, 저널 요약
- **LLM**: 텍스트 전용 (예: OpenAI GPT-4o-mini)
- **기능**
  - RSI, MACD, Stochastic, Williams %R, Bollinger Band, ROC 등 계산
  - 과매수/과매도 구간 판별
  - 지표 기반 시그널 컨센서스 산출
- **출력 스키마**: `IndicatorResult` (Pydantic)

### 2.2 Pattern Agent (`app/agents/pattern_agent.py`)

- **입력**: 캔들 차트 이미지(4H/1H/15M), 인디케이터 요약
- **LLM**: 비전 모델 필수 (예: Gemini 2.0 Flash, GPT-4o)
- **기능**
  - 캔들 패턴 감지 (Doji, Hammer, Engulfing, Morning Star 등)
  - 패턴 강도 및 신뢰도 계산
  - 횡보장 전용 패턴 경고 (False breakouts 등)
- **출력 스키마**: `PatternResult`

### 2.3 Trend Agent (`app/agents/trend_agent.py`)

- **입력**: 지지/저항 시각화 이미지, Pattern/Indicator 출력
- **LLM**: 비전 모델 필수
- **기능**
  - 추세 방향(uptrend/downtrend/sideways) 판별
  - 지지/저항 레벨 및 채널 폭 추정
  - 횡보장 시 진입 구간/돌파 기준 정의
- **출력 스키마**: `TrendResult`

### 2.4 Decision Agent (`app/agents/decision_agent.py`)

- **입력**: 앞선 모든 에이전트 출력, 실시간 포지션, 리스크 설정
- **LLM**: 텍스트 모델 (OpenRouter, DeepSeek 등)
- **기능**
  - LONG/SHORT/HOLD/STOP 결정
  - 포지션 규모, 레버리지, TP/SL 산출
  - 횡보장 시 보수적 전략 자동 적용
- **출력 스키마**: `TradeDecision`

---

## 3. Adaptive-OPRO 시스템 (`app/opro/`)

### 3.1 목적

- 횡보장(Range-bound)에서 발생하는 오류(거짓 돌파, 과매수/과매도 반복 등)를
  줄이기 위해 실시간으로 프롬프트를 재학습/최적화

### 3.2 구성 요소

| 모듈                | 파일                 | 역할                                                         |
| ------------------- | -------------------- | ------------------------------------------------------------ |
| Regime Detector     | `regime_detector.py` | ADX, ATR, 볼린저 폭 등을 통해 시장 레짐 분류                 |
| Meta Prompt Manager | `meta_prompt.py`     | 이전 프롬프트 + 성과 + 레짐 정보를 결합한 메타 프롬프트 생성 |
| Optimizer           | `optimizer.py`       | OPRO 루프 실행, 최적 프롬프트 후보 생성                      |
| Scorer              | `scorer.py`          | ROI, Sharpe, Win-rate 기반 후보 평가                         |

### 3.3 동작 절차

1. 최근 N회의 거래 결과(ROI, MDD, 승률)와 시장 레짐을 수집
2. Meta-prompt 구성: `[과거 프롬프트, 점수, 시장 메타데이터]`
3. Optimizer LLM이 새 프롬프트 후보 생성
4. Scorer LLM이 백테스트/시뮬레이션 점수 부여
5. 상위 후보를 실제 Indicator/Decision 프롬프트로 적용

---

## 4. LLM 설정 및 구조화된 출력

- 에이전트별 LLM은 `app/config/default_config.py`에 정의하며,
  관리자 웹 UI에서 변경 가능
- LangChain 1.0의 `with_structured_output`과 `ToolStrategy`를 사용하여
  JSON 파싱 로직 제거
- 표준 출력 스키마는 `app/agents/schemas.py`에 정의

```python
class TradeDecision(BaseModel):
    status: Literal["long", "short", "hold", "stop"]
    tp: Optional[float]
    sl: Optional[float]
    leverage: Optional[float]
    buy_now: bool = False
    explain: str
```

---

## 5. 인증 및 권한 시스템 (`app/auth/`)

- **세션 기반 인증** (FastAPI `SessionMiddleware`)
- **역할**
  - `admin`: 에이전트 LLM 설정, 스케줄러 주기, 사용자 관리, 심볼 관리, 포지션 일괄 청산
  - `user`: 대시보드/저널 조회, 자동매매 상태 확인
- **저장소**: 기본 SQLite (SQLAlchemy) → 필요 시 MySQL로 확장
- **라우트**
  - `POST /auth/login`
  - `POST /auth/logout`
  - `GET /auth/me`

---

## 6. 웹 UI (`app/web/`, `templates/`, `static/`)

- **관리자 페이지**
  - 에이전트별 모델/파라미터 변경
  - APScheduler 주기(분 단위) 설정
  - Adaptive-OPRO 상태, 최근 프롬프트 히스토리 조회
- **사용자 페이지**
  - 잔고, 포지션, 최근 거래, 저널
  - 현재 전략/시장 레짐 설명

`webapp.py` API 확장:

- `GET /api/models`
- `GET /api/agent-config`
- `POST /api/agent-config`
- `POST /api/schedule`
- `POST /api/analyze` (매매 실행 없이 분석만 수행)

---

## 7. API 및 워크플로우

### 7.1 LangGraph StateGraph (`app/graph/workflow.py`)

- 노드: `indicator`, `pattern`, `trend`, `decision`, `adaptive_opro`
- `TradingState` (`app/agents/state.py`)를 통해 상태 공유
- 조건부 엣지:
  - 횡보장 감지 → Adaptive-OPRO 우선 실행
  - 기존 포지션 보유 중 → `decision` 노드에서 리스크 감소 루틴 호출

### 7.2 Trading Workflow (`app/workflows/trading.py`)

- `_gather_prompt_context` → `TradingGraph.run(symbol)`로 대체
- 결과 구조

```python
{
    "symbol": "BTC/USDT",
    "decision": TradeDecision,
    "state": TradingState,
    "prompt_trace": [...],  # OPRO 기록
}
```

---

## 8. Bybit 거래 모듈 (`utils/bybit_utils.py`)

> 참고: [CCXT Bybit 문서](https://docs.ccxt.com/#/exchanges/bybit)

### 8.1 환경 설정

| 모드    | 환경변수            | 설명                               |
| ------- | ------------------- | ---------------------------------- |
| Demo    | `BYBIT_ENV=demo`    | `api-demo.bybit.com` (가상 자금)   |
| Testnet | `BYBIT_ENV=testnet` | `api-testnet.bybit.com` (테스트넷) |
| Mainnet | `BYBIT_ENV=mainnet` | `api.bybit.com` (실거래)           |

### 8.2 주요 기능

- **레버리지 설정**: `set_leverage(symbol, leverage, margin_mode)`
- **마진 모드 설정**: `set_margin_mode(symbol, "cross"|"isolated")`
- **포지션 모드 설정**: `set_position_mode(hedged=True|False)`
  - `hedged=True`: Hedge mode (Long/Short 별도, `positionIdx` 필요)
  - `hedged=False`: One-way mode (단일 포지션)
- **TP/SL 업데이트**: `update_symbol_tpsl(symbol, take_profit, stop_loss, tpsl_mode)`
  - `tpsl_mode="Full"`: 전체 포지션에 적용
  - `tpsl_mode="Partial"`: 일부 포지션에 적용

### 8.3 에러 코드 처리

| 코드   | 의미                  | 처리                     |
| ------ | --------------------- | ------------------------ |
| 110007 | 마진 부족             | 수량 자동 축소 후 재시도 |
| 110025 | 포지션 모드 변경 불가 | 무시                     |
| 110026 | 마진 모드 변경 불가   | 무시                     |
| 110043 | 레버리지 변경 불가    | 무시                     |

### 8.4 시간 동기화

- `BYBIT_RECV_WINDOW_MS`: 서버 요청 허용 시간 창 (기본 15000ms)
- `BYBIT_TIME_SAFETY_MS`: 시간 동기화 안전 마진 (기본 500ms)
- 서버-로컬 시간 차이는 초기화 시 자동 측정 및 보정

---

## 9. 개발 규칙

1. 새로운 기능을 추가하기 전에 `agents.md`를 최신 상태로 유지한다.
2. 에이전트별 LLM 설정 변경 시 `default_config.py`와 웹 UI를 함께 수정한다.
3. Adaptive-OPRO가 비활성이라도 프롬프트 버전 히스토리를 저장한다.
4. 모든 LLM 호출은 LangChain 1.0 인터페이스를 사용한다.
5. FastAPI 엔드포인트는 role-based access control을 반드시 거친다.
6. Bybit API 호출 시 CCXT 통합 메서드를 우선 사용하고, 없을 경우 privatePost 폴백.

---

## 10. 테스트 및 검증

- **단위 테스트**
  - Regime Detector, Meta Prompt Builder, 각 에이전트 노드
- **통합 테스트**
  - TradingGraph end-to-end 실행 (mock Bybit, mock LLM)
  - Adaptive-OPRO 루프 (샘플 성과 데이터 입력)
- **수동 검증**
  - 관리자 UI에서 모델 변경 → 즉시 config 반영 확인
  - 횡보장 샘플 데이터로 HOLD/STOP 전략이 적용되는지 확인

---

문서를 최신 상태로 유지하지 않으면 구현이 중단되며,
모든 신규 기여자는 본 문서를 읽고 서명해야 합니다.
