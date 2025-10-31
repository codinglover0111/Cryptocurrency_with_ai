# 시스템 아키텍처 문서

## 개요

`Cryptocurrency_with_ai`는 AI 기반 암호화폐 자동매매 시스템입니다. OpenAI SDK를 사용하여 시장 데이터를 분석하고, 트레이딩 결정을 내리며, 포지션을 자동으로 관리합니다.

## 주요 특징

- **AI 기반 의사결정**: OpenAI 호환 API를 통해 시장 데이터를 분석하고 트레이딩 결정을 수행
- **다중 타임프레임 분석**: 15분, 1시간, 4시간 OHLCV 데이터와 RSI 지표 활용
- **리스크 관리**: 포지션 사이즈 자동 계산, TP/SL 설정, 최대 손실 제한
- **자동 학습**: 손실 거래에 대한 AI 리뷰를 통해 지속적 개선
- **웹 대시보드**: FastAPI 기반 웹 인터페이스로 실시간 모니터링

## 시스템 플로우

### 1. 스케줄러 시작 (`scheduler.py`)

```
main.py → scheduler.py → run_scheduler()
```

- 매시각 58분, 13분, 28분, 43분에 자동매매 루틴 실행
- 5분 주기로 손실 리뷰 작업 실행
- 심볼별로 독립적으로 작업 처리

### 2. 자동매매 워크플로우 (`app/workflows/trading.py`)

```
automation_for_symbol()
├── _init_dependencies()          # 의존성 초기화
├── _gather_prompt_context()     # 프롬프트 컨텍스트 수집
│   ├── OHLCV 데이터 수집 (15m/1h/4h)
│   ├── RSI 계산 (15m/1h/4h)
│   ├── 현재 포지션 정보
│   ├── 저널 데이터 (오늘/최근/리뷰)
│   └── PromptContext 생성
├── _build_prompt()               # LLM 프롬프트 구성
├── _request_trade_decision()     # AI 의사결정 요청
│   └── AIProvider.decide_json()
├── _handle_close_now()           # 즉시 청산 처리 (필요시)
├── _execute_trade()              # 거래 실행
│   ├── _run_confirm_step()       # 확인 단계 (TP/SL 검증)
│   ├── 포지션 사이즈 계산
│   ├── 리스크 검증
│   └── BybitUtils.open_position()
└── TradeStore에 기록
```

### 3. 데이터 수집 과정

1. **시장 데이터**: `utils/price_utils.py`의 `bybit_utils` 클래스 사용
   - CCXT를 통해 Bybit에서 OHLCV 데이터 조회
   - RSI 지표 계산 (ta 라이브러리 사용)
   - CSV 형식으로 변환

2. **포지션 정보**: `utils/bybit_utils.py`의 `BybitUtils` 클래스 사용
   - 활성 포지션 조회
   - 잔고 정보 조회
   - TP/SL 업데이트

3. **저널 데이터**: `utils/storage.py`의 `TradeStore` 클래스 사용
   - 과거 거래 내역
   - AI 결정 로그
   - 손실 리뷰

### 4. AI 의사결정 과정

1. **프롬프트 구성**: 다음 정보를 포함
   - 현재 시간 및 심볼 정보
   - OHLCV CSV (4h/1h/15m)
   - RSI 값 (4h/1h/15m)
   - 현재 포지션 상태
   - 최근 저널 및 리뷰
   - 트레이딩 규칙 및 제약사항

2. **AI 호출**: `utils/ai_provider.py`의 `AIProvider` 사용
   - OpenAI 호환 API (DeepSeek/Qwen 등 지원)
   - JSON 형식 응답 강제
   - Tool calling 지원 (선택적)

3. **응답 파싱**: 
   - 성공: JSON 직접 파싱
   - 실패: 텍스트 응답을 `make_to_object`로 파싱

### 5. 거래 실행 과정

1. **검증 단계**:
   - AI 상태 추출 (`hold`, `long`, `short`, `stop`)
   - 설명(explain) 필드 필수 확인
   - `close_now` 처리 (즉시 청산)
   - `update_existing` 처리 (TP/SL 업데이트만)

2. **확인 단계** (`_run_confirm_step`):
   - TP/SL 퍼센트 계산
   - 레버리지 기준 손실률 검증
   - AI에게 최종 확인 요청
   - 필요시 TP/SL/가격/레버리지 조정

3. **포지션 사이즈 계산**:
   - 리스크 기반 계산 (`utils/risk.py`)
   - 심볼별 노출 상한 적용
   - 최소 수량 검증

4. **주문 실행**:
   - 레버리지 설정
   - 시장가/지정가 주문 실행
   - TP/SL 설정
   - TradeStore에 기록

### 6. 손실 리뷰 프로세스

```
run_loss_review()
└── JournalService.review_losing_trades()
    ├── 최근 손실 거래 조회 (10시간 이내)
    ├── AI에게 리뷰 요청
    │   ├── 진입/청산 시점 OHLCV 데이터 제공
    │   ├── 손실 원인 분석 프롬프트
    │   └── 교훈 및 개선사항 요청
    └── 리뷰를 저널에 기록 (다음 의사결정 시 참조)
```

## 폴더 구조 및 파일 역할

### 프로젝트 루트

```
Cryptocurrency_with_ai/
├── main.py              # 애플리케이션 진입점
├── scheduler.py          # 스케줄러 설정 및 실행
├── webapp.py            # FastAPI 웹 대시보드
├── requirements.txt     # Python 의존성 목록
├── .env                 # 환경변수 설정 (gitignore)
└── README.md            # 프로젝트 설명
```

### `app/` - 애플리케이션 코어

애플리케이션의 핵심 로직을 담는 패키지입니다.

#### `app/core/` - 코어 유틸리티

- **`symbols.py`**: 심볼 관련 유틸리티 함수
  - `parse_trading_symbols()`: 환경변수에서 거래 심볼 목록 파싱
  - `to_ccxt_symbols()`: Bybit 심볼을 CCXT 형식으로 변환
  - `per_symbol_allocation()`: 심볼당 배분 비율 계산

#### `app/workflows/` - 워크플로우

- **`trading.py`**: 자동매매 워크플로우의 핵심
  - `automation_for_symbol()`: 심볼별 자동매매 메인 함수
  - `_gather_prompt_context()`: 프롬프트 컨텍스트 수집
  - `_build_prompt()`: LLM 프롬프트 구성
  - `_request_trade_decision()`: AI 의사결정 요청
  - `_execute_trade()`: 거래 실행 로직
  - `_run_confirm_step()`: TP/SL 확인 단계
  - `run_loss_review()`: 손실 거래 리뷰 프로세스

#### `app/services/` - 도메인 서비스

- **`market_data.py`**: 시장 데이터 유틸리티
  - `ohlcv_csv_between()`: 특정 시간 범위의 OHLCV 데이터를 CSV로 반환

- **`journal.py`**: 저널 서비스
  - `JournalService`: 저널 관련 비즈니스 로직
    - `format_trade_reviews_for_prompt()`: 프롬프트용 리뷰 포맷팅
    - `review_losing_trades()`: 손실 거래 AI 리뷰 생성

#### `app/logging_config.py` - 로깅 설정

- `setup_logging()`: 로깅 설정 (파일 + 콘솔)

### `utils/` - 유틸리티 모듈

공통 기능과 외부 서비스 통합을 담당합니다.

#### `utils/ai_provider.py` - AI 제공자

- **`AIProvider`**: OpenAI 호환 API 클라이언트
  - `decide()`: 텍스트 응답 받기
  - `decide_json()`: JSON 응답 받기 (거래 결정용)
  - `confirm_trade_json()`: TP/SL 확인용 JSON 응답

**환경변수**:
- `OPENAI_BASE_URL`: API 엔드포인트 (예: https://api.deepseek.com/v1)
- `OPENAI_API_KEY`: API 키
- `OPENAI_MODEL`: 모델 이름 (예: deepseek-reasoner)
- `OPENAI_TOOLCALL`: Tool calling 사용 여부 ("1" = 활성화)

#### `utils/bybit_utils.py` - Bybit 거래소 통합

- **`BybitUtils`**: Bybit 거래소 API 래퍼
  - `get_positions()`: 포지션 조회
  - `get_balance()`: 잔고 조회
  - `open_position()`: 포지션 개시
  - `update_symbol_tpsl()`: TP/SL 업데이트
  - `close_symbol_positions()`: 포지션 청산
  - `set_leverage()`: 레버리지 설정

**환경변수**:
- `BYBIT_ENV`: demo | testnet | mainnet
- `BYBIT_API_KEY`: API 키
- `BYBIT_API_SECRET`: API 시크릿
- `BYBIT_RECV_WINDOW_MS`: 요청 타임아웃 (기본 15000ms)

#### `utils/storage.py` - 데이터 저장소

- **`StorageConfig`**: 저장소 설정
  - `resolve()`: MySQL 또는 SQLite 선택

- **`TradeStore`**: 거래 및 저널 데이터 저장
  - `record_trade()`: 거래 기록
  - `record_journal()`: 저널 기록
  - `load_trades()`: 거래 목록 조회
  - `fetch_journals()`: 저널 조회

**환경변수**:
- `MYSQL_URL`: MySQL 연결 문자열
- `SQLITE_PATH`: SQLite 파일 경로
- `FORCE_SQLITE`: SQLite 강제 사용 ("1" = 활성화)

#### `utils/price_utils.py` - 가격 데이터 유틸리티

- **`bybit_utils`**: 가격 데이터 조회 클래스
  - `get_ohlcv()`: OHLCV 데이터 조회
  - `get_rsi()`: RSI 지표 계산
  - `get_current_price()`: 현재가 조회

#### `utils/risk.py` - 리스크 관리

- **`calculate_position_size()`**: 리스크 기반 포지션 사이즈 계산
  - 위험금액 = 잔고 × 리스크 비율
  - 포지션 사이즈 = 위험금액 / (진입가 - 손절가)

- **`enforce_max_loss_sl()`**: 최대 손실률에 따른 SL 강제 조정

#### `utils/function.py` - 유틸리티 함수

- **`make_to_object`**: JSON 파싱 헬퍼
  - LLM 텍스트 응답을 JSON 객체로 변환
  - JSON 코드 블록에서 JSON 추출

#### `utils/types.py` - 타입 정의

- 현재 비어있음 (향후 타입 정의용)

#### `utils/__init__.py` - 패키지 초기화

- 유틸리티 모듈의 공개 API 노출

### `data/` - 데이터 디렉토리

- **`trading.sqlite`**: SQLite 데이터베이스 파일 (기본 저장소)

### `docs/` - 문서

- **`architecture.md`**: 시스템 아키텍처 문서 (이 파일)

## 환경변수 설정

프로젝트 루트에 `.env` 파일을 생성하고 다음 변수들을 설정하세요:

```bash
# OpenAI 설정
OPENAI_BASE_URL=https://api.deepseek.com/v1
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=deepseek-reasoner
OPENAI_TOOLCALL=0

# Bybit 설정
BYBIT_ENV=testnet  # 또는 mainnet, demo
BYBIT_API_KEY=your_api_key_here
BYBIT_API_SECRET=your_api_secret_here

# 거래 심볼
TRADING_SYMBOLS=XRPUSDT,WLDUSDT,ETHUSDT,BTCUSDT

# 데이터베이스 설정
MYSQL_URL=mysql+pymysql://user:password@host:3306/database
# 또는 SQLite 사용
FORCE_SQLITE=1
SQLITE_PATH=data/trading.sqlite

# 리스크 관리
MAX_ALLOC_PERCENT=20.0
DEFAULT_LEVERAGE=5
MAX_LOSS_PERCENT=80.0

# 테스트넷 모드
TESTNET=1
```

## 실행 방법

### 1. 자동매매 시스템 실행

```bash
python main.py
```

또는 스케줄러만 실행:

```bash
python scheduler.py
```

### 2. 웹 대시보드 실행

```bash
python webapp.py
```

또는:

```bash
uvicorn webapp:app --reload
```

웹 브라우저에서 `http://localhost:8000` 접속

## 주요 의존성

- **ccxt**: 암호화폐 거래소 통합 라이브러리
- **openai**: OpenAI SDK
- **pandas**: 데이터 처리
- **fastapi**: 웹 프레임워크
- **ta**: 기술적 지표 계산
- **schedule**: 스케줄링
- **sqlalchemy**: 데이터베이스 ORM

## 확장 포인트

### 1. 리스크 정책 변경

`utils/risk.py`의 `calculate_position_size()` 함수를 수정하거나, `app/workflows/trading.py`의 `_execute_trade()` 함수 내부 로직을 변경합니다.

### 2. 프롬프트 커스터마이징

`app/workflows/trading.py`의 `_build_prompt()` 함수에서 프롬프트 구조를 조정할 수 있습니다.

### 3. 추가 지표 추가

`utils/price_utils.py`의 `bybit_utils` 클래스에 새로운 지표 계산 메서드를 추가하고, `_gather_prompt_context()`에서 수집하여 프롬프트에 포함시킵니다.

### 4. 새로운 AI 모델 통합

`utils/ai_provider.py`의 `AIProvider` 클래스를 확장하여 다른 AI 제공자를 추가할 수 있습니다.

## 로깅 및 모니터링

- 로그 파일: `trading.log`
- LLM 호출 결과는 JSON 형식으로 로깅되어 추적 가능
- 예외는 모두 `LOGGER.error`를 통해 기록

## 주의사항

1. **연결 관리**: `automation_for_symbol`은 심볼마다 새로운 `BybitUtils` 인스턴스를 생성합니다. 고빈도 호출 시 커넥션 풀링을 고려하세요.

2. **확인 단계**: TP/SL 확인 단계는 둘 다 존재할 때만 작동합니다.

3. **손실 리뷰**: `run_loss_review`는 LLM 호출 실패 시 조용히 넘어갑니다. 필요시 재시도/알림을 추가하세요.

4. **테스트넷 사용**: 실제 자금을 사용하기 전에 반드시 테스트넷에서 충분히 테스트하세요.

## 자가 피드백 루프

시스템은 손실 거래를 자동으로 분석하여 다음 의사결정에 반영합니다:

1. **손실 감지**: `JournalService.review_losing_trades()`가 최근 손실 거래를 조회
2. **AI 분석**: 손실 원인과 개선사항을 AI에게 요청
3. **저널 기록**: 리뷰를 저널에 저장
4. **자기 보정**: 다음 트레이딩 사이클에서 과거 리뷰를 참고하여 전략 조정

이를 통해 시스템이 지속적으로 학습하고 개선됩니다.
