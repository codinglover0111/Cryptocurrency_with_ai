# 주의

이 프로그램은 절대로 수익을 절대로 보장 못합니다.
투자는 여러분의 선택입니다!

## 실행법

env에 환경변수를 넣고 나서 docker를 설치한 다음 아래 명령어를 실행합니다.

```text
touch trading.log;docker run -d \
  --name trading-bot \
  --restart unless-stopped \
  -v $(pwd)/.env:/app/.env \
  -v $(pwd)/trading.log:/app/trading.log \
  trading-bot
```

## TODO

- 현재 자산을 조회한 후 유동적으로 자산 비중을 선택하게 할 예정
- 웹 사이트 UI를 통해 상태 확인
- MYSQL 또는 xlsx를 통해 거래 내역을 저장하게 해야함
- 거래를 통하여 얼마나 이익을 봤는 지 통계를 내주는 기능
- 레버리지를 코딩으로 선택하는 기능

## Docker Compose로 앱 + MySQL + 웹 UI 실행

```text
docker compose up -d --build
```

환경변수(.env) 예시:

```text
TESTNET=1
GEMINI_API_KEY=...
AI_PROVIDER=gemini
OPENAI_BASE_URL=https://api.deepseek.com/v1
OPENAI_API_KEY=...
OPENAI_MODEL=deepseek-reasoner

# 리스크는 고정 20% (코드에 강제), MAX 노출 퍼센트만 조절 가능
MAX_ALLOC_PERCENT=20
DEFAULT_LEVERAGE=5

# MySQL
MYSQL_ROOT_PASSWORD=rootpass
MYSQL_DATABASE=cryptobot
MYSQL_USER=bot
MYSQL_PASSWORD=botpass

# 다중 코인 지원 (쉼표로 구분, 예시는 이미지의 6종)
TRADING_SYMBOLS=XRPUSDT,WLDUSDT,ETHUSDT,BTCUSDT,SOLUSDT,DOGEUSDT
```

웹 UI: http://localhost:8000 (엔드포인트: /health, /status, /leverage, /stats)

## 웹 UI GET 파라미터 가이드

### 루트(/)

- `tz` (선택): 시간대 지정. 지원 형식:
  - `UTC` (UTC 고정)
  - `UTC+9`, `UTC-5` (UTC 기준 시간 오프셋, 시간 단위 정수)

예시:

```text
http://localhost:8000/?tz=UTC
http://localhost:8000/?tz=UTC+9
```

루트 페이지의 저널 목록, 모달 상세 시간 표기가 `tz`에 맞게 렌더됩니다. 지정하지 않으면 브라우저 로컬 타임존을 사용합니다.

### 판단 오버레이(/overlay)

- `limit` (기본: 10): 표시할 저널 카드 개수
- `symbol` (선택): 특정 심볼만 필터 (예: `BTC/USDT:USDT`)
- `types` (선택): 콤마(,)로 구분된 저널 타입 필터(`thought,decision,action,review` 중)
- `today_only` (기본: 1): 오늘자만 표시(1) / 전체(0)
- `ascending` (기본: 1): 시간 오름차순(1) / 내림차순(0)
- `refresh` (기본: 5): 자동 새로고침 주기(초)
- `tz` (선택): 시간대 지정(`UTC`, `UTC+9`, `UTC-5` 등)

예시:

```text
http://localhost:8000/overlay?limit=12&today_only=1&tz=UTC
http://localhost:8000/overlay?symbol=BTC/USDT:USDT&types=decision,action&refresh=10&tz=UTC+9
```

### 포지션 오버레이(/overlay_positions)

- `symbol` (선택): 특정 심볼의 포지션만 표시
- `refresh` (기본: 5): 자동 새로고침 주기(초)
- `fs` (선택): 폰트 크기(px)
- `minfs` (선택): 자동 폰트 축소 시 최소 폰트 크기(px)

예시:

```text
http://localhost:8000/overlay_positions?symbol=BTC/USDT:USDT&refresh=3&fs=18
http://localhost:8000/overlay_positions?refresh=5&fs=20&minfs=14
```

## 프롬프트(LLM) 포함 정보

- 현재 UTC 시간: 애플리케이션이 프롬프트에 `YYYY-MM-DD HH:MM:SS UTC` 형식으로 삽입합니다.
- 캔들/포지션/저널 요약 등 컨텍스트가 포함됩니다. 프로바이더(API 키)가 없으면 AI 리뷰는 비활성화됩니다.
