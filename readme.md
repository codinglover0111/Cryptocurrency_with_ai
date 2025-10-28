# 주의!!!!

이 프로그램은 절대로 수익을 절대로 보장 못합니다.
투자는 여러분의 선택입니다!

## 실행법

env에 환경변수를 넣고 나서 docker를 설치한 다음 아래 명령어를 실행합니다.

```
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

```
docker compose up -d --build
```

환경변수(.env) 예시:

```
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
