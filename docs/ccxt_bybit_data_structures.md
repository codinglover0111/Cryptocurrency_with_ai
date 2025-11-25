# CCXT Bybit 데이터 구조 레퍼런스

> **문서 버전**: 2024-11  
> **참조**: [CCXT Bybit 공식 문서](https://docs.ccxt.com/#/exchanges/bybit)

이 문서는 CCXT 라이브러리를 통해 Bybit 거래소와 연동할 때 사용되는 핵심 데이터 구조를 설명합니다. 프로젝트의 `utils/bybit_utils.py`에서 이 구조들을 직접 활용합니다.

---

## 목차

1. [Markets (시장 정보)](#1-markets-시장-정보)
2. [Ticker (현재가 정보)](#2-ticker-현재가-정보)
3. [Order Book (호가창)](#3-order-book-호가창)
4. [OHLCV (캔들스틱)](#4-ohlcv-캔들스틱)
5. [Trades (체결 내역)](#5-trades-체결-내역)
6. [Orders (주문)](#6-orders-주문)
7. [Positions (포지션)](#7-positions-포지션)
8. [Balance (잔고)](#8-balance-잔고)
9. [프로젝트 내 활용 예시](#9-프로젝트-내-활용-예시)

---

## 1. Markets (시장 정보)

`exchange.load_markets()` 호출 시 반환되는 거래 가능한 마켓 정보입니다.

### 데이터 구조

```python
{
    "id": "BTCUSDT",                    # 거래소 내부 식별자
    "symbol": "BTC/USDT:USDT",          # CCXT 통합 심볼 (base/quote:settle)
    "base": "BTC",                      # 기준 통화
    "quote": "USDT",                    # 견적 통화
    "settle": "USDT",                   # 결제 통화 (선물)
    "baseId": "BTC",                    # 거래소 내부 기준 통화 ID
    "quoteId": "USDT",                  # 거래소 내부 견적 통화 ID
    "type": "swap",                     # 마켓 유형: spot | swap | future | option
    "spot": False,                      # 현물 여부
    "margin": False,                    # 마진 거래 지원 여부
    "swap": True,                       # 무기한 선물 여부
    "future": False,                    # 만기 선물 여부
    "option": False,                    # 옵션 여부
    "active": True,                     # 현재 거래 가능 여부
    "linear": True,                     # 선형 계약 (USDT 결제)
    "inverse": False,                   # 역방향 계약 (BTC 결제)
    "contract": True,                   # 계약 상품 여부
    "contractSize": 1,                  # 계약 당 수량
    "maker": 0.0002,                    # 메이커 수수료율 (0.02%)
    "taker": 0.00055,                   # 테이커 수수료율 (0.055%)
    "precision": {
        "amount": 3,                    # 수량 소수점 자릿수
        "price": 1,                     # 가격 소수점 자릿수
        "cost": 8,                      # 비용 소수점 자릿수
        "base": 8,                      # 기준 통화 소수점
        "quote": 8                      # 견적 통화 소수점
    },
    "limits": {
        "amount": {
            "min": 0.001,               # 최소 주문 수량
            "max": 100                  # 최대 주문 수량
        },
        "price": {
            "min": 0.5,                 # 최소 주문 가격
            "max": 999999               # 최대 주문 가격
        },
        "cost": {
            "min": 5,                   # 최소 주문 금액 (USDT)
            "max": None                 # 최대 주문 금액
        },
        "leverage": {
            "min": 1,                   # 최소 레버리지
            "max": 100                  # 최대 레버리지
        }
    },
    "info": { ... }                     # 거래소 원본 응답 (비정규화)
}
```

### 주요 필드 설명

| 필드        | 타입     | 설명                                               |
| ----------- | -------- | -------------------------------------------------- |
| `id`        | `string` | Bybit 내부에서 사용하는 마켓 ID (예: `BTCUSDT`)    |
| `symbol`    | `string` | CCXT 표준 심볼. 선물은 `BTC/USDT:USDT` 형식        |
| `type`      | `string` | `spot`, `swap` (무기한), `future` (만기), `option` |
| `linear`    | `bool`   | USDT/USDC 마진 선물 여부                           |
| `inverse`   | `bool`   | 코인(BTC/ETH) 마진 선물 여부                       |
| `precision` | `object` | 주문 시 허용되는 소수점 자릿수                     |
| `limits`    | `object` | 수량/가격/비용의 최솟값·최댓값                     |
| `active`    | `bool`   | `False`면 현재 거래 불가 (유지보수 등)             |

### 마켓 유형 구분

```python
# 프로젝트에서 카테고리 감지 로직 (bybit_utils.py)
def _detect_category(symbol: str) -> str:
    market = exchange.market(symbol)
    if market.get("linear"):
        return "linear"      # USDT/USDC 마진
    if market.get("inverse"):
        return "inverse"     # 코인 마진
    if market.get("option"):
        return "option"      # 옵션
    return "linear"          # 기본값
```

---

## 2. Ticker (현재가 정보)

`exchange.fetch_ticker(symbol)` 호출 시 반환되는 실시간 시세 정보입니다.

### 데이터 구조

```python
{
    "symbol": "BTC/USDT:USDT",          # CCXT 통합 심볼
    "timestamp": 1700000000000,          # Unix 타임스탬프 (밀리초)
    "datetime": "2024-11-15T12:00:00.000Z",  # ISO8601 형식
    "high": 95000.0,                     # 24시간 최고가
    "low": 92000.0,                      # 24시간 최저가
    "bid": 94500.0,                      # 최우선 매수호가
    "bidVolume": 10.5,                   # 매수호가 수량
    "ask": 94510.0,                      # 최우선 매도호가
    "askVolume": 8.2,                    # 매도호가 수량
    "vwap": 93750.5,                     # 24시간 거래량 가중평균가
    "open": 93000.0,                     # 24시간 전 시가
    "close": 94500.0,                    # 현재가 (= last)
    "last": 94500.0,                     # 최근 체결가
    "previousClose": 93000.0,            # 전일 종가
    "change": 1500.0,                    # 24시간 변동액 (close - open)
    "percentage": 1.61,                  # 24시간 변동률 (%)
    "average": 93750.0,                  # 24시간 평균가
    "baseVolume": 15000.5,               # 24시간 기준통화 거래량 (BTC)
    "quoteVolume": 1406287500.0,         # 24시간 견적통화 거래량 (USDT)
    # 선물 전용 필드
    "indexPrice": 94480.0,               # 인덱스 가격 (지표가)
    "markPrice": 94490.0,                # 마크 가격 (청산 기준)
    "fundingRate": 0.0001,               # 펀딩비율
    "fundingTimestamp": 1700000000000,   # 다음 펀딩 시간
    "openInterest": 50000.0,             # 미결제약정 (OI)
    "info": { ... }                      # 거래소 원본 응답
}
```

### 주요 필드 설명

| 필드           | 타입    | 설명                                  |
| -------------- | ------- | ------------------------------------- |
| `last`         | `float` | 가장 최근 체결된 가격                 |
| `bid` / `ask`  | `float` | 현재 최우선 매수/매도 호가            |
| `high` / `low` | `float` | 24시간 기준 최고/최저 가격            |
| `baseVolume`   | `float` | 24시간 기준통화(BTC 등) 거래량        |
| `quoteVolume`  | `float` | 24시간 견적통화(USDT 등) 거래량       |
| `markPrice`    | `float` | 청산 계산에 사용되는 마크 가격 (선물) |
| `fundingRate`  | `float` | 8시간마다 적용되는 펀딩비율 (선물)    |

### 프로젝트 활용

```python
# bybit_utils.py
def get_last_price(self, symbol: str) -> Optional[float]:
    ticker = self.exchange.fetch_ticker(symbol)
    return ticker.get("last") if ticker else None
```

---

## 3. Order Book (호가창)

`exchange.fetch_order_book(symbol, limit)` 호출 시 반환되는 호가 정보입니다.

### 데이터 구조

```python
{
    "symbol": "BTC/USDT:USDT",
    "timestamp": 1700000000000,
    "datetime": "2024-11-15T12:00:00.000Z",
    "nonce": 123456789,                  # 호가창 시퀀스 번호 (증분 업데이트용)
    "bids": [
        # [가격, 수량] 쌍, 가격 내림차순 정렬
        [94500.0, 1.5],                  # 최우선 매수호가
        [94490.0, 2.3],
        [94480.0, 0.8],
        # ...
    ],
    "asks": [
        # [가격, 수량] 쌍, 가격 오름차순 정렬
        [94510.0, 1.2],                  # 최우선 매도호가
        [94520.0, 3.1],
        [94530.0, 0.5],
        # ...
    ]
}
```

### 구조 시각화

```
asks (매도)     │  가격      │  수량
                │  94530.0   │  0.5   ← 3번째 매도
                │  94520.0   │  3.1   ← 2번째 매도
                │  94510.0   │  1.2   ← 최우선 매도 (best ask)
────────────────┼────────────┼─────────
                │   스프레드  │  10.0 USDT
────────────────┼────────────┼─────────
bids (매수)     │  94500.0   │  1.5   ← 최우선 매수 (best bid)
                │  94490.0   │  2.3   ← 2번째 매수
                │  94480.0   │  0.8   ← 3번째 매수
```

### 주요 필드 설명

| 필드    | 타입         | 설명                                     |
| ------- | ------------ | ---------------------------------------- |
| `bids`  | `list[list]` | 매수 주문 목록 `[[price, amount], ...]`  |
| `asks`  | `list[list]` | 매도 주문 목록 `[[price, amount], ...]`  |
| `nonce` | `int`        | WebSocket 증분 업데이트 시 시퀀스 확인용 |

### limit 파라미터

```python
# Bybit 지원 호가 깊이
exchange.fetch_order_book("BTC/USDT:USDT", limit=25)   # 양쪽 25단계
exchange.fetch_order_book("BTC/USDT:USDT", limit=50)   # 양쪽 50단계
exchange.fetch_order_book("BTC/USDT:USDT", limit=200)  # 양쪽 200단계
```

---

## 4. OHLCV (캔들스틱)

`exchange.fetch_ohlcv(symbol, timeframe, since, limit)` 호출 시 반환되는 캔들 데이터입니다.

### 데이터 구조

```python
# 반환값: list[list]
[
    #   [timestamp,    open,      high,      low,       close,     volume]
    [1700000000000, 94000.0,   94500.0,   93800.0,   94300.0,   150.5],
    [1700003600000, 94300.0,   94800.0,   94100.0,   94700.0,   180.2],
    [1700007200000, 94700.0,   95000.0,   94500.0,   94900.0,   200.8],
    # ...
]
```

### 캔들 배열 인덱스

| 인덱스 | 필드        | 타입    | 설명                    |
| ------ | ----------- | ------- | ----------------------- |
| `[0]`  | `timestamp` | `int`   | 캔들 시작 시간 (밀리초) |
| `[1]`  | `open`      | `float` | 시가                    |
| `[2]`  | `high`      | `float` | 고가                    |
| `[3]`  | `low`       | `float` | 저가                    |
| `[4]`  | `close`     | `float` | 종가                    |
| `[5]`  | `volume`    | `float` | 거래량 (기준통화)       |

### 지원 타임프레임

```python
# Bybit 지원 타임프레임
exchange.timeframes = {
    '1m': '1',       # 1분
    '3m': '3',       # 3분
    '5m': '5',       # 5분
    '15m': '15',     # 15분
    '30m': '30',     # 30분
    '1h': '60',      # 1시간
    '2h': '120',     # 2시간
    '4h': '240',     # 4시간
    '6h': '360',     # 6시간
    '12h': '720',    # 12시간
    '1d': 'D',       # 1일
    '1w': 'W',       # 1주
    '1M': 'M'        # 1개월
}
```

### 프로젝트 활용

```python
# app/services/market_data.py
def ohlcv_csv_between(symbol: str, timeframe: str, start: int, end: int) -> str:
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=start, limit=500)
    # CSV 변환: timestamp, open, high, low, close, volume
    return "\n".join([
        f"{candle[0]},{candle[1]},{candle[2]},{candle[3]},{candle[4]},{candle[5]}"
        for candle in ohlcv
    ])
```

---

## 5. Trades (체결 내역)

### 5.1 Public Trades (공개 체결)

`exchange.fetch_trades(symbol, since, limit)` 호출 시 반환됩니다.

```python
{
    "id": "2389472893472",               # 체결 고유 ID
    "timestamp": 1700000000000,          # 체결 시간 (밀리초)
    "datetime": "2024-11-15T12:00:00.000Z",
    "symbol": "BTC/USDT:USDT",
    "type": "limit",                     # 주문 유형
    "side": "buy",                       # 방향: buy | sell
    "price": 94500.0,                    # 체결 가격
    "amount": 0.5,                       # 체결 수량
    "cost": 47250.0,                     # 체결 금액 (price × amount)
    "takerOrMaker": "taker",             # taker | maker
    "info": { ... }                      # 거래소 원본
}
```

### 5.2 My Trades (개인 체결)

`exchange.fetch_my_trades(symbol, since, limit)` 호출 시 반환됩니다. 인증 필요.

```python
{
    "id": "2389472893472",
    "timestamp": 1700000000000,
    "datetime": "2024-11-15T12:00:00.000Z",
    "symbol": "BTC/USDT:USDT",
    "order": "order-id-123",             # 연관 주문 ID
    "type": "market",
    "side": "buy",
    "price": 94500.0,
    "amount": 0.5,
    "cost": 47250.0,
    "takerOrMaker": "taker",
    "fee": {
        "currency": "USDT",              # 수수료 통화
        "cost": 26.0,                    # 수수료 금액
        "rate": 0.00055                  # 수수료율
    },
    "fees": [
        {"currency": "USDT", "cost": 26.0, "rate": 0.00055}
    ],
    "info": { ... }
}
```

### 주요 필드 설명

| 필드           | 타입     | 설명                                  |
| -------------- | -------- | ------------------------------------- |
| `id`           | `string` | 체결 고유 식별자                      |
| `order`        | `string` | 연관된 주문 ID (개인 체결만)          |
| `side`         | `string` | `buy` 또는 `sell`                     |
| `takerOrMaker` | `string` | 테이커/메이커 구분 (수수료 차등 적용) |
| `fee`          | `object` | 수수료 정보 (개인 체결만)             |

### 프로젝트 활용

```python
# bybit_utils.py
def get_my_trades(self, symbol=None, since_ms=None, limit=100):
    return self.exchange.fetch_my_trades(symbol, since_ms, limit)
```

---

## 6. Orders (주문)

`exchange.fetch_orders()`, `fetch_open_orders()`, `fetch_closed_orders()` 등으로 조회합니다.

### 데이터 구조

```python
{
    "id": "order-123456789",             # 주문 고유 ID
    "clientOrderId": "my-order-001",     # 사용자 지정 ID (선택)
    "timestamp": 1700000000000,          # 주문 생성 시간
    "datetime": "2024-11-15T12:00:00.000Z",
    "lastTradeTimestamp": 1700000100000, # 마지막 체결 시간
    "lastUpdateTimestamp": 1700000100000,# 마지막 상태 변경 시간
    "symbol": "BTC/USDT:USDT",
    "type": "limit",                     # limit | market | stop_limit | ...
    "timeInForce": "GTC",                # GTC | IOC | FOK | PostOnly
    "postOnly": False,                   # 메이커 전용 여부
    "reduceOnly": False,                 # 감소 전용 여부 (선물)
    "side": "buy",                       # buy | sell
    "price": 94000.0,                    # 주문 가격 (지정가)
    "triggerPrice": None,                # 트리거 가격 (조건부 주문)
    "stopPrice": None,                   # 스탑 가격
    "amount": 1.0,                       # 주문 수량
    "cost": 94000.0,                     # 총 주문 금액
    "average": 94050.0,                  # 평균 체결가
    "filled": 0.8,                       # 체결된 수량
    "remaining": 0.2,                    # 미체결 수량
    "status": "open",                    # open | closed | canceled | expired
    "fee": {
        "currency": "USDT",
        "cost": 20.7,
        "rate": 0.00055
    },
    "trades": [ ... ],                   # 체결 내역 배열
    "stopLoss": {                        # TP/SL 설정
        "triggerPrice": 92000.0,
        "price": None,                   # None = 시장가 청산
        "type": "market"
    },
    "takeProfit": {
        "triggerPrice": 98000.0,
        "price": None,
        "type": "market"
    },
    "info": { ... }
}
```

### 주문 상태 (status)

| 상태       | 설명                       |
| ---------- | -------------------------- |
| `open`     | 미체결 또는 부분 체결 상태 |
| `closed`   | 완전 체결됨                |
| `canceled` | 사용자에 의해 취소됨       |
| `expired`  | 유효기간 만료 (IOC/FOK 등) |
| `rejected` | 거래소에서 거부됨          |

### 주문 유형 (type)

| 유형                 | 설명                                     |
| -------------------- | ---------------------------------------- |
| `market`             | 시장가 주문 - 즉시 체결                  |
| `limit`              | 지정가 주문 - 지정 가격에 도달 시 체결   |
| `stop_market`        | 스탑 시장가 - 트리거 도달 시 시장가 주문 |
| `stop_limit`         | 스탑 지정가 - 트리거 도달 시 지정가 주문 |
| `take_profit_market` | TP 시장가                                |
| `take_profit_limit`  | TP 지정가                                |

### 유효기간 (timeInForce)

| 옵션       | 설명                                           |
| ---------- | ---------------------------------------------- |
| `GTC`      | Good Till Cancel - 취소 전까지 유효            |
| `IOC`      | Immediate Or Cancel - 즉시 체결, 미체결분 취소 |
| `FOK`      | Fill Or Kill - 전량 체결 또는 전량 취소        |
| `PostOnly` | 메이커로만 체결 (테이커 시 취소)               |

### 프로젝트 활용

```python
# bybit_utils.py
def open_position(self, position: Open_Position):
    if position.type == "market":
        return self.exchange.create_market_buy_order(
            position.symbol,
            position.quantity,
            params={
                "takeProfit": position.tp,
                "stopLoss": position.sl,
            }
        )
```

---

## 7. Positions (포지션)

`exchange.fetch_positions(symbols)` 호출 시 반환되는 선물 포지션 정보입니다.

### 데이터 구조

```python
{
    "id": None,                          # 포지션 ID (거래소에 따라 다름)
    "symbol": "BTC/USDT:USDT",
    "timestamp": 1700000000000,
    "datetime": "2024-11-15T12:00:00.000Z",
    "contracts": 0.5,                    # 보유 계약 수
    "contractSize": 1,                   # 계약 당 수량
    "side": "long",                      # long | short
    "notional": 47250.0,                 # 명목 가치 (contracts × price)
    "leverage": 10,                      # 적용 레버리지
    "unrealizedPnl": 250.0,              # 미실현 손익
    "realizedPnl": 100.0,                # 실현 손익
    "percentage": 5.3,                   # 손익률 (%)
    "entryPrice": 94000.0,               # 평균 진입가
    "markPrice": 94500.0,                # 마크 가격
    "liquidationPrice": 85000.0,         # 청산가
    "marginMode": "cross",               # cross | isolated
    "marginType": "cross",               # 마진 유형
    "maintenanceMargin": 189.0,          # 유지 증거금
    "maintenanceMarginPercentage": 0.4,  # 유지 증거금률 (%)
    "initialMargin": 4725.0,             # 개시 증거금
    "initialMarginPercentage": 10.0,     # 개시 증거금률 (%)
    "collateral": 5000.0,                # 담보금
    "hedged": False,                     # 헷지 모드 여부
    "stopLoss": {
        "triggerPrice": 92000.0,
        "price": None,
        "type": "market"
    },
    "takeProfit": {
        "triggerPrice": 98000.0,
        "price": None,
        "type": "market"
    },
    "info": { ... }                      # 거래소 원본
}
```

### 주요 필드 설명

| 필드               | 타입     | 설명                                            |
| ------------------ | -------- | ----------------------------------------------- |
| `contracts`        | `float`  | 보유 중인 계약(포지션) 수량                     |
| `side`             | `string` | `long` (매수), `short` (매도)                   |
| `entryPrice`       | `float`  | 평균 진입 가격                                  |
| `markPrice`        | `float`  | 현재 마크 가격 (PnL 계산 기준)                  |
| `liquidationPrice` | `float`  | 강제 청산 가격                                  |
| `unrealizedPnl`    | `float`  | 미실현 손익                                     |
| `leverage`         | `int`    | 적용된 레버리지 배율                            |
| `marginMode`       | `string` | `cross` (전체 마진) 또는 `isolated` (격리 마진) |

### 포지션 모드

| 모드         | 설명                      | positionIdx |
| ------------ | ------------------------- | ----------- |
| One-way      | 단방향 포지션만 보유 가능 | 0           |
| Hedge (Buy)  | 양방향 모드의 롱 포지션   | 1           |
| Hedge (Sell) | 양방향 모드의 숏 포지션   | 2           |

### 프로젝트 활용

```python
# bybit_utils.py
def get_positions_by_symbol(self, symbol: str):
    positions = self.exchange.fetch_positions(None)
    return [p for p in positions
            if p.get("symbol") == symbol
            and abs(float(p.get("contracts") or 0)) > 1e-8]
```

---

## 8. Balance (잔고)

`exchange.fetch_balance()` 호출 시 반환되는 계정 잔고 정보입니다.

### 데이터 구조

```python
{
    "info": { ... },                     # 거래소 원본 응답
    "timestamp": 1700000000000,
    "datetime": "2024-11-15T12:00:00.000Z",
    # 통화별 상세 잔고
    "USDT": {
        "free": 5000.0,                  # 사용 가능 잔고
        "used": 1000.0,                  # 사용 중 (주문/마진)
        "total": 6000.0                  # 총 잔고 (free + used)
    },
    "BTC": {
        "free": 0.1,
        "used": 0.05,
        "total": 0.15
    },
    # 전체 요약 (모든 통화)
    "free": {
        "USDT": 5000.0,
        "BTC": 0.1,
        # ...
    },
    "used": {
        "USDT": 1000.0,
        "BTC": 0.05,
        # ...
    },
    "total": {
        "USDT": 6000.0,
        "BTC": 0.15,
        # ...
    }
}
```

### 주요 필드 설명

| 필드    | 타입    | 설명                              |
| ------- | ------- | --------------------------------- |
| `free`  | `float` | 출금/주문에 사용 가능한 잔고      |
| `used`  | `float` | 미체결 주문 또는 마진에 묶인 잔고 |
| `total` | `float` | `free + used`                     |

### 계정 유형 (Bybit)

Bybit는 여러 계정 유형을 가지며, 파라미터로 지정할 수 있습니다:

```python
# 통합 계정 (Unified Trading Account)
balance = exchange.fetch_balance({"type": "swap"})

# 스팟 계정
balance = exchange.fetch_balance({"type": "spot"})

# 펀딩 계정
balance = exchange.fetch_balance({"type": "funding"})
```

### 프로젝트 활용

```python
# bybit_utils.py
def get_balance(self, currency: str = "USDT") -> Optional[Dict[str, Any]]:
    balance = self.exchange.fetch_balance()
    return {
        "currency": currency,
        "total": balance.get("total", {}).get(currency),
        "free": balance.get("free", {}).get(currency),
        "used": balance.get("used", {}).get(currency),
        "raw": balance,
    }
```

---

## 9. 프로젝트 내 활용 예시

### 9.1 심볼 변환

```python
# CCXT 심볼 ↔ Bybit 마켓 ID 변환
#
# CCXT:  "BTC/USDT:USDT"  (선물)
# Bybit: "BTCUSDT"

def _symbol_to_market_id(symbol: str) -> str:
    market = exchange.market(symbol)
    return market.get("id") or symbol.replace("/", "").replace(":", "")

def _market_id_to_symbol(market_id: str) -> str:
    market = exchange.markets_by_id.get(market_id)
    return market.get("symbol") if market else market_id
```

### 9.2 정밀도 처리

```python
# 주문 수량/가격은 거래소 정밀도에 맞춰야 함
amount = exchange.amount_to_precision("BTC/USDT:USDT", 0.123456789)
# → "0.123"

price = exchange.price_to_precision("BTC/USDT:USDT", 94567.89123)
# → "94567.9"
```

### 9.3 Closed PnL 조회 (청산된 포지션 손익)

```python
# bybit_utils.py - get_position_history()
# Bybit V5 API: /v5/position/closed-pnl

def get_position_history(self, symbol=None, since_ms=None, limit=50):
    request = {
        "category": "linear",
        "symbol": self._symbol_to_market_id(symbol),
        "limit": limit,
    }
    if since_ms:
        request["startTime"] = since_ms

    raw = self.exchange.privateGetV5PositionClosedPnl(request)
    return raw.get("result", {}).get("list", [])
```

### 9.4 TP/SL 설정

```python
# 포지션 TP/SL 업데이트
def update_symbol_tpsl(self, symbol, take_profit=None, stop_loss=None):
    params = {
        "category": self._detect_category(symbol),
        "tpslMode": "Full",  # Full = 전체 포지션에 적용
    }

    return self.exchange.set_trading_stop(
        symbol,
        stopLoss=stop_loss,
        takeProfit=take_profit,
        params=params
    )
```

### 9.5 레버리지 및 마진 모드 설정

```python
# 레버리지 설정 (포지션이 없을 때만 변경 가능)
exchange.set_leverage(10, "BTC/USDT:USDT", {"marginMode": "cross"})

# 마진 모드 설정
exchange.set_margin_mode("cross", "BTC/USDT:USDT")  # cross | isolated

# 포지션 모드 설정
exchange.set_position_mode(False)  # False=One-way, True=Hedge
```

---

## 참고 자료

- [CCXT 공식 문서](https://docs.ccxt.com)
- [CCXT Bybit 문서](https://docs.ccxt.com/#/exchanges/bybit)
- [Bybit API V5 문서](https://bybit-exchange.github.io/docs/v5/intro)
- [CCXT GitHub Wiki](https://github.com/ccxt/ccxt/wiki/manual)

---

## 변경 이력

| 날짜    | 버전 | 설명           |
| ------- | ---- | -------------- |
| 2024-11 | 1.0  | 초기 문서 작성 |
