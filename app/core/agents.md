# app/core - 공통 유틸

## 역할

- 거래 심볼과 숫자 포맷에 대한 공통 헬퍼를 제공합니다. 워크플로와 서비스 계층에서 재사용됩니다.

## 파일 가이드

- `symbols.py`
  - `DEFAULT_SYMBOLS`: 기본 거래 심볼 목록
  - `parse_trading_symbols`: 환경변수(`TRADING_SYMBOLS`)를 읽어 심볼 리스트 생성
  - `to_ccxt_symbols` / `contract_to_spot_symbol`: Bybit 심볼을 CCXT 스팟/선물 포맷으로 변환
  - `per_symbol_allocation`: 심볼 수에 따라 자산 배분 비율 계산

## 유지보수 체크리스트

- 심볼 포맷을 바꿀 때 `utils/bybit_utils.py`, `app/workflows/trading.py`의 포맷 기대치를 함께 확인하세요.
- 정밀도(`round_price`) 변경이 필요한 경우 체결/포지션 로직에서 부작용이 없는지 검증합니다.
