# app/core - 공통 유틸

## 역할

- 거래 심볼과 숫자 포맷에 대한 공통 헬퍼를 제공합니다. 워크플로와 서비스 계층에서 재사용됩니다.

## 파일 가이드

- `symbols.py`
  - `DEFAULT_SYMBOLS`: 기본 거래 심볼 목록
  - `AVAILABLE_SYMBOLS`: Bybit에서 거래 가능한 주요 USDT 선물 심볼 목록 (약 100개)
  - `parse_trading_symbols`: 심볼 목록 반환 (우선순위: raw 인자 > DB > 환경변수 > 기본값)
  - `get_trading_symbols_from_db`: DB에서 거래 심볼 목록 조회
  - `save_trading_symbols_to_db`: DB에 거래 심볼 목록 저장
  - `to_ccxt_symbols` / `contract_to_spot_symbol`: Bybit 심볼을 CCXT 스팟/선물 포맷으로 변환
  - `per_symbol_allocation`: 심볼 수에 따라 자산 배분 비율 계산

## 심볼 설정 우선순위

1. 함수에 직접 전달된 `raw` 문자열
2. DB에 저장된 심볼 목록 (`runtime_config` 테이블, `trading_symbols` 섹션)
3. 환경변수 `TRADING_SYMBOLS`
4. `DEFAULT_SYMBOLS` 기본값

## 유지보수 체크리스트

- 심볼 포맷을 바꿀 때 `utils/bybit_utils.py`, `app/workflows/trading.py`의 포맷 기대치를 함께 확인하세요.
- 정밀도(`round_price`) 변경이 필요한 경우 체결/포지션 로직에서 부작용이 없는지 검증합니다.
- 새로운 심볼을 `AVAILABLE_SYMBOLS`에 추가할 때 Bybit Linear USDT 선물에서 실제 거래 가능한지 확인하세요.
- 관리자 UI에서 심볼 설정 시 DB에 저장되며, 서버 재시작 후에도 유지됩니다.
