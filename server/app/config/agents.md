# app/config - 설정

## 역할

- LLM/스케줄러/Adaptive-OPRO/리스크 기본값을 정의하고, 관리자 UI에서 수정된 런타임 설정을 DB에 저장/로드합니다.

## 파일 가이드

- `default_config.py`
  - `AGENT_CONFIG`: Indicator/Pattern/Trend/Decision 기본 provider·model·temperature
  - `SCHEDULER_CONFIG`: 자동매매/손실 리뷰 주기, 콜드 스타트 옵션
  - `ADAPTIVE_OPRO_CONFIG`: OPRO 윈도우·최소 트레이드 수·사용 모델·사이드웨이 임계값
  - `RISK_CONFIG`: 리스크 관리 설정
    - `default_leverage`: 기본 레버리지 (기본값: 5)
    - `max_loss_percent`: 최대 손실 허용 % (레버리지 후 기준, 기본값: 40)
    - `position_allocation_percent`: 포지션당 최대 할당 % (초기 잔고 기준, 기본값: 20)
  - `load_runtime_config` / `save_runtime_config` / `update_runtime_config`: 런타임 설정 입출력
- `__init__.py`: 상수/함수 익스포트

## 런타임 설정 저장소

런타임 설정은 **DB 우선** 저장 방식을 사용합니다:

1. **DB 저장 (기본)**: `utils/storage.py`의 `runtime_config` 테이블에 섹션별로 저장
2. **JSON 파일 폴백**: DB 저장 실패 시 `runtime_config.json`에 저장
3. **자동 마이그레이션**: 기존 JSON 파일이 있으면 DB로 자동 마이그레이션

### 저장 우선순위

- **로드**: DB → JSON 파일 → 기본값
- **저장**: DB → JSON 파일 (폴백)

### 주요 함수

- `_get_trade_store()`: `TradeStore` 싱글톤 인스턴스 반환 (DB 연결 풀 재사용)
- `_load_from_db()`: DB에서 런타임 설정 로드
- `_save_to_db()`: 런타임 설정을 DB에 일괄 저장 (`save_runtime_configs_bulk` 사용)
- `_load_from_json()`: JSON 파일에서 로드 (폴백용)
- `_migrate_json_to_db()`: JSON → DB 자동 마이그레이션

### 성능 최적화

- **싱글톤 패턴**: `get_trade_store()` 함수로 앱 전체에서 하나의 DB 연결 풀 사용
- **일괄 저장**: `save_runtime_configs_bulk()`로 여러 섹션을 한 트랜잭션에서 저장
- 이전: 요청당 DB 연결 2회 + 쿼리 5회 → 현재: 연결 0회 (풀 재사용) + 쿼리 2회

## 유지보수 체크리스트

- 관리자 API(`app/web/admin.py`)가 payload를 DB에 저장하므로, 스키마를 바꿀 때는 관리자 UI와 이 모듈을 동시에 수정하세요.
- `.env.sample`의 기본값과 여기 정의된 기본값이 어긋나지 않는지 검증하세요.
- DB 연결 실패 시 JSON 파일로 자동 폴백됩니다.
- 리스크 설정 변경 시 `app/workflows/trading.py`의 `_get_risk_config()` 함수와 연동되는지 확인하세요.
- `runtime_config` 테이블 스키마는 `utils/storage.py`의 `RUNTIME_CONFIG_TABLE`에 정의되어 있습니다.
