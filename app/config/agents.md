# app/config - 설정

## 역할

- LLM/스케줄러/Adaptive-OPRO 기본값을 정의하고, 관리자 UI에서 수정된 런타임 설정을 파일(`runtime_config.json`)로 저장/로드합니다.

## 파일 가이드

- `default_config.py`
  - `AGENT_CONFIG`: Indicator/Pattern/Trend/Decision 기본 provider·model·temperature
  - `SCHEDULER_CONFIG`: 자동매매/손실 리뷰 주기, 콜드 스타트 옵션
  - `ADAPTIVE_OPRO_CONFIG`: OPRO 윈도우·최소 트레이드 수·사용 모델·사이드웨이 임계값
  - `load_runtime_config` / `save_runtime_config` / `update_runtime_config`: 런타임 설정 파일 입출력
- `__init__.py`: 상수/함수 익스포트

## 유지보수 체크리스트

- 관리자 API(`app/web/admin.py`)가 payload를 그대로 `runtime_config.json`에 저장하므로, 스키마를 바꿀 때는 관리자 UI와 이 모듈을 동시에 수정하세요.
- `.env.sample`의 기본값과 여기 정의된 기본값이 어긋나지 않는지 검증하세요.
- 파일 쓰기 권한이 없는 환경에서는 외부 설정 저장소(예: DB/Redis)로 교체하는 패치를 고려하세요.
