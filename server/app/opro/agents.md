# app/opro - Adaptive-OPRO

## 역할

- 실거래/백테스트 성과를 반영해 에이전트 프롬프트를 점진적으로 최적화합니다.

## 파일 가이드

- `regime_detector.py`: ADX/ATR 등 지표로 시장 레짐(트렌딩/횡보) 분류
- `meta_prompt.py`: 과거 프롬프트·성과·레짐 정보를 결합한 메타 프롬프트 생성
- `optimizer.py`: OPRO 루프 실행, 최적 프롬프트 후보를 생성/갱신
- `scorer.py`: ROI/Sharpe/승률 기반 점수 계산 (`PerformanceScorer`)
- `__init__.py`: 편의 익스포트

## 유지보수 체크리스트

- 성과 창(`performance_window`)과 최소 트레이드 수(`min_trades_for_update`)는 `app/config/default_config.py`와 관리자 UI 설정이 일치해야 합니다.
- 레짐 판별 로직을 바꿀 때는 패턴/트렌드 에이전트 입력 데이터가 충분히 제공되는지 확인하세요.
- OPRO 모델 변경 시 비용/속도를 고려해 스케줄 주기를 함께 조정합니다.
