# app/workflows - 트레이딩 워크플로

## 개요

- 자동 매매 파이프라인의 엔트리 포인트를 제공하고 LangGraph 기반 멀티 에이전트 실행 흐름을 묶습니다.

## 파일 가이드

- `trading.py`
  - `automation_for_symbol`: 심볼별 자동 매매 실행(멀티 에이전트 그래프 호출, 주문/저널 기록)
  - `run_loss_review`: 손실 트레이드 리뷰 루틴
  - `build_trading_graph`: 캐시된 `TradingGraph` 재구성/검증
  - `PromptContext`, `AutomationDependencies`: LLM 입력 및 공용 상태 관리
  - `__init__.py`: 패키지 부트스트랩

## 유지보수 체크리스트
- 단일 LLM 워크플로우는 제거되어 멀티 에이전트 그래프만 사용합니다. 그래프 로직 변경 시 상태/캐시 일관성을 확인하세요.
- 스케줄러 주기는 `main.py`와 `app/config/default_config.py` 기본값을 함께 관리하세요.
- 공용 객체(`BybitUtils`, `TradeStore`, `AIProvider`) 초기화 비용이 크므로 캐싱/지속성 변경 시 성능 영향도 검토하세요.
