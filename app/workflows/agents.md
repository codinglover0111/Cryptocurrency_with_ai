# app/workflows - 트레이딩 워크플로

## 개요

- 자동 매매 파이프라인의 엔트리 포인트를 제공하고 LangGraph 기반 멀티 에이전트 실행 흐름을 묶습니다.

## 파일 가이드

- `trading.py`
  - `automation_for_symbol`: 심볼별 자동 매매 실행(멀티 에이전트 그래프 호출, 주문/저널 기록)
  - `run_automation_for_all_symbols`: 전체 거래 심볼에 대해 자동매매 분석 실행 (관리자 UI 즉시 실행용)
  - `run_loss_review`: 손실 트레이드 리뷰 루틴
  - `build_trading_graph`: 캐시된 `TradingGraph` 재구성/검증
  - `PromptContext`, `AutomationDependencies`: LLM 입력 및 공용 상태 관리
  - `_run_multi_agent_cycle`: 멀티 에이전트 그래프 실행 및 결과 수집 (indicator/pattern/trend/decision)
  - `_get_btc_analysis_context`: BTC 분석 결과를 다른 심볼 컨텍스트에 추가
  - `_save_btc_analysis`: BTC 분석 결과를 공유 테이블에 저장
  - `_get_risk_config`: 런타임 리스크 설정 로드
  - `__init__.py`: 패키지 부트스트랩

## 에이전트 분석 결과 저장

멀티 에이전트 사이클 실행 후 모든 에이전트의 분석 결과를 저널 `meta.agents` 필드에 저장합니다:

```python
{
  "decision": { ... },
  "agents": {
    "indicator": { "rsi": 45.2, "macd_signal": "bullish", "summary": "..." },
    "pattern": { "patterns_found": [...], "analysis": "..." },
    "trend": { "trend_direction": "uptrend", "analysis": "..." },
    "decision": { "status": "long", "explain": "..." }
  }
}
```

이 데이터는 대시보드의 에이전트 분석 모달에서 사용됩니다.

## BTC 우선 분석

- `main.py`에서 심볼 목록을 BTC 우선으로 정렬 (`_sort_symbols_btc_first`)
- BTC 분석 완료 시 결과를 `shared_analysis` 테이블에 저장
- 다른 심볼 분석 시 BTC 결과를 프롬프트 컨텍스트에 포함 (최대 60분 이내 결과)

## 리스크 설정 연동

- `_get_risk_config()`로 `app/config/runtime_config.json`의 리스크 설정 로드
- `default_leverage`, `max_loss_percent`, `position_allocation_percent` 적용
- 환경변수(`MAX_LOSS_PERCENT`, `DEFAULT_LEVERAGE`)로 오버라이드 가능

## 유지보수 체크리스트

- 단일 LLM 워크플로우는 제거되어 멀티 에이전트 그래프만 사용합니다. 그래프 로직 변경 시 상태/캐시 일관성을 확인하세요.
- 스케줄러 주기는 `main.py`와 `app/config/default_config.py` 기본값을 함께 관리하세요.
- 공용 객체(`BybitUtils`, `TradeStore`, `AIProvider`) 초기화 비용이 크므로 캐싱/지속성 변경 시 성능 영향도 검토하세요.
- BTC 분석 결과 공유 로직 변경 시 `utils/storage.py`의 `shared_analysis` 테이블 스키마를 확인하세요.
