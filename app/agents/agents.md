# app/agents - 멀티 에이전트

## 역할

- Indicator/Pattern/Trend/Decision 4개 에이전트와 공용 스키마·프롬프트를 정의합니다.
- `TradingState`를 통해 LangGraph 노드 간 컨텍스트를 공유하며, 워크플로는 `app/graph/workflow.py`에서 호출됩니다.

## 파일 가이드

- `indicator_agent.py`: RSI, MACD, Stochastic, Bollinger 등 기술 지표 요약을 생성
- `pattern_agent.py`: 캔들 이미지 기반 패턴 감지(비전 LLM), 패턴 강도/신뢰도 산출
- `trend_agent.py`: 지지/저항·채널·추세 방향 추정(비전 LLM)
- `decision_agent.py`: 위 결과를 종합해 LONG/SHORT/HOLD/STOP, 포지션 크기, TP/SL을 결정
- `prompts.py`: 각 에이전트의 기본 시스템 프롬프트와 OPRO용 문맥 구성
- `schemas.py`: IndicatorResult/PatternResult/TrendResult/TradeDecision Pydantic 스키마
- `state.py`: `TradingState` 데이터 클래스 및 helper 필드
- `__init__.py`: 주요 클래스/함수 re-export

## 유지보수 체크리스트

- 스키마 변경 시 `app/graph/workflow.py`의 노드 반환 형태와 `app/workflows/trading.py`의 소비 로직을 함께 수정합니다.
- 새 에이전트를 추가할 경우 `app/config/default_config.py`의 기본 모델 설정과 관리자 UI(`app/web/admin.py`)의 payload 구조를 맞춰야 합니다.
- 비전 모델을 쓰는 패턴/트렌드 에이전트는 API 키(Gemini/OpenAI 등)와 입력 이미지 인코딩 규약을 확인하세요.
