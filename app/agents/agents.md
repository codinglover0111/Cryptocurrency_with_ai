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

## LLM 구조화 출력 (Structured Output)

모든 에이전트는 LangChain의 `with_structured_output`을 사용하여 Pydantic 스키마 기반의 구조화된 응답을 받습니다.

### Function Calling 방식 사용

```python
llm.with_structured_output(TradeDecision, method="function_calling")
```

- **`method="function_calling"`**: OpenRouter를 통해 다양한 모델(DeepSeek, Gemini 등)을 사용할 때 함수 호출 방식을 명시적으로 지정합니다.
- 이 방식은 LLM이 JSON 텍스트 대신 tool call로 응답하도록 강제하여 파싱 오류를 방지합니다.
- `strict=True`는 일부 모델에서 지원하지 않으므로 사용하지 않습니다.

### 주의사항

- OpenRouter를 통해 사용하는 모델 중 일부는 `strict=True` 모드를 지원하지 않아 마크다운 텍스트를 반환할 수 있습니다.
- 이 경우 `method="function_calling"`을 명시하면 tool call 형태로 응답을 강제할 수 있습니다.

## 유지보수 체크리스트

- 스키마 변경 시 `app/graph/workflow.py`의 노드 반환 형태와 `app/workflows/trading.py`의 소비 로직을 함께 수정합니다.
- 새 에이전트를 추가할 경우 `app/config/default_config.py`의 기본 모델 설정과 관리자 UI(`app/web/admin.py`)의 payload 구조를 맞춰야 합니다.
- 비전 모델을 쓰는 패턴/트렌드 에이전트는 API 키(Gemini/OpenAI 등)와 입력 이미지 인코딩 규약을 확인하세요.
- LLM 모델 변경 시 `with_structured_output`의 `method` 파라미터 호환성을 확인하세요.
