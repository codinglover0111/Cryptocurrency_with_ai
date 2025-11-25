# app/graph - LangGraph 워크플로

## 역할

- 멀티 에이전트 트레이딩 플로우를 LangGraph `StateGraph`로 정의하고, 필요 LLM 인스턴스를 생성합니다.

## 파일 가이드

- `workflow.py`: `build_trading_graph`와 `TradingGraph` 클래스를 통해 indicator → pattern → trend → decision → adaptive_opro 노드 흐름을 구성하고 실행
- `llm_factory.py`: OpenAI/OpenRouter/Gemini/Anthropic provider별 Chat 모델 생성기 (`LLMConfigurationError` 포함)
- `__init__.py`: 그래프와 팩토리 익스포트

## 유지보수 체크리스트

- 노드에서 반환하는 키는 `app/agents/schemas.py`와 `app/workflows/trading.py`가 기대하는 구조를 따라야 합니다.
- 새 provider 지원 시 `llm_factory.py`에 API 키 명과 기본 엔드포인트를 추가하고 `.env.sample`도 업데이트하세요.
- 그래프 분기 로직을 수정하면 Adaptive-OPRO 호출 조건과 스케줄러 주기에 미치는 영향을 테스트하세요.
