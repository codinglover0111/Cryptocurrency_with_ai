# app/agents - 멀티 에이전트

## 역할

- Indicator/Pattern/Trend/Decision 4개 에이전트와 공용 스키마·프롬프트를 정의합니다.
- `TradingState`를 통해 LangGraph 노드 간 컨텍스트를 공유하며, 워크플로는 `app/graph/workflow.py`에서 호출됩니다.

## 파일 가이드

- `indicator_agent.py`: RSI, MACD, Stochastic, Bollinger 등 기술 지표 요약을 생성
- `pattern_agent.py`: 캔들 이미지 기반 패턴 감지(비전 LLM), 패턴 강도/신뢰도 산출
- `trend_agent.py`: 지지/저항·채널·추세 방향 추정(비전 LLM)
- `decision_agent.py`: 위 결과를 종합해 LONG/SHORT/HOLD/STOP, 포지션 크기, TP/SL을 결정
- `prompts.py`: 각 에이전트의 **기본** 시스템 프롬프트 정의 (DB에 저장된 프롬프트가 없을 때 폴백용)
- `prompt_service.py`: 프롬프트 관리 서비스 (DB 우선, 기본값 폴백)
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

## 프롬프트 관리 (DB 기반)

에이전트 프롬프트는 DB에서 관리되며, 관리자 대시보드에서 실시간으로 수정할 수 있습니다.

### 프롬프트 로딩 순서

1. **DB 조회**: `agent_prompts` 테이블에서 해당 에이전트의 프롬프트 조회
2. **폴백**: DB에 없으면 `prompts.py`의 기본값 사용

### 프롬프트 서비스 API (`prompt_service.py`)

```python
from app.agents.prompt_service import get_prompt, get_all_prompts, save_prompt

# 단일 프롬프트 조회 (DB 우선, 기본값 폴백)
prompt = get_prompt("indicator")

# 모든 프롬프트 조회 (소스, 업데이트 시간 포함)
all_prompts = get_all_prompts()

# 프롬프트 저장
save_prompt("indicator", "새로운 프롬프트 템플릿...")
```

### 프롬프트 변수

각 에이전트마다 사용 가능한 변수가 다릅니다:

| 에이전트  | 변수                                                                                       |
| --------- | ------------------------------------------------------------------------------------------ |
| indicator | `{symbol}`, `{regime}`, `{position_summary}`, `{indicator_block}`                          |
| pattern   | `{symbol}`, `{regime}`, `{indicator_summary}`                                              |
| trend     | `{symbol}`, `{regime}`, `{pattern_summary}`                                                |
| decision  | `{indicator_summary}`, `{pattern_summary}`, `{trend_summary}`, `{regime}`, `{meta_prompt}` |

### 관리자 UI

- **경로**: 관리자 대시보드 → 프롬프트 설정
- **기능**: 에이전트별 프롬프트 조회/수정/초기화
- **API**: `GET/POST /admin/prompts`, `POST /admin/prompts/reset/{agent_type}`

## 유지보수 체크리스트

- 스키마 변경 시 `app/graph/workflow.py`의 노드 반환 형태와 `app/workflows/trading.py`의 소비 로직을 함께 수정합니다.
- 새 에이전트를 추가할 경우 `app/config/default_config.py`의 기본 모델 설정과 관리자 UI(`app/web/admin.py`)의 payload 구조를 맞춰야 합니다.
- 비전 모델을 쓰는 패턴/트렌드 에이전트는 API 키(Gemini/OpenAI 등)와 입력 이미지 인코딩 규약을 확인하세요.
- LLM 모델 변경 시 `with_structured_output`의 `method` 파라미터 호환성을 확인하세요.
- 프롬프트 변경 시 `prompts.py`의 기본값과 `prompt_service.py`의 변수 목록을 함께 수정하세요.
