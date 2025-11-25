# app/ - 운영 가이드

QuantAgent의 실행 계층을 구성하는 핵심 모듈들이 모여 있습니다. 각 하위 폴더의 세부 책임은 해당 폴더의 `agents.md`에서 더 자세히 설명합니다.

## 하위 폴더 개요

- `agents/`: Indicator/Pattern/Trend/Decision 에이전트와 프롬프트·스키마 정의
- `auth/`: 세션 기반 인증/권한 부여, 관리자/일반 사용자 관리
- `config/`: LLM, 스케줄러, Adaptive-OPRO 기본 설정 및 런타임 설정 저장소
- `core/`: 공통 심볼/숫자 포맷 유틸리티
- `graph/`: LangGraph 기반 트레이딩 워크플로 그래프와 LLM 팩토리
- `opro/`: Adaptive-OPRO(프롬프트 최적화) 루프 구성 요소
- `services/`: 마켓 데이터/저널링 서비스 계층
- `web/`: FastAPI 라우터(관리자/사용자)
- `workflows/`: 자동 트레이딩 워크플로의 엔트리 포인트

## 유지보수 메모

- `app/config/runtime_config.json`은 관리자 UI 변경 사항을 저장하므로, 수동 수정 시 스키마를 깨지 않도록 주의합니다.
- 새 기능 추가 시 `docs/agents.md`와 각 폴더의 `agents.md`를 먼저 갱신하고 구현을 진행합니다.
- 스케줄러나 LLM 설정을 바꿀 때는 `.env`와 `app/config/default_config.py`의 기본값이 서로 어긋나지 않는지 확인합니다.
