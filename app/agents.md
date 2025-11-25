# app/ - 운영 가이드

QuantAgent의 실행 계층을 구성하는 핵심 모듈들이 모여 있습니다. 각 하위 폴더의 세부 책임은 해당 폴더의 `agents.md`에서 더 자세히 설명합니다.

## 하위 폴더 개요

- `agents/`: Indicator/Pattern/Trend/Decision 에이전트와 프롬프트·스키마 정의
- `auth/`: 세션 기반 인증/권한 부여, 관리자/일반 사용자 관리, IP 차단 시스템
- `config/`: LLM, 스케줄러, Adaptive-OPRO, 리스크 기본 설정 및 런타임 설정 저장소
- `core/`: 공통 심볼/숫자 포맷 유틸리티
- `graph/`: LangGraph 기반 트레이딩 워크플로 그래프와 LLM 팩토리
- `opro/`: Adaptive-OPRO(프롬프트 최적화) 루프 구성 요소
- `services/`: 마켓 데이터/저널링 서비스 계층
- `web/`: FastAPI 라우터(관리자/사용자), 스케줄러 제어, 리스크 설정 API
- `workflows/`: 자동 트레이딩 워크플로의 엔트리 포인트, BTC 우선 분석

## 최근 추가된 주요 기능 (v3.0)

- **IP 차단 시스템**: 로그인 실패 횟수 초과 시 자동 IP 차단 (`app/auth/`)
- **스케줄러 제어**: 관리자 UI에서 일시 중단/재개 가능 (`main.py`, `app/web/admin.py`)
- **리스크 설정 UI**: 레버리지, 최대 손실 %, 포지션 할당 % 조정 (`app/config/`, `app/web/admin.py`)
- **로그 뷰어**: `trading.log` 실시간 조회 (레벨별 필터링) (`webapp.py`)
- **BTC 우선 분석**: BTCUSDT를 먼저 분석하고 결과를 다른 심볼에 공유 (`app/workflows/trading.py`)
- **다크/라이트 모드**: 테마 전환 버튼 (`static/theme.js`)

## 유지보수 메모

- `app/config/runtime_config.json`은 관리자 UI 변경 사항을 저장하므로, 수동 수정 시 스키마를 깨지 않도록 주의합니다.
- 새 기능 추가 시 `docs/agents.md`와 각 폴더의 `agents.md`를 먼저 갱신하고 구현을 진행합니다.
- 스케줄러나 LLM 설정을 바꿀 때는 `.env`와 `app/config/default_config.py`의 기본값이 서로 어긋나지 않는지 확인합니다.
- 스케줄러 상태와 공유 분석 결과는 `utils/storage.py`의 테이블에 저장됩니다.
