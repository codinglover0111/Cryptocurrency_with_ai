# app/ - 핵심 애플리케이션 모듈

이 디렉토리는 QuantAgent 시스템의 핵심 비즈니스 로직을 담고 있습니다.

## 디렉토리 구조

```text
app/
├── agents/        # 멀티 에이전트 구현
├── auth/          # 인증 및 권한 관리
├── config/        # 런타임 설정
├── core/          # 공용 유틸리티
├── graph/         # LangGraph 워크플로우
├── opro/          # Adaptive-OPRO 시스템
├── services/      # 도메인 서비스
├── web/           # 웹 라우트
└── workflows/     # 트레이딩 파이프라인
```

## 모듈 의존성

```text
workflows/trading.py
    ├── graph/workflow.py (멀티 에이전트 모드)
    │       └── agents/*.py
    │       └── opro/*.py
    ├── services/journal.py
    └── utils/bybit_utils.py
```

## 주요 파일

| 파일                | 설명          |
| ------------------- | ------------- |
| `__init__.py`       | 패키지 초기화 |
| `logging_config.py` | 로깅 설정     |

## 관련 문서

- [agents/README.md](agents/README.md): 에이전트 상세 설명
- [auth/README.md](auth/README.md): 인증 시스템 설명
- [opro/README.md](opro/README.md): Adaptive-OPRO 설명
- [docs/agents.md](../docs/agents.md): 전체 아키텍처 문서
- [docs/ccxt_bybit_data_structures.md](../docs/ccxt_bybit_data_structures.md): CCXT Bybit 데이터 구조
