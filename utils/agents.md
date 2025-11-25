# utils - 공통 헬퍼

## 역할

- 거래所(CCXT Bybit), AI 호출, 리스크 계산, 스토리지 작업 등에서 재사용되는 헬퍼 모듈 모음입니다.

## 파일 가이드

- `ai_provider.py`: Gemini/OpenAI(DeepSeek·Qwen) 래퍼, 이미지/텍스트 호출 및 재시도 로직 포함
- `bybit_utils.py`: CCXT 기반 Bybit 주문/포지션/TP·SL/마진 모드 제어, 백오프/에러 코드 처리
- `function.py`: LangChain 함수/도구 포맷 변환 헬퍼
- `price_utils.py`: 캔들 데이터프레임 → 이미지/CSV 변환, 지표 계산 보조
- `risk.py`: 포지션 사이징, 최대 손실 강제(`enforce_max_loss_sl`) 등 리스크 유틸
- `storage.py`: SQLite/MySQL 스토리지 백엔드, `TradeStore`로 트레이드/저널/리뷰 CRUD 제공
- `types.py`: 타입 힌트 플레이스홀더
- `__init__.py`: 익스포트

## 유지보수 체크리스트

- API 키/엔드포인트 변경 시 `ai_provider.py`와 `.env.sample`을 함께 수정하고 호출 제한(재시도 간격)을 조정하세요.
- Bybit 래퍼는 예외 메시지를 그대로 전달하므로, 프런트/워크플로에서 사용자 친화적 메시지가 필요한 경우 래핑을 고려하세요.
- 스토리지 경로와 엔진(`SQLITE_PATH`, `MYSQL_URL`)은 환경변수에 의존합니다. 마이그레이션 시 스키마(`SCHEMA_METADATA`)를 같이 업데이트하세요.
