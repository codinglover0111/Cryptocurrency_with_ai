# app/services - 서비스 계층

## 역할

- 마켓 데이터, 트레이드/저널 기록, 공용 분석 등을 제공하는 서비스 레이어입니다.
- FastAPI 라우터와 워크플로가 의존하며, 외부 저장소/LLM 접근을 캡슐화합니다.

## 구성 파일

- `market_data.py`: CCXT/Bybit 기반 OHLCV·주문 조회 유틸.
- `journal.py`: `JournalService`로 트레이드/액션 로그 기록, 리뷰 요약.
- `supabase_repo.py`: Supabase Python 클라이언트(`supabase-py`)를 사용한 CRUD 래퍼. `agent_prompts`, `scheduler_state`, `shared_analysis`, `runtime_config` 테이블을 PostgREST로 읽고 씁니다. 서비스 롤 키만 사용합니다.
- `__init__.py`: 패키지 초기화.

## 운용 메모

- Supabase 연동은 `SUPABASE_URL`과 `SUPABASE_SERVICE_ROLE_KEY`가 있어야 활성화됩니다. 없으면 기존 `utils.storage.TradeStore`(SQLite/MySQL)로 폴백됩니다.
- Supabase 사용 시 RLS가 설정되어 있는지 반드시 확인하고, 서버측에서는 서비스 롤 키만 사용하세요.
- LLM 호출 비용은 `TRADE_REVIEW_WAIT_HOURS` 등으로 제어합니다.
