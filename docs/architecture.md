# Architecture Overview

## Runtime Flow

1. `main.py` loads environment variables, configures logging, 그리고 스케줄러를 시작합니다.
2. 5분 주기로 `automation_for_symbol`이 호출되어 다음 단계를 처리합니다.
   - **Context 수집**: `app/workflows/trading._gather_prompt_context`
     - 시장 데이터(`utils.bybit_utils.bybit_utils`)를 통해 4h/1h/15m OHLCV CSV 확보
     - 현재 포지션, 금일 저널, 최근 의사결정, 리뷰 텍스트를 정리
   - **프롬프트 구성**: `_build_prompt`
     - LLM이 참고할 CSV/저널 정보를 블록별로 포함
   - **LLM 결정**: `_request_trade_decision`
     - `AIProvider.decide_json()` → 실패 시 `decide()` + `make_to_object()` fallback
   - **즉시 청산 여부**: `_handle_close_now`
     - `close_now` 신호가 있으면 reduceOnly 주문으로 포지션 정리 후, 트레이드/저널 기록
   - **주문 계획 수립**: `_run_confirm_step` + `_execute_trade`
     - SL/TP 유효성 확인, 필요 시 추가 Confirm 프롬프트로 재검증
     - 리스크 기반 포지션 사이즈 산출 + 심볼별 노출 상한 적용
     - `BybitUtils.open_position`으로 주문 실행
     - 체결 결과를 `TradeStore`에 기록, 저널(`decision`, `action`) 남김
3. 별도 5분 주기 작업 `run_loss_review`
   - `JournalService.review_losing_trades`가 최근 손실 포지션을 AI에게 분석시켜 리뷰 저널을 추가합니다.

## Module Breakdown

| 경로 | 역할 | 주요 함수 |
|------|------|-----------|
| `app/core/symbols.py` | 심볼 관련 유틸리티 | `parse_trading_symbols`, `to_ccxt_symbols`, `per_symbol_allocation` |
| `app/services/market_data.py` | OHLCV 수집 래퍼 | `ohlcv_csv_between` |
| `app/services/journal.py` | 저널/리뷰 도메인 서비스 | `JournalService.format_trade_reviews_for_prompt`, `JournalService.review_losing_trades` |
| `app/workflows/trading.py` | 자동매매 파이프라인 | `_gather_prompt_context`, `_build_prompt`, `_run_confirm_step`, `_execute_trade`, `automation_for_symbol` |
| `utils/` | 거래소/AI/스토리지 레거시 모듈 | `BybitUtils`, `AIProvider`, `TradeStore`, etc. |

## Data Persistence

- 모든 거래/저널 기록은 `TradeStore`를 통해 MySQL(기본) 또는 SQLite로 저장됩니다.
- `StorageConfig.resolve()`가 환경변수에 따라 연결을 결정합니다.
- `journal_service.review_losing_trades`는 동일 DB를 이용하여 손실 거래를 조회한 뒤 리뷰 기록을 추가합니다.

## Extension Points

- **리스크 정책 변경**: `utils/risk.py`의 `calculate_position_size` 또는 `_execute_trade` 내부 로직을 수정합니다.
- **프롬프트 커스터마이징**: `app/workflows/trading._build_prompt`에서 섹션별 텍스트를 조정할 수 있습니다.
- **확장된 리뷰 전략**: `JournalService`를 상속하거나 구성(extending composition)하여 다른 프롬프트/분석 기법을 도입할 수 있습니다.
- **테스트 도입**: `_init_dependencies` 함수로 외부 의존성을 주입하기 쉬워졌기 때문에, 단위 테스트 시 Mock `BybitUtils`/`AIProvider`를 주입할 수 있습니다.

## Logging & Monitoring

- `app/logging_config.setup_logging`에서 파일(`trading.log`) + 콘솔 핸들러를 설정합니다.
- LLM 호출 결과(`llm_response_parsed`, `llm_confirm_response_parsed`)는 JSON으로 로깅되어 추적이 용이합니다.
- 예외는 모두 `LOGGER.error` / `logging.exception`을 통해 기록됩니다.

## Known Considerations

- `automation_for_symbol`는 심볼마다 새로운 `BybitUtils` 인스턴스를 생성합니다. 고빈도 호출이 필요하다면 커넥션 풀링/재사용 전략을 검토하세요.
- Confirm 단계는 TP/SL 둘 다 존재할 때만 작동합니다. 시장 상황에 따라 조건을 완화해야 할 수도 있습니다.
- 손실 리뷰 기능(`run_loss_review`)은 LLM 호출 실패 시 조용히 넘어가도록 되어 있습니다. 필요하면 재시도/알림을 추가하세요.
