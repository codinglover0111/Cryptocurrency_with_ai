# Architecture Overview

`Cryptocurrency_with_ai`는 5분 주기로 데이터를 수집하고 LLM에게 의사결정을 맡기는 자동매매 파이프라인입니다. 이 문서는 핵심 컴포넌트, 스케줄링 흐름, 프롬프트 구조, 손실 피드백, 차트 데이터 입력 경로를 간단히 정리합니다.

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

## Five-Minute Job Cycle

- **주기 관리**: `scheduler.py`에서 APScheduler가 심볼별 `automation_for_symbol`과 `run_loss_review`를 5분 간격으로 스케줄링합니다.
- **거래 자동화 루틴**: 각 심볼은 독립적으로 프롬프트를 구성하고 주문 결정을 수행하여 병렬 실행이 가능합니다.
- **손실 리뷰 루틴**: 동일한 주기로 미체크 손실 거래를 조회해 리뷰를 생성하므로, 시장 변동이 심할 때도 피드백이 늦지 않습니다.
- **장애 복구**: 스케줄러는 애플리케이션 재기동 시 작업을 다시 등록하며, 각 작업은 내부적으로 예외를 캡처해 다음 주기를 유지합니다.

## Prompt Structure

프롬프트는 `_build_prompt`에서 섹션 단위로 구성합니다.

- **System Guardrails**: 거래 목표, 리스크 한도, 응답 포맷(JSON) 요구사항을 명시합니다.
- **Market Snapshot**: 4h/1h/15m OHLCV CSV, 현재 가격, 유동성, 지표 등 차트 데이터를 붙여 LLM이 패턴을 읽을 수 있게 합니다.
- **Position State**: 보유 포지션, 진입가, 증거금, 미실현 손익 등 현재 상태를 전달합니다.
- **Journal & Decisions**: 직전 결정, 실행 결과, 피드백/리뷰 내용을 포함해 LLM이 과거 맥락을 학습하도록 합니다.
- **Task Definition**: 신규 진입, 유지, 청산 중 선택하도록 지시하며, `close_now`, `entries`, `stop_loss`, `take_profit` 필드를 요구합니다.
- **JSON Contract**: LLM 응답은 `Status`(대문자)와 `status`(소문자) 필드가 반드시 포함되어야 합니다. 워크플로는 외부 프로바이더 스키마 호환을 위해 `Status`를 요구하고, 내부 로직은 이를 소문자로 정규화한 `status`를 사용하므로 두 필드를 동시에 유지해야 합니다.

Confirm 단계에서는 원본 결정 프롬프트와 LLM 응답을 다시 제공하며, TP/SL 유효성 검증, 포지션 사이즈 조정 등에 대해 "검토" 역할을 주도록 짧은 프롬프트를 생성합니다.

## Module Breakdown

| 경로                          | 역할                           | 주요 함수                                                                                                 |
| ----------------------------- | ------------------------------ | --------------------------------------------------------------------------------------------------------- |
| `app/core/symbols.py`         | 심볼 관련 유틸리티             | `parse_trading_symbols`, `to_ccxt_symbols`, `per_symbol_allocation`                                       |
| `app/services/market_data.py` | OHLCV 수집 래퍼                | `ohlcv_csv_between`                                                                                       |
| `app/services/journal.py`     | 저널/리뷰 도메인 서비스        | `JournalService.format_trade_reviews_for_prompt`, `JournalService.review_losing_trades`                   |
| `app/workflows/trading.py`    | 자동매매 파이프라인            | `_gather_prompt_context`, `_build_prompt`, `_run_confirm_step`, `_execute_trade`, `automation_for_symbol` |
| `utils/`                      | 거래소/AI/스토리지 레거시 모듈 | `BybitUtils`, `AIProvider`, `TradeStore`, etc.                                                            |

## Data Persistence

- 모든 거래/저널 기록은 `TradeStore`를 통해 MySQL(기본) 또는 SQLite로 저장됩니다.
- `StorageConfig.resolve()`가 환경변수에 따라 연결을 결정합니다.
- `journal_service.review_losing_trades`는 동일 DB를 이용하여 손실 거래를 조회한 뒤 리뷰 기록을 추가합니다.

## Extension Points

- **리스크 정책 변경**: `app/core/symbols.per_symbol_allocation`과 `app/workflows/trading._execute_trade`가 총 USDT 대비 20% 고정 증거금 한도를 계산합니다. 필요 시 `MAX_ALLOC_PERCENT`나 `AVAILABLE_NOTIONAL_SAFETY` 환경 변수를 조정해 배분 규칙을 변경하세요.
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

## Self-Feedback Loop

- **손실 감지**: `JournalService.review_losing_trades`가 DB에서 손실 거래를 가져옵니다.
- **피드백 프롬프트**: 손실 포지션의 차트, 주문 내역, 당시 결정을 요약해 "무엇이 잘못됐는지"를 LLM에 질문합니다.
- **저널 기록**: LLM 응답은 리뷰 저널로 저장되어 다음 트레이딩 프롬프트의 `Journal & Decisions` 섹션에 포함됩니다.
- **자기보정**: 이후 주기에서 LLM이 과거 리뷰를 참고해 리스크를 줄이거나 전략을 조정하도록 설계되었습니다.

## Market Data Intake

- `app/services/market_data.py`가 심볼별로 4h/1h/15m 캔들 데이터를 CCXT/Bybit API를 통해 수집하고 CSV 문자열로 변환합니다.
- `_gather_prompt_context`는 이 CSV를 프롬프트에 그대로 삽입해 LLM이 가격 패턴과 추세를 직접 파싱하게 합니다.
- 최신 가격, 미청산 수량, 손익 정보는 `BybitUtils.get_position`과 `BybitUtils.get_last_price`에서 가져옵니다.
- 외부 데이터 오류나 API Rate Limit 발생 시, 로깅 후 실패 지표를 프롬프트에 기록해 LLM이 데이터 부족 상황을 인지하도록 합니다.
