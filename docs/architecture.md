# Architecture Overview (2.0)

`Cryptocurrency_with_ai` v2.0은 `src/crypto_bot` 패키지를 중심으로 구성되며, OpenAI 기반 다중 턴 의사결정과 Supabase 저장소를 기본값으로 사용합니다. 아래 문서는 주요 디렉터리 구조, 런타임 흐름, LLM 상호작용 계약, 데이터 저장 방식, 배포 전략을 정리합니다.

## Project Layout

```text
src/crypto_bot/
  core/            # 공용 유틸(심볼, 리스크 계산 등)
  data/            # OHLCV/차트 생성 유틸리티
  integrations/    # 외부 API 연동 (Bybit 등)
  llm/             # OpenAI 클라이언트와 대화 규약 정의
  persistence/     # Supabase + SQLAlchemy 백엔드
  services/        # 도메인 서비스 (저널, 시장 데이터)
  workflows/       # 자동매매 오케스트레이션
docker/            # Railway / Scheduler 전용 Dockerfile
docs/              # 아키텍처 및 운영 문서
main.py            # 스케줄러 엔트리포인트
```

v2.0에서는 레거시 모듈이 제거되고 `src/crypto_bot` 이하 패키지만 유지됩니다. 프런트엔드는 별도 Next.js 레포지토리에서 관리합니다.

## Runtime Environment

- 프로젝트는 Python 3.11과 uv 기반으로 동작합니다.
- 로컬 개발 및 운영 환경 모두 `uv sync`로 의존성을 준비한 뒤 `uv run main.py`를 실행하는 흐름을 권장합니다.
- FastAPI 웹 UI는 `uv run uvicorn webapp:app --host 0.0.0.0 --port 8000`으로 기동할 수 있으며, 동일한 명령이 Docker/배포 스크립트에서도 사용됩니다.

## Runtime Flow

1. `main.py`가 환경변수를 로드하고 로깅을 설정한 뒤 5분 주기 스케줄러를 기동합니다.
2. 각 심볼마다 `workflows.trading.automation_for_symbol`이 호출되어 아래 단계를 수행합니다.
   - **컨텍스트 수집**: `_gather_prompt_context`가 4h/1h/15m OHLCV CSV와 RSI, 현재 포지션, 저널 로그를 정리합니다.
   - **다중 턴 의사결정**: `_request_trade_decision`이 아래 프로토콜을 수행합니다.
     1. LLM이 `phase="analysis_request"` JSON을 반환하면서 초안(draft)과 필요한 지표(needs)를 알립니다.
     2. 봇은 손익률(기초/레버리지), 허용 손실 한도, 심볼별 노출 한도 등을 계산해 `phase="metrics"` JSON으로 응답합니다.
     3. LLM이 `phase="final_decision"` JSON으로 최종 주문 계획을 회신합니다.
     4. 구조화 응답이 실패하면 `AIProvider.decide_json()` → `decide()` + `decision_parser` 순으로 폴백합니다.
   - **즉시 청산 처리**: `close_now` 신호가 포함되면 reduceOnly 시장가 주문으로 포지션을 부분/전체 청산합니다.
   - **확정 및 실행**: `_run_confirm_step`에서 손익률과 손절 폭을 검증하고, `_execute_trade`가 포지션 사이즈 계산 → Bybit 주문 → 트레이드/저널 기록을 진행합니다.
3. `run_loss_review`는 동일한 주기로 손실 거래를 Supabase에서 조회하여 LLM 리뷰를 생성, 저널에 기록합니다.

## Multi-Turn Conversation Contract

- **analysis_request** – LLM 초안

  ```json
  {
    "phase": "analysis_request",
    "draft": {
      "Status": "long",
      "order_type": "limit",
      "price": 1.2345,
      "tp": 1.3456,
      "sl": 1.2,
      "leverage": 8,
      "buy_now": false,
      "update_existing": false
    },
    "needs": ["leveraged_loss_pct", "position_capacity"]
  }
  ```

- **metrics** – 봇이 계산한 값을 다시 전달

  ```json
  {
    "phase": "metrics",
    "metrics": {
      "baseline_loss_pct": 2.78,
      "leveraged_loss_pct": 22.24,
      "baseline_profit_pct": 9.0,
      "leveraged_profit_pct": 72.0,
      "max_loss_cap_pct": 40.0
    },
    "account": {
      "balance_total": 1250.4,
      "per_symbol_allocation_pct": 16.6,
      "max_notional_per_symbol": 1666.4
    }
  }
  ```

- **final_decision** – 최종 JSON 응답 (Status, order_type, price, tp, sl, buy_now, leverage, close_now, close_percent, update_existing, explain)

Confirm 단계는 이전과 동일하게 `AIProvider.confirm_trade_json()`을 사용하며, TP/SL·레버리지 제안이 허용된 손실 한도를 초과할 경우 재조정하도록 합니다.

## Prompt & Context

- **Market Snapshot**: 4h/1h/15m CSV + RSI(각 타임프레임) + 현재가
- **Position State**: 진입가, 증거금, 미실현 PnL, TP/SL, 레버리지
- **Journals**: 금일 기록, 최근 결정/리뷰, 포지션 오픈 이후 로그
- **Guardrails**: 레버리지 범위, JSON 스키마, update_existing/close_now 처리 규칙

CSV 데이터는 그대로 전달하여 LLM이 패턴을 직접 파싱할 수 있게 하고, RSI는 빠른 추세 파악을 위해 별도 블록(`[RSI_OVERVIEW]`)으로 제공됩니다.

## Persistence Layer

`persistence.store.TradeStore`는 세 가지 백엔드를 지원합니다.

| Backend  | 사용 조건                                    | 비고                                        |
| -------- | -------------------------------------------- | ------------------------------------------- |
| Supabase | `SUPABASE_URL` + `SUPABASE_SERVICE_ROLE_KEY` | 기본 경로. `trades`, `journals` 테이블 사용 |
| MySQL    | `MYSQL_URL`                                  | 레거시 호환. SQLAlchemy 연결                |
| SQLite   | 기본값 (`data/trading.sqlite`)               | 로컬 개발/폴백용                            |

Supabase 테이블 권장 스키마:

```sql
create table if not exists public.trades (
  id uuid primary key default gen_random_uuid(),
  ts timestamptz not null,
  symbol text not null,
  side text,
  type text,
  price double precision,
  quantity double precision,
  tp double precision,
  sl double precision,
  leverage double precision,
  status text,
  order_id text,
  pnl double precision
);

create table if not exists public.journals (
  id uuid primary key default gen_random_uuid(),
  ts timestamptz not null default now(),
  symbol text,
  entry_type text,
  content text,
  reason text,
  meta jsonb,
  ref_order_id text
);

create index if not exists idx_trades_symbol_ts on public.trades(symbol, ts);
create index if not exists idx_journals_symbol_ts on public.journals(symbol, ts);
```

`TradeStore`는 먼저 Supabase에 기록 후(optional) SQL 백엔드에 미러링합니다. `load_trades()`와 `fetch_journals()`는 두 백엔드를 자동으로 전환하며, 반환값은 항상 Pandas DataFrame입니다.

### Supabase MCP 활용

- 스키마 변경은 `mcp_supabase_apply_migration`을 사용해 배포합니다.
- 운영 중 데이터 조회/진단은 `mcp_supabase_execute_sql`로 수행할 수 있습니다.
- MCP 명령으로 적용된 변경 사항은 즉시 Python 서비스에서 사용 가능합니다.

## Deployment Targets

| 파일                          | 목적                      | 기본 명령        |
| ----------------------------- | ------------------------- | ---------------- |
| `Dockerfile`                  | 로컬/스케줄러 기본 이미지 | `python main.py` |
| `docker/Dockerfile.scheduler` | Railway Worker 등 배치용  | `python main.py` |

모든 Dockerfile은 `ghcr.io/astral-sh/uv:python3.11-slim` 기반이며 `uv pip install --system .`로 의존성을 설치합니다. `.dockerignore`가 SQLite 파일과 가상환경을 제외하도록 구성되어 있습니다.

## Frontend (Next.js)

- FastAPI 및 템플릿 자산은 제거되었으며, SSR UI는 별도 Next.js 레포에서 관리합니다.
- Next.js 측에서 Supabase REST/RPC 혹은 클라이언트를 사용해 `trades`, `journals` 데이터를 직접 조회해야 합니다.
- Python 서비스는 자동매매 스케줄러와 데이터 적재 역할에만 집중하며, 추가 HTTP API를 노출하지 않습니다.

## Monitoring & Resilience

- 로깅은 파일(`trading.log`) + STDOUT에 동시에 기록됩니다.
- 멀티턴 응답은 `llm_multi_turn` 이벤트로 JSON 로깅되어 재현성을 제공합니다.
- 의사결정 폴백 순서: 구조화 Responses → JSON completion → 자유 텍스트 + 의도 파서.
- 스케줄러는 예외를 잡고 다음 주기로 진행하며, Supabase 장애 시 SQL 백엔드로 폴백합니다.

## Extension Points

- **리스크 정책**: `core.risk.calculate_position_size`, `_compute_max_loss_percent` 조정.
- **프롬프트 확장**: `workflows.trading._build_context_payload` 및 `_build_system_prompt`에서 통제.
- **대체 저장소**: `TradeStore`에 새로운 backend 어댑터 추가.
- **테스트**: `_init_dependencies`가 모든 외부 의존성을 주입하므로, Mock `BybitUtils`, `AIProvider`, `TradeStore`로 단위 테스트가 용이합니다.

---

v2.0 구조는 모듈화된 디렉터리와 명확한 대화 계약을 기반으로 하며, Supabase를 기본 데이터 레이크로 채택해 배포 타겟(Railway, Supabase functions 등)에 맞춰 쉽게 확장할 수 있도록 설계되었습니다.
