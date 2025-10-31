# 안내

- **본 프로젝트는 투자 수익을 보장하지 않습니다.** 모든 자동매매 의사결정과 결과는 사용자 책임입니다.
- 실계좌 투입 전에 백테스트와 리스크 점검을 반드시 수행하세요.

## 핵심 변경 사항 (v2.0)

- 코드 베이스가 `src/crypto_bot` 패키지로 재구성되어 모듈 경계가 명확해졌습니다.
- 모든 LLM 호출은 OpenAI Responses API를 사용하며, **다중 턴(JSON) 대화 프로토콜**을 준수합니다.
- `Supabase`를 기본 데이터 스토리지로 사용하며, 필요 시 MySQL/SQLite로 폴백합니다.
- Dockerfile이 역할별(`scheduler`, `web`)로 분리되어 Railway/Supabase 배포와 로컬 개발이 간편해졌습니다.
- Python 3.11 + `uv`(Astral) 환경을 공식 지원합니다.

## 디렉터리 구조

```text
src/crypto_bot/
  core/          # 심볼·리스크 유틸리티
  data/          # OHLCV, RSI, 차트 생성
  integrations/  # Bybit/외부 API 연동
  llm/           # OpenAI 클라이언트 및 프로토콜
  persistence/   # Supabase + SQL 백엔드
  services/      # 저널, 시장 데이터 도메인 서비스
  workflows/     # 자동매매 오케스트레이션
docker/          # 배포 대상별 Dockerfile
docs/            # 아키텍처 / 운영 문서
main.py          # 스케줄러 진입점
```

레거시 `app/`, `utils/` 패키지는 제거되었으며, 모든 구현은 `crypto_bot` 패키지에 있습니다.

## 빠른 시작

```bash
# uv 설치 (https://github.com/astral-sh/uv)
uv python install 3.11
uv venv
source .venv/bin/activate

# 의존성 설치
uv pip install --upgrade pip
uv pip install --system .

# 환경변수 설정 (.env)
cp .env.example .env  # 필요 시 작성

# 자동매매 스케줄러 실행
python main.py
```

## Docker 이미지

| 파일                          | 용도                        | 기본 커맨드      |
| ----------------------------- | --------------------------- | ---------------- |
| `Dockerfile`                  | 로컬 실행 / 스케줄러 베이스 | `python main.py` |
| `docker/Dockerfile.scheduler` | Railway Worker 등 배치용    | `python main.py` |

모든 이미지가 `ghcr.io/astral-sh/uv:python3.11-slim`을 사용하며, `uv pip install --system .`으로 의존성을 설치합니다.

빌드 예시:

```bash
docker build -f docker/Dockerfile.scheduler -t crypto-bot-scheduler .
docker run --env-file .env crypto-bot-scheduler

```

## 주요 환경 변수

```text
# 거래 기본 설정
BybitEnv=demo  # demo | testnet | mainnet
TRADING_SYMBOLS=XRPUSDT,WLDUSDT,ETHUSDT,BTCUSDT,SOLUSDT,DOGEUSDT
MAX_ALLOC_PERCENT=20
DEFAULT_LEVERAGE=5
MAX_LOSS_PERCENT=80
AVAILABLE_NOTIONAL_SAFETY=0.95

# OpenAI
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4.1-mini        # 기본값
OPENAI_BASE_URL=https://api.openai.com/v1  # 선택 사항
OPENAI_TEMPERATURE=0.2

# Supabase (우선순위)
SUPABASE_URL=https://xxxx.supabase.co
SUPABASE_SERVICE_ROLE_KEY=your-service-role-key
SUPABASE_SCHEMA=public
SUPABASE_TRADES_TABLE=trades
SUPABASE_JOURNALS_TABLE=journals

# SQL 폴백 (선택)
MYSQL_URL=mysql+pymysql://user:pass@host:3306/cryptobot
SQLITE_PATH=data/trading.sqlite
```

환경파일 `.env` 에 위 값을 추가하면 `TradeStore`가 순서대로 Supabase → MySQL → SQLite를 사용합니다.

## LLM 다중 턴 의사결정

1. **analysis_request** – LLM이 초안과 필요 지표를 JSON으로 반환합니다.
2. **metrics** – 봇이 손익률(기초/레버리지), 허용 손실 한도, 심볼별 노출 상한 등을 계산해 JSON으로 다시 전달합니다.
3. **final_decision** – LLM이 확정안을 JSON으로 응답합니다. `Status`, `order_type`, `price`, `tp`, `sl`, `buy_now`, `leverage`, `close_now`, `close_percent`, `update_existing`, `explain` 필드가 필수입니다.
4. 구조화 응답이 실패하면 JSON → 자유 텍스트 + 휴리스틱 파서 순으로 폴백합니다.

이 프로토콜은 `src/crypto_bot/workflows/trading.py`에서 확인할 수 있습니다.

## 데이터 저장소

- 기본값은 Supabase(`supabase` Python 클라이언트)이며, `trades`, `journals` 테이블을 사용합니다.
- 동일한 API를 통해 MySQL/SQLite에 자동 폴백합니다.
- Pandas DataFrame을 일관되게 반환하여 분석·리포팅 시 동일한 코드를 재사용할 수 있습니다.

Supabase 테이블 생성 예시는 `docs/architecture.md`를 참고하세요.

## Supabase MCP 사용

- Supabase 스키마와 데이터를 관리할 때는 Supabase MCP 명령을 활용하세요.
- 예시: `mcp_supabase_apply_migration` 호출로 `trades`/`journals` 테이블 DDL을 적용하고, `mcp_supabase_execute_sql`로 쿼리 결과를 즉시 확인할 수 있습니다.
- MCP를 통해 실행한 변경 사항은 이 Python 서비스가 자동으로 소비합니다.

## 프런트엔드 (Next.js)

- FastAPI 기반 웹 서버와 템플릿 자산은 v2.0에서 제거되었습니다.
- 별도 레포에서 Next.js로 SSR 대시보드를 구현하고, Supabase 클라이언트로 `trades`, `journals` 데이터를 조회하세요.
- 필요한 API는 Next.js Route Handler 또는 서버 컴포넌트에서 Supabase REST/RPC를 호출해 대체합니다.

## 참고 문서

- `docs/architecture.md`: 디렉터리 구조, 멀티턴 프로토콜, Supabase 스키마를 상세히 정리
- `docs/` 내 기타 문서(추가 예정)

## 체크리스트

- [ ] `.env`에 OpenAI / Supabase 키를 등록했는가?
- [ ] `uv pip install --system .`으로 의존성을 설치했는가?
- [ ] Supabase에 `trades`, `journals` 테이블이 생성되었는가?
- [ ] `python main.py` 실행 후 `trading.log`와 Supabase 레코드가 정상적으로 기록되는가?

안정적인 운영을 위해 스테이징 환경에서 충분한 모니터링과 로그 검증을 수행한 뒤 실계좌에 적용하시기 바랍니다.
