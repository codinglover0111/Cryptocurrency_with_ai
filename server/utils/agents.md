# utils - 공통 헬퍼

## 역할

- 거래所(CCXT Bybit), AI 호출, 리스크 계산, 스토리지 작업 등에서 재사용되는 헬퍼 모듈 모음입니다.

## 파일 가이드

- `ai_provider.py`: Gemini/OpenAI(DeepSeek·Qwen) 래퍼, 이미지/텍스트 호출 및 재시도 로직 포함
- `bybit_utils.py`: CCXT 기반 Bybit 주문/포지션/TP·SL/마진 모드 제어, 백오프/에러 코드 처리
- `function.py`: LangChain 함수/도구 포맷 변환 헬퍼
- `price_utils.py`: 캔들 데이터프레임 → 이미지/CSV 변환, 지표 계산 보조
- `risk.py`: 포지션 사이징, 최대 손실 강제(`enforce_max_loss_sl`) 등 리스크 유틸
- `storage.py`: SQLite/MySQL 스토리지 백엔드, `TradeStore` 클래스 제공
  - 트레이드/저널/리뷰 CRUD
  - **스케줄러 상태 관리**: `set_scheduler_state`, `get_scheduler_state`, `get_all_scheduler_states`
  - **공유 분석 결과**: `save_shared_analysis`, `get_btc_analysis` (BTC 분석 결과 공유용)
  - **런타임 설정 관리**: `save_runtime_config`, `get_runtime_config`, `get_all_runtime_configs`, `delete_runtime_config`
  - **에이전트 프롬프트 관리**: `save_agent_prompt`, `get_agent_prompt`, `get_all_agent_prompts`, `delete_agent_prompt`
- `types.py`: 타입 힌트 플레이스홀더
- `__init__.py`: 익스포트

## 스케줄러 상태 테이블 (`scheduler_state`)

| 키                    | 설명                                  |
| --------------------- | ------------------------------------- |
| `is_running`          | 스케줄러 실행 중 여부 ("1" / "0")     |
| `paused`              | 일시 중단 상태 ("1" / "0")            |
| `last_automation_run` | 마지막 자동매매 실행 시간 (ISO 형식)  |
| `last_review_run`     | 마지막 손실 리뷰 실행 시간 (ISO 형식) |
| `automation_minutes`  | 자동매매 주기 (분)                    |
| `loss_review_minutes` | 손실 리뷰 주기 (분)                   |

## 공유 분석 테이블 (`shared_analysis`)

- BTC 분석 결과를 저장하여 다른 심볼 분석 시 컨텍스트로 제공
- `get_btc_analysis(max_age_minutes)`: 지정 시간 내 BTC 분석 결과 조회

## 런타임 설정 테이블 (`runtime_config`)

관리자 UI에서 변경한 런타임 설정을 DB에 저장합니다.

| 컬럼          | 타입       | 설명                                               |
| ------------- | ---------- | -------------------------------------------------- |
| `id`          | Integer    | Primary Key (자동 증가)                            |
| `section`     | String(64) | 설정 섹션 (agents, scheduler, risk, adaptive_opro) |
| `config_data` | Text       | JSON 문자열 형태의 설정 데이터                     |
| `updated_at`  | DateTime   | 마지막 업데이트 시간                               |

### 주요 메서드

- `save_runtime_config(section, config_data)`: 개별 섹션 설정 저장 (upsert)
- `save_runtime_configs_bulk(configs)`: 여러 섹션을 한 트랜잭션에서 일괄 저장
- `get_runtime_config(section)`: 특정 섹션 설정 조회
- `get_all_runtime_configs()`: 전체 설정 조회
- `delete_runtime_config(section)`: 설정 삭제

## 에이전트 프롬프트 테이블 (`agent_prompts`)

관리자 UI에서 변경한 에이전트 프롬프트를 DB에 저장합니다.

| 컬럼              | 타입       | 설명                                          |
| ----------------- | ---------- | --------------------------------------------- |
| `id`              | Integer    | Primary Key (자동 증가)                       |
| `agent_type`      | String(32) | 에이전트 타입 (indicator, pattern, trend, decision) |
| `prompt_template` | Text       | 프롬프트 템플릿 텍스트                        |
| `updated_at`      | DateTime   | 마지막 업데이트 시간                          |

### 주요 메서드

- `save_agent_prompt(agent_type, prompt_template)`: 프롬프트 저장 (upsert)
- `save_agent_prompts_bulk(prompts)`: 여러 프롬프트를 한 트랜잭션에서 일괄 저장
- `get_agent_prompt(agent_type)`: 특정 에이전트 프롬프트 조회
- `get_all_agent_prompts()`: 전체 프롬프트 조회
- `delete_agent_prompt(agent_type)`: 프롬프트 삭제 (기본값으로 복원)

## 싱글톤 패턴

DB 연결 풀을 재사용하기 위해 `TradeStore` 싱글톤 패턴을 사용합니다:

- `get_trade_store()`: 앱 전체에서 하나의 `TradeStore` 인스턴스 반환
- 매 요청마다 새 인스턴스를 생성하지 않아 DB 연결 오버헤드 제거
- MySQL 연결 타임아웃 및 풀 설정 포함 (connect: 10초, read/write: 30초)

## 유지보수 체크리스트

- API 키/엔드포인트 변경 시 `ai_provider.py`와 `.env.sample`을 함께 수정하고 호출 제한(재시도 간격)을 조정하세요.
- Bybit 래퍼는 예외 메시지를 그대로 전달하므로, 프런트/워크플로에서 사용자 친화적 메시지가 필요한 경우 래핑을 고려하세요.
- 스토리지 경로와 엔진(`SQLITE_PATH`, `MYSQL_URL`)은 환경변수에 의존합니다. 마이그레이션 시 스키마(`SCHEMA_METADATA`)를 같이 업데이트하세요.
- 스케줄러 상태, 공유 분석, 런타임 설정, 에이전트 프롬프트 테이블은 앱 시작 시 자동 생성됩니다 (`_ensure_schema`).
