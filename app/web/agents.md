# app/web - FastAPI 라우터

## 역할

- 관리자/사용자용 API 엔드포인트를 정의하여 설정 변경과 상태 조회를 제공합니다.

## 파일 가이드

- `admin.py`: `/admin` 프리픽스
  - `GET /admin/models`: 에이전트별 사용 가능 모델 목록
  - `GET/POST /admin/agent-config`: 에이전트 LLM 설정 조회/갱신
  - `GET/POST /admin/scheduler`: 스케줄러 설정 조회/갱신 (실행 상태 포함)
  - `POST /admin/scheduler/pause`: 스케줄러 일시 중단
  - `POST /admin/scheduler/resume`: 스케줄러 재개
  - `GET/POST /admin/risk-config`: 리스크 설정 조회/갱신
  - `GET /admin/trading-symbols/available`: 거래 가능한 심볼 목록 조회
  - `GET/POST /admin/trading-symbols`: 현재 거래 심볼 조회/갱신
  - API 키 관리: `api-keys`, `api-key`, `bulk-api-keys`
- `user.py`: `/user` 프리픽스. 현재 로그인 사용자, 에이전트/스케줄러 설정 조회
- `__init__.py`: 라우터 익스포트

## 거래 심볼 설정

`GET /admin/trading-symbols` 응답:

- `symbols`: 현재 활성화된 심볼 목록
- `source`: 설정 출처 (`db` | `env_or_default`)
- `defaults`: 기본 심볼 목록

`POST /admin/trading-symbols` 요청:

- `symbols`: 설정할 심볼 목록 (최소 1개 이상)

심볼 목록은 `runtime_config` 테이블의 `trading_symbols` 섹션에 JSON 형태로 저장됩니다.

## 스케줄러 상태 조회

`GET /admin/scheduler` 응답에 포함되는 실행 상태:

- `is_running`: 스케줄러 실행 중 여부
- `paused`: 일시 중단 상태
- `last_automation_run`: 마지막 자동매매 실행 시간
- `last_review_run`: 마지막 손실 리뷰 실행 시간
- `next_automation_run`: 다음 자동매매 예상 시간 (계산값)

## DB 연결 최적화

- `TradeStore` 싱글톤 패턴 사용: `get_trade_store()` 함수로 앱 전체에서 하나의 DB 연결 풀 재사용
- 설정 저장 시 `save_runtime_configs_bulk()`로 여러 섹션을 한 트랜잭션에서 일괄 저장
- 연결 오버헤드 제거로 API 응답 시간 대폭 개선

## 유지보수 체크리스트

- 응답/요청 스키마를 바꿀 때는 프런트엔드(`static/admin.js`)와 관리자 UI 폼을 함께 수정하세요.
- 설정 저장은 DB `runtime_config` 테이블에 저장됩니다. DB 실패 시 `app/config/runtime_config.json`으로 폴백됩니다.
- 모든 라우트가 `auth.middleware` 의존성을 통해 역할 검증을 거치도록 유지하세요.
- 스케줄러 상태는 `utils/storage.py`의 `scheduler_state` 테이블에서 읽습니다.
- 세션 기반 관리자 UI는 CORS 설정에 민감합니다. `CORS_ALLOWED_ORIGINS` 환경변수(쉼표 구분 URL, 예: `https://web.example.com,https://admin.example.com`)로 허용 오리진을 구성하고, 기본값은 로컬 개발 및 Railway 기본 도메인을 포함합니다. `CORS_ALLOWED_ORIGIN_REGEX`(기본 `https://.*\.up\.railway\.app`)로 와일드카드 오리진을 추가할 수 있으며, 새 도메인을 붙일 때는 이 목록과 `static/admin.js`의 인증 fetch 로직을 동시에 확인하세요.
- **세션 쿠키**: Railway 등 프로덕션 환경에서는 `https_only=True`가 자동 설정됩니다. Uvicorn 실행 시 `--proxy-headers --forwarded-allow-ips=*` 옵션이 필요합니다 (`Dockerfile.web` 참조). 로컬 개발 시에는 HTTP에서도 쿠키가 작동합니다.
