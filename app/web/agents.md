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

## 유지보수 체크리스트

- 응답/요청 스키마를 바꿀 때는 프런트엔드(`static/admin.js`)와 관리자 UI 폼을 함께 수정하세요.
- 설정 저장은 `app/config/runtime_config.json`에 즉시 반영되므로, 필수 필드 누락 시 기본값 병합 로직을 확인합니다.
- 모든 라우트가 `auth.middleware` 의존성을 통해 역할 검증을 거치도록 유지하세요.
- 스케줄러 상태는 `utils/storage.py`의 `scheduler_state` 테이블에서 읽습니다.
- 세션 기반 관리자 UI는 CORS 설정에 민감합니다. `CORS_ALLOWED_ORIGINS` 환경변수(쉼표 구분 URL, 예: `https://web.example.com,https://admin.example.com`)로 허용 오리진을 구성하고, 기본값은 로컬 개발 및 Railway 기본 도메인을 포함합니다. `CORS_ALLOWED_ORIGIN_REGEX`(기본 `https://.*\.up\.railway\.app`)로 와일드카드 오리진을 추가할 수 있으며, 새 도메인을 붙일 때는 이 목록과 `static/admin.js`의 인증 fetch 로직을 동시에 확인하세요.
