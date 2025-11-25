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
  - API 키 관리: `api-keys`, `api-key`, `bulk-api-keys`
- `user.py`: `/user` 프리픽스. 현재 로그인 사용자, 에이전트/스케줄러 설정 조회
- `__init__.py`: 라우터 익스포트

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
