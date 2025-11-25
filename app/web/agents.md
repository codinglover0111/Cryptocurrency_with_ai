# app/web - FastAPI 라우터

## 역할

- 관리자/사용자용 API 엔드포인트를 정의하여 설정 변경과 상태 조회를 제공합니다.

## 파일 가이드

- `admin.py`: `/admin` 프리픽스. 에이전트 모델 목록 조회, `agent-config`/`scheduler` 설정 조회·갱신
- `user.py`: `/user` 프리픽스. 현재 로그인 사용자, 에이전트/스케줄러 설정 조회
- `__init__.py`: 라우터 익스포트

## 유지보수 체크리스트

- 응답/요청 스키마를 바꿀 때는 프런트엔드(`static/app.js`)와 관리자 UI 폼을 함께 수정하세요.
- 설정 저장은 `app/config/runtime_config.json`에 즉시 반영되므로, 필수 필드 누락 시 기본값 병합 로직을 확인합니다.
- 모든 라우트가 `auth.middleware` 의존성을 통해 역할 검증을 거치도록 유지하세요.
