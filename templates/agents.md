# templates - Jinja2 템플릿

## 역할

- FastAPI가 렌더링하는 HTML 템플릿을 보관합니다. 정적 에셋(`static/`)과 함께 대시보드/오버레이 UI를 구성합니다.

## 파일 가이드

- `index.html`: 공개 대시보드 (로그인 불필요), 잔고/포지션/저널 조회
- `admin.html`: 관리자 대시보드 (로그인 필요)
  - 에이전트 LLM 설정 카드
  - 스케줄러 설정 및 중단/재개 버튼
  - 리스크 설정 (레버리지, 손실 한도, 할당 %)
  - 로그 뷰어 (레벨별 필터링)
  - 다크/라이트 모드 토글
- `overlay.html`: 심플 오버레이 뷰
- `overlay_positions.html`: 포지션 중심 오버레이 뷰

## 유지보수 체크리스트

- CSP/폰트/스타일 링크는 헤더에서 관리합니다. 외부 리소스를 추가하면 `static/agents.md`의 안내에 따라 CSP도 업데이트하세요.
- API 응답 필드가 바뀌면 템플릿의 데이터 바인딩과 JS 처리 로직을 동기화해야 합니다.
- 관리자 페이지(`admin.html`)는 `static/admin.js`와 `static/admin.css`를 사용합니다.
