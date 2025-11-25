# templates - Jinja2 템플릿

## 역할

- FastAPI가 렌더링하는 HTML 템플릿을 보관합니다. 정적 에셋(`static/`)과 함께 대시보드/오버레이 UI를 구성합니다.

## 파일 가이드

- `index.html`: 메인 대시보드, 관리자/사용자 설정 카드와 상태 표시
- `overlay.html`: 심플 오버레이 뷰
- `overlay_positions.html`: 포지션 중심 오버레이 뷰

## 유지보수 체크리스트

- CSP/폰트/스타일 링크는 헤더에서 관리합니다. 외부 리소스를 추가하면 `static/agents.md`의 안내에 따라 CSP도 업데이트하세요.
- API 응답 필드가 바뀌면 템플릿의 데이터 바인딩과 JS 처리 로직을 동기화해야 합니다.
