# templates - Jinja2 템플릿

## 역할

- FastAPI가 렌더링하는 HTML 템플릿을 보관합니다. 정적 에셋(`static/`)과 함께 대시보드/오버레이 UI를 구성합니다.

## 파일 가이드

- `index.html`: 공개 대시보드 (로그인 불필요), 잔고/포지션/저널 조회
  - **에이전트 분석 모달**: 최근 활동 클릭 시 4개 에이전트 분석 보고서 표시
  - 스크립트는 `static/index.js`로 외부화 (CSP 호환)
  - jQuery CDN 사용 (`https://code.jquery.com`)
- `admin.html`: 관리자 대시보드 (로그인 필요)
  - 에이전트 LLM 설정 카드
  - 스케줄러 설정 및 중단/재개 버튼
  - **즉시 실행 버튼**: 전체 심볼/특정 심볼 즉시 분석 실행
  - **심볼 선택 모달**: 특정 심볼 실행 시 심볼 선택 UI
  - 리스크 설정 (레버리지, 손실 한도, 할당 %)
  - 거래 심볼 설정 (검색 가능한 멀티셀렉트)
  - **프롬프트 설정**: 에이전트별 프롬프트 편집기 (textarea, 변수 힌트)
  - 로그 뷰어 (레벨별 필터링)
  - 다크/라이트 모드 토글
- `overlay.html`: 심플 오버레이 뷰
- `overlay_positions.html`: 포지션 중심 오버레이 뷰

## 에이전트 분석 모달

공개/관리자 대시보드에서 "최근 활동" 항목을 클릭하면 에이전트 분석 모달이 표시됩니다:

- **헤더 라벨**: 선택한 저널의 심볼을 `에이전트 분석 보고서(BTCUSDT)` 형태로 즉시 노출하여 어떤 심볼의 보고서인지 혼동이 없습니다.
- **탭 구성**: Indicator / Pattern / Trend / Decision
- **데이터 소스**: 저널 `meta.agents` 필드
- **마크다운 지원**: 에이전트 분석 텍스트가 마크다운으로 렌더링
- **반응형 레이아웃**: 640px 이하 화면에서는 모달 폭을 100%로 확장하고 탭 버튼을 가로 스크롤/줄바꿈으로 전환하여 모바일에서도 동일한 경험을 제공합니다.

## jQuery CDN

두 HTML 파일은 jQuery CDN을 사용합니다:

```html
<script
  src="https://code.jquery.com/jquery-3.7.1.min.js"
  integrity="sha256-/JqT3SQfawRcv/BIHPThkBvs0OEvtFFmqPF/lYI/Cxo="
  crossorigin="anonymous"
></script>
```

CSP에 `https://code.jquery.com`이 허용되어 있습니다.

## 유지보수 체크리스트

- CSP/폰트/스타일 링크는 헤더에서 관리합니다. 외부 리소스를 추가하면 `static/agents.md`의 안내에 따라 CSP도 업데이트하세요.
- API 응답 필드가 바뀌면 템플릿의 데이터 바인딩과 JS 처리 로직을 동기화해야 합니다.
- 공개 페이지(`index.html`)는 `static/index.js`를 사용합니다 (jQuery 기반).
- 관리자 페이지(`admin.html`)는 `static/admin.js`와 `static/admin.css`를 사용합니다.
- 에이전트 분석 모달은 `meta.agents` 필드에 의존합니다. 스키마 변경 시 `app/workflows/trading.py`와 동기화하세요.
