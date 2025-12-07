# static - 프런트엔드 에셋

## 역할

- 대시보드/오버레이 화면에서 사용하는 정적 JS·CSS 파일을 보관합니다.

## 파일 가이드

- `index.js`: 공개 대시보드 (index.html) 전용 스크립트 (jQuery 기반)
  - `renderMarkdownToHtml()`: 마크다운 텍스트를 HTML로 변환
  - `loadStatus()`, `loadStats()`, `loadPositions()`, `loadActivity()`: 데이터 로딩
  - `showAgentModal()`, `closeAgentModal()`: 에이전트 분석 모달
  - jQuery `$.getJSON()` 사용으로 CSP 호환
- `app.js`: 사용자 대시보드 동작, API 호출, 폼 핸들러
  - `renderMarkdownToHtml()`: 마크다운 텍스트를 HTML로 변환
- `admin.js`: 관리자 대시보드 전용 동작
  - 로그 뷰어 로드/자동 새로고침
  - 스케줄러 중단/재개 제어
  - **즉시 실행**: `runAnalysisNow()`, `openSymbolRunModal()`, `confirmRunSymbol()`
  - 리스크 설정 로드/저장
  - 거래 심볼 설정 (검색, 선택, 저장)
  - **프롬프트 설정**: `loadPrompts()`, `savePrompt()`, `resetCurrentPrompt()`, `resetAllPrompts()`
  - 스케줄러 일시중단 배지/버튼이 서버 상태와 즉시 동기화
  - 사용자명 대신 "관리자" 표시
  - **에이전트 분석 모달**: 최근 활동 클릭 시 4개 에이전트 보고서 표시
  - **이벤트 위임**: CSP 호환을 위해 인라인 onclick 대신 data-action 속성과 이벤트 위임 사용
- `admin.css`: 관리자 페이지 전용 스타일
  - 로그 뷰어 스타일 (레벨별 색상)
  - 리스크 설정 그리드
  - 스케줄러 제어 버튼
  - 심볼 선택기 칩 스타일 (BTC 강조, 선택/미선택 상태)
  - **프롬프트 편집기 스타일**: textarea, 변수 힌트, 상태 배지
  - **기본 모달 스타일 (`.modal`)**: 심볼 실행 모달 등에서 사용
  - **에이전트 분석 모달 스타일**: 탭, 카드, 분석 섹션
- `dashboard.css`: 메인 대시보드 레이아웃/카드 스타일
  - 다크/라이트 모드 테마 아이콘 전환
- `style.css`: 공통 폰트/버튼/그리드 스타일
  - `.modal-backdrop`는 `hidden` 속성 토글만으로 표시/숨김 처리 (별도 `.active` 클래스 불필요)
- `theme.js`: 다크/라이트 모드 토글 (`window.toggleTheme`, `window.theme`)
- `overlay.css` / `overlay.js`: 간소화된 오버레이 UI용 스타일/동작
- `overlay_positions.js`: 포지션 오버레이 표시 스크립트

## 에이전트 분석 모달 UX

- `showAgentModal()`(공개) / `showAgentAnalysisModal()`(관리자)는 공통으로 `#agent-modal-symbol` 요소에 `에이전트 분석 보고서(BTCUSDT)` 형식의 라벨을 채워 넣습니다. 심볼이 없으면 `System`을 fallback으로 사용해 헤더가 비지 않도록 했습니다.
- `static/style.css`, `templates/index.html` 인라인 스타일 모두 640px 이하에서 모달을 전체 폭으로 확장하고 탭을 줄바꿈/가로 스크롤로 처리하도록 media query가 추가되었습니다.
- `static/admin.css`도 동일한 media query를 갖추고 있어 모바일에서 탭 스택, 카드 간격 축소, 모달 액션 버튼이 세로로 배치됩니다.

## 심볼 설정 UI

관리자 대시보드의 "거래 심볼" 섹션에서 심볼을 관리합니다:

- 검색창에서 심볼을 빠르게 검색
- 클릭으로 심볼 선택/해제
- BTCUSDT는 특별 강조 표시 (⭐)
- 저장 시 DB에 영구 저장

## 즉시 실행 기능

관리자 대시보드 스케줄러 섹션에서 즉시 분석을 실행할 수 있습니다.

**중요**: 즉시 실행은 스케줄러 일시 중단 상태와 **무관하게** 동작합니다.

- **전체 심볼 즉시 실행**: `runAnalysisNow()` → `POST /admin/run-now`
  - 확인 대화 상자 표시 후 실행
  - 응답에 분석 대상 심볼 개수 포함
- **특정 심볼 실행**: `openSymbolRunModal()` → select에서 심볼 선택 → `confirmRunSymbol()` → `POST /admin/run-symbol`
  - 모달에서 API를 통해 심볼 목록을 직접 로드
  - 설정된 심볼이 있으면 해당 목록, 없으면 전체 사용 가능 심볼 표시

## jQuery CDN

두 HTML 파일(`index.html`, `admin.html`)은 jQuery CDN을 사용합니다:

```html
<script
  src="https://code.jquery.com/jquery-3.7.1.min.js"
  integrity="sha256-/JqT3SQfawRcv/BIHPThkBvs0OEvtFFmqPF/lYI/Cxo="
  crossorigin="anonymous"
></script>
```

CSP에 `https://code.jquery.com`이 허용되어 있습니다.

## CSP 호환 이벤트 핸들링

인라인 `onclick` 핸들러는 CSP `script-src 'self'` 정책을 위반합니다. 대신 다음 패턴을 사용합니다:

- HTML: `data-action="action-name"` 및 `data-*` 속성으로 데이터 전달
- JS (Vanilla): `document.addEventListener("click", ...)` 에서 `e.target.closest("[data-action]")`로 이벤트 위임
- JS (jQuery): `$(document).on("click", "[data-action]", ...)` 이벤트 위임

예시 (Vanilla JS):

```html
<button data-action="toggle-symbol" data-symbol="BTCUSDT">BTCUSDT</button>
```

```javascript
document.addEventListener("click", (e) => {
  const target = e.target.closest("[data-action]");
  if (target?.dataset.action === "toggle-symbol") {
    toggleSymbol(target.dataset.symbol);
  }
});
```

예시 (jQuery):

```javascript
$(document).on("click", "[data-action]", function () {
  const action = $(this).data("action");
  if (action === "toggle-symbol") {
    toggleSymbol($(this).data("symbol"));
  }
});
```

## 디버깅

문제 발생 시 브라우저 개발자 도구(F12)의 콘솔 탭을 확인하세요:

- `[admin.js]` 접두사: 스크립트 초기화 관련 로그
- `[setupEventListeners]` 접두사: 이벤트 리스너 등록 로그
- `[openSymbolRunModal]` 접두사: 심볼 실행 모달 관련 로그

주요 확인 사항:

1. 콘솔에 JavaScript 오류가 있는지 확인
2. 네트워크 탭에서 API 호출이 성공하는지 확인
3. 캐시 문제가 의심되면 Ctrl+Shift+R로 강력 새로고침

## 프롬프트 설정 UI

관리자 대시보드의 "프롬프트 설정" 섹션에서 에이전트 프롬프트를 관리합니다:

- 드롭다운에서 에이전트 선택 (Indicator, Pattern, Trend, Decision)
- 현재 프롬프트 소스 표시 (DB 저장됨 / 기본값)
- 사용 가능한 변수 목록 표시
- 프롬프트 저장/초기화 기능

### 관련 API

- `GET /admin/prompts`: 모든 프롬프트 조회
- `POST /admin/prompts`: 단일 프롬프트 저장
- `POST /admin/prompts/reset/{agent_type}`: 특정 프롬프트 초기화
- `POST /admin/prompts/reset-all`: 전체 프롬프트 초기화

## 유지보수 체크리스트

- CSP 설정은 `templates/admin.html`과 `templates/index.html`에서 정의됩니다. 외부 리소스를 추가할 때 헤더와 링크를 함께 갱신하세요.
- **인라인 onclick 사용 금지**: CSP 정책 위반 방지를 위해 이벤트 위임 패턴 사용 (공개/관리자 대시보드 모두 적용)
- API 스키마가 바뀌면 JS에서 사용하는 엔드포인트와 응답 파싱 로직을 맞춰야 합니다.
- 테마 전환은 `theme.js`에서 `data-theme` 속성으로 제어합니다. CSS 변수는 `dashboard.css`에 정의되어 있습니다.
- 심볼 설정 관련 API: `/admin/trading-symbols/available`, `/admin/trading-symbols`
- 즉시 실행 API: `/admin/run-now`, `/admin/run-symbol`
- 프롬프트 설정 API: `/admin/prompts`, `/admin/prompts/reset/{agent_type}`
