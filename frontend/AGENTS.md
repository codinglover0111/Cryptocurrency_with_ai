# frontend - Next.js + Supabase + FastAPI

## 역할

- Next.js(App Router) 프런트엔드가 Supabase 인증을 담당하고, API 라우트로 FastAPI를 프록시하여 관리자/사용자 UI를 제공한다.

## 아키텍처 개요

- **Auth**: Supabase Auth 이메일/비밀번호 기본, OAuth 선택적 활성화. `@supabase/auth-helpers-nextjs`로 서버/클라이언트 세션을 일관되게 읽는다.
- **API 프록시**: `app/api/proxy/[...path]` 라우트 핸들러가 Supabase 세션을 검증하고 `Authorization: Bearer <supabase_jwt>` 헤더를 붙여 FastAPI로 프록시한다. 브라우저는 FastAPI를 직접 호출하지 않는다.
- **미들웨어**: `/admin` 등 보호 라우트는 서버 미들웨어에서 세션을 확인하고 `role=admin`이 아닐 경우 로그인/권한 오류로 리다이렉트한다.
- **렌더링 전략**: 공개 대시보드는 SSR(Server Components)로 초기 로드, 관리자 화면은 클라이언트 컴포넌트 + SWR/React Query로 실시간 갱신.
- **역할 클레임**: `app_metadata.role` 우선, 없으면 `user_metadata.role`, 기본 `user`.

## 주요 페이지/기능

- `/` 공개 대시보드: 최근 활동, 에이전트 분석 모달(Indicator/Pattern/Trend/Decision 탭, 마크다운 렌더링), 모바일 전체화면 모달.
- `/login`, `/signup`, `/reset-password`: Supabase Auth 폼, 에러/로딩 핸들링 포함.
- `/admin`: 스케줄러 상태/일시중단 뱃지, run-now/run-symbol, 리스크/모델/프롬프트 설정, 로그 뷰어.
- 공통 UI: 다크/라이트 모드, 심볼 라벨 표시(`에이전트 분석 보고서(BTCUSDT)` 형식), 새로고침 후 상태 보존.

## 데이터 페칭 규칙

- 모든 데이터 요청은 Next.js API 라우트를 통해 이루어지며 FastAPI로 프록시된다.
- Admin 영역은 SWR/React Query로 상태를 폴링/갱신하고, 공개 영역은 SSR + 클라이언트 하이드레이션을 사용한다.
- 마크다운 렌더링 규칙은 기존 `static/app.js`와 동일하게 적용한다.

## 환경변수

- `NEXT_PUBLIC_SUPABASE_URL`, `NEXT_PUBLIC_SUPABASE_ANON_KEY`
- `SUPABASE_SERVICE_ROLE_KEY` (서버 전용: 라우트 핸들러/서버 액션에서만 사용)
- `SUPABASE_JWKS_URL` (백엔드 검증과 동일하게 맞춘다)
- `FASTAPI_BASE_URL` (프록시 대상 FastAPI origin, 예: `http://localhost:8000` 또는 배포 도메인)

## 체크리스트

- API 라우트는 세션 만료/서명 오류 시 401을 반환하고, 클라이언트는 재로그인으로 유도한다.
- 관리자 라우트 접근 시 `role=admin`이 아니면 즉시 차단한다.
- SSR 페이지에서 필요한 데이터는 API 라우트를 통해 서버 컴포넌트에서 prefetch하고, 클라이언트 전환 시 동일 캐시를 재사용한다.
- 쿠키 SameSite/secure 설정이 프로덕션 도메인에서 올바르게 작동하는지 점검한다.
