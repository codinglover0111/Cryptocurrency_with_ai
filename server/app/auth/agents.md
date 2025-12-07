# app/auth - 인증/권한 (Supabase 기반)

## 역할

- Supabase JWT를 검증하고 `role` 클레임을 FastAPI 요청 컨텍스트에 주입한다.
- Next.js API 라우트에서 전달된 `Authorization: Bearer <supabase_jwt>`를 재검증해 `admin`/`user` 권한을 강제한다.
- 관리자/사용자 메타데이터를 Supabase에서 조회(서비스 롤 키 사용)하고 필요한 경우 로컬 캐싱한다.

## 인증 흐름 (Next.js → FastAPI)

1. Next.js 클라이언트가 Supabase Auth(이메일/비밀번호, 선택적 OAuth)로 로그인/회원가입/비밀번호 재설정을 처리한다.
2. Next.js API 라우트에서 Supabase 세션을 읽어 만료 여부를 확인한 뒤 FastAPI로 프록시하며 `Authorization: Bearer <supabase_jwt>` 헤더를 첨부한다.
3. FastAPI `verify_supabase_token`이 Supabase JWKS(`SUPABASE_JWKS_URL`)로 JWT를 검증하고 `role` 클레임(`app_metadata.role` 우선, 없으면 `user_metadata.role`)을 추출한다.
4. `require_user`/`require_admin` 의존성이 `request.state.user`에 id/email/role을 저장하고 역할을 검사한다.

## 파일 가이드

- `supabase.py`: JWKS 가져오기/캐싱, JWT 검증 헬퍼, 서비스 롤 키 기반 프로필 조회 유틸.
- `deps.py`: `verify_supabase_token`, `get_current_user`, `require_user`, `require_admin` FastAPI 의존성.
- `roles.py`(옵션): 역할 매핑/기본값 정의 (`role` 없을 경우 `user` 기본).
- `__init__.py`: 익스포트 편의.

## 필수 환경변수

- `SUPABASE_URL`
- `SUPABASE_JWKS_URL`
- `SUPABASE_ANON_KEY` (Next.js 클라이언트)
- `SUPABASE_SERVICE_ROLE_KEY` (FastAPI에서 역할 조회/관리 작업)

## 유지보수 체크리스트

- JWT 만료/서명 오류 시 401을 반환하고, Next.js API 라우트는 로그아웃/재로그인으로 유도해야 한다.
- 역할은 `app_metadata.role`을 우선 사용하고 없으면 `user_metadata.role` → 기본 `user`.
- Supabase JWKS는 TTL을 두고 캐싱하되 키 회전 시 즉시 새로고침할 수 있어야 한다.
