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
  - `POST /admin/run-now`: 전체 심볼 즉시 분석 실행 (백그라운드)
  - `POST /admin/run-symbol`: 특정 심볼 즉시 분석 실행 (백그라운드)
  - `GET/POST /admin/risk-config`: 리스크 설정 조회/갱신
  - `GET /admin/trading-symbols/available`: 거래 가능한 심볼 목록 조회
  - `GET/POST /admin/trading-symbols`: 현재 거래 심볼 조회/갱신
  - `GET/POST /admin/prompts`: 에이전트 프롬프트 조회/갱신
  - `POST /admin/prompts/reset/{agent_type}`: 특정 에이전트 프롬프트 기본값 복원
  - `POST /admin/prompts/reset-all`: 전체 프롬프트 기본값 복원
  - API 키 관리: `api-keys`, `api-key`, `bulk-api-keys`
- `user.py`: `/user` 프리픽스. 현재 로그인 사용자, 에이전트/스케줄러 설정 조회
- `__init__.py`: 라우터 익스포트

## 인증/프록시 규칙

- 클라이언트는 Next.js API 라우트 하나로 통일하며, 브라우저가 FastAPI를 직접 호출하지 않는다.
- 모든 요청은 `Authorization: Bearer <supabase_jwt>` 헤더를 포함해야 하며, FastAPI는 Supabase JWKS로 재검증한 뒤 `admin`/`user` 역할을 강제한다.
- 역할 클레임은 `app_metadata.role`(없으면 `user_metadata.role`)을 사용한다.
- CORS는 Next.js 도메인을 `CORS_ALLOWED_ORIGINS` 또는 `CORS_ALLOWED_ORIGIN_REGEX`에 추가하고, HTTPS 환경에서는 SameSite/secure 쿠키를 유지한다.

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

## 즉시 실행 API

스케줄러 주기를 기다리지 않고 즉시 분석을 실행할 수 있습니다.

**중요**: 즉시 실행은 스케줄러 일시 중단 상태와 **무관하게** 동작합니다.

### `POST /admin/run-now`

전체 심볼에 대해 분석을 즉시 실행합니다. 백그라운드 스레드에서 실행되며, 요청 즉시 응답이 반환됩니다.

응답:

- `ok`: 성공 여부
- `message`: 결과 메시지 (심볼 개수 포함)
- `symbols`: 분석 대상 심볼 목록

### `POST /admin/run-symbol`

특정 심볼에 대해 분석을 즉시 실행합니다. 프론트엔드에서는 select 드롭다운으로 심볼을 선택합니다.

요청:

- `symbol`: 분석할 심볼 (예: `BTCUSDT`)

응답:

- `ok`: 성공 여부
- `message`: 결과 메시지
- `symbol`: 분석 대상 심볼

## 프롬프트 관리 API

에이전트 프롬프트를 관리자 UI에서 실시간으로 수정할 수 있습니다.

### `GET /admin/prompts`

모든 에이전트 프롬프트를 조회합니다.

응답:

```json
{
  "prompts": {
    "indicator": {
      "prompt": "현재 사용 중인 프롬프트",
      "default": "기본 프롬프트",
      "source": "db" | "default",
      "label": "Indicator Agent (기술적 지표 분석)",
      "variables": ["symbol", "regime", ...],
      "updated_at": "2024-01-01T00:00:00"
    },
    ...
  }
}
```

### `POST /admin/prompts`

단일 에이전트 프롬프트를 저장합니다.

요청:

```json
{
  "agent_type": "indicator",
  "prompt_template": "새로운 프롬프트 템플릿..."
}
```

### `POST /admin/prompts/reset/{agent_type}`

특정 에이전트의 프롬프트를 기본값으로 복원합니다 (DB에서 삭제).

### `POST /admin/prompts/reset-all`

모든 에이전트의 프롬프트를 기본값으로 복원합니다.

## DB 연결 최적화

- `TradeStore` 싱글톤 패턴 사용: `get_trade_store()` 함수로 앱 전체에서 하나의 DB 연결 풀 재사용
- 설정 저장 시 `save_runtime_configs_bulk()`로 여러 섹션을 한 트랜잭션에서 일괄 저장
- 연결 오버헤드 제거로 API 응답 시간 대폭 개선

## 유지보수 체크리스트

- 응답/요청 스키마를 바꿀 때는 프런트엔드(`static/admin.js`)와 관리자 UI 폼을 함께 수정하세요.
- 설정 저장은 DB `runtime_config` 테이블에 저장됩니다. DB 실패 시 `app/config/runtime_config.json`으로 폴백됩니다.
- 모든 라우트가 `auth.deps.require_user`/`require_admin` 등 Supabase JWT 검증 의존성을 거치도록 유지하세요.
- 스케줄러 상태는 `utils/storage.py`의 `scheduler_state` 테이블에서 읽습니다.
- Next.js 도메인을 `CORS_ALLOWED_ORIGINS` 환경변수(쉼표 구분 URL, 예: `https://web.example.com,https://admin.example.com`) 또는 `CORS_ALLOWED_ORIGIN_REGEX`(기본 `https://.*\.up\.railway\.app`)에 추가하고, Supabase 세션 쿠키가 전달되는지 확인하세요.
- (레거시) 세션 기반 관리자 UI를 사용할 경우 Uvicorn `--proxy-headers --forwarded-allow-ips=*` 옵션과 HTTPS-only 쿠키 설정을 유지하세요 (`Dockerfile.web` 참조).
- Supabase 전환 후에는 Next.js API 라우트가 FastAPI를 호출하므로 CORS/쿠키 설정을 Next.js 도메인 기준으로 재검토하세요.
