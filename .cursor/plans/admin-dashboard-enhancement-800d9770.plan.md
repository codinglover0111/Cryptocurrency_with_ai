<!-- 800d9770-0ae6-4237-a2df-18ebbd670a0c 3a986935-a8ff-4fa7-9862-8bbb2b6e359b -->
# 관리자 대시보드 기능 확장 계획

## 요약

8가지 요청 사항 중 7가지를 구현합니다 (8번 MySQL 자동 스키마 생성은 이미 구현됨).

---

## 1. 로그 뷰어 추가

**파일 변경:**

- `templates/admin.html` - 로그 섹션 UI 추가
- `static/admin.js` - 로그 로드/자동 스크롤 함수
- `static/admin.css` - 로그 뷰어 스타일
- `webapp.py` - `/admin/logs` API 엔드포인트 추가

**구현 내용:**

- trading.log 파일의 마지막 N줄을 읽어 반환하는 API
- 실시간 자동 새로고침 (10초 간격)
- 로그 레벨별 색상 구분 (INFO/WARNING/ERROR)

---

## 2. 스케줄러 중단/재개 기능

**파일 변경:**

- `main.py` - 스케줄러 상태 플래그 추가 및 체크
- `app/web/admin.py` - `/admin/scheduler/stop`, `/admin/scheduler/start` 엔드포인트
- `utils/storage.py` - scheduler_state에 `paused` 키 추가
- `templates/admin.html` - 중단/재개 버튼 UI
- `static/admin.js` - 버튼 이벤트 핸들러

**구현 내용:**

- DB의 scheduler_state 테이블에 `paused` 상태 저장
- main.py의 while 루프에서 paused 상태 확인 후 스킵

---

## 3-4. 다크/라이트 모드 전환 버튼 수정

**파일 변경:**

- `static/theme.js` - `toggleTheme` 함수가 window에 노출되어 있으나 연결 확인
- `templates/admin.html` - onclick 핸들러 확인/수정

**현재 상황:**

- `dashboard.css`에 `[data-theme="light"]` 변수 정의됨
- `theme.js`에서 `window.toggleTheme = theme.toggle` 노출됨
- admin.html의 버튼에 `onclick="toggleTheme()"` 존재

**수정 내용:**

- theme.js 로드 순서 확인 (defer 제거하여 즉시 로드)
- toggleTheme 함수 호출 방식 수정

---

## 5. 관리자 표기 변경

**파일 변경:**

- `static/admin.js` - `updateAuthUI` 함수에서 username 대신 "관리자" 표시

**수정 위치:**

```javascript
// admin.js 117-119행 수정
userInfo.innerHTML = `
  <span class="user-avatar">👤</span>
  <span class="user-name">관리자</span>
`;
```

---

## 6. 리스크 설정 UI 추가

**파일 변경:**

- `app/config/default_config.py` - RISK_CONFIG 기본값 추가
- `app/web/admin.py` - `/admin/risk-config` GET/POST 엔드포인트
- `templates/admin.html` - 리스크 설정 섹션 추가
- `static/admin.js` - 리스크 설정 로드/저장 함수
- `static/admin.css` - 설정 카드 스타일
- `app/workflows/trading.py` - 런타임 설정에서 리스크 값 로드

**설정 항목:**

| 항목 | 기본값 | 설명 |

|------|--------|------|

| default_leverage | 5 | 디폴트 레버리지 |

| max_loss_percent | 40 | 최대 손실 허용 % (레버리지 후) |

| position_allocation_percent | 20 | 포지션당 최대 할당 % |

**포지션 할당 로직 설명:**

- 100 USDT 잔고, 20% 설정 시 → 각 포지션 최대 20 USDT
- 거래 후 80 USDT 남아도 다음 포지션도 최대 20 USDT (초기 100 기준)

---

## 7. 비트코인 분석 우선 처리

**파일 변경:**

- `main.py` - 심볼 정렬하여 BTCUSDT 먼저 실행
- `app/workflows/trading.py` - BTC 분석 결과 저장/공유 로직 추가
- `utils/storage.py` - 공유 분석 결과 저장용 테이블/메서드

**구현 내용:**

1. BTCUSDT를 항상 첫 번째로 분석
2. BTC 분석 결과(추세, 시장 판단)를 DB에 저장
3. 다른 심볼 분석 시 BTC 결과를 프롬프트 컨텍스트에 포함

**새 테이블:**

```sql
CREATE TABLE shared_analysis (
  id INT AUTO_INCREMENT PRIMARY KEY,
  symbol VARCHAR(64),
  analysis_type VARCHAR(32),
  content TEXT,
  created_at DATETIME
);
```

---

## 8. MySQL 관련 (변경 없음)

현재 `utils/storage.py`의 `_ensure_schema()` 메서드가 앱 시작 시 테이블을 자동 생성합니다. 추가 작업 불필요.

---

## 파일 변경 요약

| 파일 | 변경 내용 |

|------|----------|

| `templates/admin.html` | 로그 뷰어, 스케줄러 제어 버튼, 리스크 설정 섹션 |

| `static/admin.js` | 로그 로드, 스케줄러 제어, 리스크 설정, 관리자 표기 |

| `static/admin.css` | 로그 뷰어/리스크 설정 스타일 |

| `static/theme.js` | 테마 전환 버그 수정 |

| `webapp.py` | 로그 API 엔드포인트 |

| `main.py` | 스케줄러 중단 로직, BTC 우선 처리 |

| `app/web/admin.py` | 스케줄러/리스크 설정 API |

| `app/config/default_config.py` | RISK_CONFIG 추가 |

| `app/workflows/trading.py` | BTC 분석 결과 공유 로직 |

| `utils/storage.py` | shared_analysis 테이블, scheduler paused 상태 |

### To-dos

- [ ] 로그 뷰어 추가 (admin.html, admin.js, admin.css, webapp.py)
- [ ] 스케줄러 중단/재개 기능 (main.py, admin.py, storage.py, UI)
- [ ] 다크/라이트 모드 전환 버튼 수정 (theme.js)
- [ ] 관리자 표기 변경 (admin.js)
- [ ] 리스크 설정 UI 추가 (config, admin.py, trading.py, UI)
- [ ] 비트코인 분석 우선 처리 (main.py, trading.py, storage.py)
- [ ] 다크/라이트 모드 전환 버튼 동작 수정 (theme.js)
- [ ] 관리자 ID 대신 '관리자' 표기로 변경 (admin.js)
- [ ] 스케줄러 중단/재개 기능 구현 (main.py, admin.py, storage.py)
- [ ] 로그 뷰어 UI 및 API 추가 (admin.html, admin.js, webapp.py)
- [ ] 리스크 설정 UI 및 API 추가 (레버리지, 손실 한도, 할당 %)
- [ ] 비트코인 분석 우선 처리 및 결과 공유 로직 구현
- [ ] 다크/라이트 모드 전환 버튼 동작 수정 (theme.js)
- [ ] 관리자 ID 대신 '관리자' 표기로 변경 (admin.js)
- [ ] 스케줄러 중단/재개 기능 구현 (main.py, admin.py, storage.py)
- [ ] 로그 뷰어 UI 및 API 추가 (admin.html, admin.js, webapp.py)
- [ ] 리스크 설정 UI 및 API 추가 (레버리지, 손실 한도, 할당 %)
- [ ] 비트코인 분석 우선 처리 및 결과 공유 로직 구현