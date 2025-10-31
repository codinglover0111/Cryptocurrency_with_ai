# Vercel 환경 변수 매핑

Vercel에 배포할 때 필요한 주요 환경 변수와 설명은 아래 표를 참고하세요.

| 이름 | 예시 값 | 설명 | 비고 |
| --- | --- | --- | --- |
| `BACKEND_API_BASE_URL` | `https://api.example.com/api` | Next.js 서버 액션이 호출할 FastAPI 백엔드 기본 URL | Vercel 프로젝트 환경 변수로 설정 |
| `NEXT_PUBLIC_APP_NAME` | `Crypto AI Control Center` | 공개적으로 노출될 앱 이름. 필요 시 UI 커스터마이징에 활용 | 선택 |
| `TESTNET` | `1` | FastAPI 백엔드에서 Bybit 테스트넷 연결 여부 | 백엔드 런타임용 |
| `MYSQL_URL` | `mysql+pymysql://user:pass@host:3306/db` | 트레이드 로그/통계를 저장하는 MySQL 연결 문자열 | 백엔드 런타임용 |
| `SQLITE_PATH` | `/data/trades.db` | SQLite를 사용할 경우의 데이터 경로 | 백엔드 런타임용 |
| `OPENAI_API_KEY` | `sk-...` | AI 의사결정 모델 호출 시 필요한 OpenAI 키 | 백엔드 런타임용 |

> **참고:** Next.js 프런트엔드는 `BACKEND_API_BASE_URL`만 필요하며, 기타 비공개 키는 서버 전용 FastAPI 애플리케이션에서 사용해야 합니다.
