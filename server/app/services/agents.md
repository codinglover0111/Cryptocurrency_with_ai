# app/services - 서비스 계층

## 역할

- 거래 데이터 수집과 저널링/리뷰 기능을 제공하는 서비스 레이어입니다. 워크플로와 웹 라우터에서 공통으로 사용됩니다.

## 파일 가이드

- `market_data.py`: CCXT/Bybit 유틸을 이용해 OHLCV, 포지션, 주문 데이터를 조회하고 CSV/이미지용 포맷으로 가공
- `journal.py`: `JournalService`로 트레이드/액션 로그 저장, 리뷰용 리포트 생성, LLM 요약 호출 (LLM은 `resolve_ai_provider()`로 OpenRouter/OpenAI 등 자동 감지)
- `__init__.py`: 익스포트 편의

## 유지보수 체크리스트

- 데이터 스키마를 바꿀 때는 `utils/storage.py`와 `app/workflows/trading.py`의 호출부를 함께 검토하세요.
- 리뷰/요약에 사용하는 AI 호출은 비용이 높을 수 있으니 주기(`TRADE_REVIEW_WAIT_HOURS` 등)를 조정해 관리합니다.
- 마켓 데이터 응답 포맷이 변경되면 그래프 노드 입력(특히 Indicator/Pattern)도 업데이트해야 합니다.
