"""시장 데이터 수집과 지표 계산을 담당합니다."""


class MarketDataCollector:
    """OLCHV와 볼린저 밴드 데이터를 관리하는 수집기."""

    # TODO: 거래소 API 연결을 구성하고 OLCHV 데이터를 주기적으로 수집하세요.
    # TODO: 수집된 데이터로 BB 밴드를 계산하고 캐싱 전략을 설계하세요.
    # TODO: 데이터 품질 확인 및 예외 처리 로직을 구현하세요.


def bootstrap_info_agent() -> None:
    """정보 에이전트 초기화 진입점."""

    # TODO: 환경변수 로딩과 스케줄링 설정을 수행하세요.
    # TODO: 데이터 수집 파이프라인을 FastAPI/LLM 호출과 연동하세요.
