"""차트 패턴을 기반으로 시나리오를 도출합니다."""

from dataclasses import dataclass
from typing import Iterable


@dataclass
class Scenario:
    """생성된 시나리오 모델."""

    name: str
    description: str
    resistance_levels: list[float]
    support_levels: list[float]


class ScenarioAnalyzer:
    """지지/저항선과 패턴을 평가하는 분석기."""

    # TODO: 실시간/과거 데이터를 입력받아 지지선과 저항선을 계산하세요.
    # TODO: 현재 차트 패턴을 분류하고 가능한 시나리오를 생성하세요.
    # TODO: 생성된 시나리오를 판단 에이전트가 소비할 수 있는 구조로 직렬화하세요.


def generate_scenarios(data: Iterable[float]) -> list[Scenario]:
    """시나리오 생성 파이프라인."""

    # TODO: 입력 데이터 검증과 예외 처리를 구현하세요.
    # TODO: 분석 결과를 기반으로 Scenario 리스트를 작성하세요.
    return []
