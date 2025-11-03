"""실패한 매매에 대한 리뷰 로직을 정의합니다."""

from datetime import datetime
from typing import Any


class TradeReviewService:
    """2일 지연 피드백을 생성하는 서비스."""

    # TODO: 종료된 포지션을 로드하고 결과를 시나리오와 비교하세요.
    # TODO: 예상과 실제 차이가 발생한 사유를 분석하여 저장하세요.
    # TODO: 예외 케이스를 별도로 태깅하고 리포트 포맷을 정의하세요.


def schedule_review_jobs(now: datetime | None = None) -> list[Any]:
    """리뷰 작업을 예약합니다."""

    # TODO: 2일 지연 규칙을 따르는 스케줄링 로직을 구현하세요.
    # TODO: FastAPI 백그라운드 작업 또는 외부 워커와의 연동을 고려하세요.
    return []
