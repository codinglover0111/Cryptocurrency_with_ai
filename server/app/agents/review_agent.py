"""Review Agent 구현 - 거래 리뷰 분석용."""

from __future__ import annotations

import logging
from typing import Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

LOGGER = logging.getLogger(__name__)


class ReviewResult(BaseModel):
    """거래 리뷰 결과 스키마."""

    summary: str = Field(description="리뷰 요약 (1-2문장)")
    key_factors: list[str] = Field(
        description="핵심 요인 목록 (3-5개 bullet)",
        default_factory=list,
    )
    improvements: list[str] = Field(
        description="개선 사항 목록 (3-5개 bullet)",
        default_factory=list,
    )
    was_decision_appropriate: bool = Field(
        description="진입 결정이 적절했는지 여부",
        default=False,
    )
    exit_timing_assessment: str = Field(
        description="청산 타이밍 평가 (적절/조기/지연)",
        default="unknown",
    )


REVIEW_TEMPLATE = """당신은 암호화폐 트레이딩 {role_type}입니다. 한국어로 답하세요.

[거래 정보]
- 심볼: {symbol}
- 포지션: {side}
- 손익: {pnl} USDT
- 진입가: {entry_price}
- 청산가: {close_price}
- TP: {tp}
- SL: {sl}
- 진입 시간: {open_ts}
- 청산 시간: {close_ts}

[분석 출력 양식]
- 진입 결정이 적절했는지 여부
- 임의로 익절/손절 할 수 있었는지
- 너무 욕심을 부린 것은 아닌지
- 차트 시나리오가 예상대로 진행되었는지
[/분석 출력 양식]

[포지션 진입 기록]
{entry_notes}

[CSV 데이터 - 진입~청산 구간]
{csv_between}

[CSV 데이터 - 청산 이후 {wait_hours}시간]
{csv_after_close}

작업:
1. {analysis_task}
2. 청산 후 {wait_hours}시간 경과한 가격 흐름도 함께 참고해 판단의 적절성을 평가하세요.
3. 600자 이내로 작성하세요."""


class ReviewAgent:
    """거래 리뷰 에이전트 - 손실/수익 거래 분석용."""

    def __init__(self, llm: BaseChatModel, use_structured_output: bool = True) -> None:
        """ReviewAgent 초기화.

        Args:
            llm: LangChain 호환 LLM 인스턴스
            use_structured_output: 구조화된 출력 사용 여부 (False면 텍스트 출력)
        """
        self.llm = llm
        self.use_structured_output = use_structured_output
        self.prompt = ChatPromptTemplate.from_template(REVIEW_TEMPLATE)

        if use_structured_output:
            self.chain = self.prompt | llm.with_structured_output(
                ReviewResult, strict=True
            )
        else:
            self.chain = self.prompt | llm

    def review_trade(
        self,
        *,
        symbol: str,
        side: str,
        pnl: float,
        entry_price: Optional[float],
        close_price: float,
        tp: Optional[float],
        sl: Optional[float],
        open_ts: str,
        close_ts: str,
        csv_between: str = "(no data)",
        csv_after_close: str = "(no data)",
        entry_notes: str = "(관련 기록 없음)",
        wait_hours: int = 48,
    ) -> ReviewResult | str:
        """거래 리뷰를 생성합니다.

        Args:
            symbol: 거래 심볼
            side: 포지션 방향 (long/short)
            pnl: 실현 손익
            entry_price: 진입가 (없으면 None)
            close_price: 청산가
            tp: 목표가
            sl: 손절가
            open_ts: 진입 시간 문자열
            close_ts: 청산 시간 문자열
            csv_between: 진입~청산 구간 OHLCV CSV
            csv_after_close: 청산 이후 OHLCV CSV
            entry_notes: 진입 시 의사결정/액션 기록
            wait_hours: 청산 후 대기 시간

        Returns:
            ReviewResult (구조화) 또는 str (텍스트)
        """
        is_loss = pnl < 0
        role_type = "손실 원인 분석가" if is_loss else "수익 요인 분석가"

        if is_loss:
            analysis_task = (
                "손실 발생의 핵심 원인과 재발 방지를 위한 교훈/체크리스트를 "
                "3~5개 불릿으로 제시하세요."
            )
        else:
            analysis_task = (
                "수익 발생의 핵심 요인과 재현 방법, 리스크 관리/익절·손절 "
                "개선 포인트를 3~5개 불릿으로 제시하세요."
            )

        input_data = {
            "role_type": role_type,
            "symbol": symbol,
            "side": side,
            "pnl": pnl,
            "entry_price": entry_price if entry_price else "불명",
            "close_price": close_price,
            "tp": tp if tp else "설정 안 됨",
            "sl": sl if sl else "설정 안 됨",
            "open_ts": open_ts,
            "close_ts": close_ts,
            "csv_between": csv_between,
            "csv_after_close": csv_after_close,
            "entry_notes": entry_notes,
            "wait_hours": wait_hours,
            "analysis_task": analysis_task,
        }

        try:
            result = self.chain.invoke(input_data)
            if self.use_structured_output:
                return result
            # 텍스트 출력인 경우
            content = result.content if hasattr(result, "content") else str(result)
            return content.strip()[:2000]
        except Exception as exc:
            LOGGER.error("ReviewAgent 실행 실패: %s", exc)
            if self.use_structured_output:
                return ReviewResult(
                    summary="리뷰 생성 실패",
                    key_factors=[],
                    improvements=[],
                    was_decision_appropriate=False,
                    exit_timing_assessment="unknown",
                )
            return "리뷰 생성 실패"

