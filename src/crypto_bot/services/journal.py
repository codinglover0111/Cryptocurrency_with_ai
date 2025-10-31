"""Trade journal helpers."""

# pylint: disable=broad-except
# ruff: noqa: E722, BLE001
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from crypto_bot.llm.provider import AIProvider
from crypto_bot.persistence.store import TradeStore

from crypto_bot.core.symbols import contract_to_spot_symbol
from crypto_bot.services.market_data import ohlcv_csv_between


LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class JournalService:
    """Encapsulates journal and review operations on top of TradeStore."""

    store: TradeStore
    ai_provider: Optional[AIProvider] = None

    def _ensure_ai_provider(self) -> AIProvider:
        if self.ai_provider is None:
            self.ai_provider = AIProvider()
        return self.ai_provider

    def _collect_entry_notes(
        self,
        contract_symbol: str,
        *,
        open_ts: pd.Timestamp,
        close_ts: pd.Timestamp,
    ) -> str:
        """Gather decision/action journals around the position entry."""

        try:
            open_ts_pd = pd.Timestamp(open_ts)
        except Exception:
            return ""

        try:
            close_ts_pd = pd.Timestamp(close_ts)
        except Exception:
            close_ts_pd = open_ts_pd

        if open_ts_pd.tz is None:
            open_ts_pd = open_ts_pd.tz_localize("UTC")
        if close_ts_pd.tz is None:
            close_ts_pd = close_ts_pd.tz_localize("UTC")

        since_ts = open_ts_pd - pd.Timedelta(minutes=30)
        until_ts = close_ts_pd + pd.Timedelta(minutes=5)

        try:
            df = self.store.fetch_journals(
                symbol=contract_symbol,
                types=["decision", "action"],
                since_ts=since_ts,
                until_ts=until_ts,
                limit=50,
                ascending=True,
            )
        except Exception:
            df = None

        if df is None or getattr(df, "empty", True):
            return ""

        lines = []
        for _, row in df.sort_values("ts").iterrows():
            ts_val = row.get("ts")
            try:
                ts_pd = pd.Timestamp(ts_val)
                if ts_pd.tz is None:
                    ts_pd = ts_pd.tz_localize("UTC")
                ts_str = ts_pd.tz_convert("Asia/Seoul").strftime("%m-%d %H:%M:%S")
            except Exception:
                ts_str = str(ts_val)

            entry_type = (row.get("entry_type") or "").strip()
            reason = (row.get("reason") or "").strip()
            content = (row.get("content") or "").strip()

            if len(content) > 200:
                content = content[:200] + "..."

            line_parts = [f"[{ts_str}]"]
            if entry_type:
                line_parts.append(f"({entry_type})")
            if reason:
                line_parts.append(reason)
            if content:
                line_parts.append(f"| {content}")

            lines.append(" ".join(line_parts).strip())

        return "\n".join(lines)

    def format_trade_reviews_for_prompt(self, contract_symbol: str) -> str:
        """Return the latest trade reviews as prompt-ready text."""
        try:
            df = self.store.fetch_journals(
                symbol=contract_symbol, types=["review"], limit=5, ascending=False
            )
            if df is None or getattr(df, "empty", True):
                return ""

            lines = []
            for _, row in df.iterrows():
                ts = row.get("ts")
                ts_str = (
                    ts.strftime("%Y-%m-%d %H:%M:%S")
                    if hasattr(ts, "strftime")
                    else str(ts)
                )
                reason = row.get("reason") or ""
                content = row.get("content") or ""
                if len(content) > 500:
                    content = content[:500]
                lines.append(f"[{ts_str}] {reason} | {content}")
            return "\n".join(lines)
        except Exception:
            return ""

    def review_losing_trades(self, since_minutes: int = 600) -> None:
        """Review recent losing trades and store AI-generated feedback."""
        try:
            trades_df = self.store.load_trades()
            if trades_df is None or getattr(trades_df, "empty", True):
                return

            trades_df["ts"] = pd.to_datetime(trades_df["ts"], errors="coerce", utc=True)

            now_utc = pd.Timestamp.now(tz="UTC")
            since_ts = now_utc - pd.Timedelta(minutes=int(since_minutes))

            closed_recent = trades_df[
                (trades_df["status"] == "closed")
                & (trades_df["pnl"].notna())
                & (trades_df["pnl"] <= 0)
                & (trades_df["ts"] >= since_ts)
            ].copy()

            if getattr(closed_recent, "empty", True):
                return

            recent_reviews = {}
            try:
                review_df = self.store.fetch_journals(
                    types=["review"], limit=100, ascending=False
                )
                if review_df is not None and not getattr(review_df, "empty", True):
                    for _, row in review_df.iterrows():
                        sym = row.get("symbol")
                        meta = row.get("meta")
                        if isinstance(meta, str):
                            try:
                                meta = json.loads(meta)
                            except Exception:
                                meta = None
                        if not isinstance(meta, dict):
                            continue
                        key = (sym, meta.get("closed_ts"))
                        recent_reviews[key] = True
            except Exception:
                pass

            ai_provider = self._ensure_ai_provider()

            for _, row in closed_recent.sort_values("ts").iterrows():
                contract_symbol = str(row.get("symbol"))
                side = row.get("side")
                close_ts = row.get("ts")
                pnl = float(row.get("pnl") or 0.0)

                closed_key = (contract_symbol, pd.Timestamp(close_ts).isoformat())
                if recent_reviews.get(closed_key):
                    continue

                opened_df = trades_df[
                    (trades_df["symbol"].astype(str) == contract_symbol)
                    & (trades_df["status"] == "opened")
                    & (trades_df["ts"] <= close_ts)
                ]

                if not getattr(opened_df, "empty", True):
                    open_ts = opened_df["ts"].max()
                    entry_price = float(
                        opened_df.sort_values("ts").iloc[-1].get("price") or 0
                    )
                else:
                    open_ts = pd.Timestamp(close_ts) - pd.Timedelta(hours=2)
                    entry_price = None

                spot_symbol = contract_to_spot_symbol(contract_symbol)
                since_ms = int(pd.Timestamp(open_ts).timestamp() * 1000)
                until_ms = int(pd.Timestamp(close_ts).timestamp() * 1000)
                csv_1m = ohlcv_csv_between(spot_symbol, "1m", since_ms, until_ms)

                open_ts_str = pd.Timestamp(open_ts).strftime("%Y-%m-%d %H:%M:%S UTC")
                close_ts_str = pd.Timestamp(close_ts).strftime("%Y-%m-%d %H:%M:%S UTC")
                is_loss = pnl < 0
                role_line = "손실 원인 분석가" if is_loss else "수익 요인 분석가"
                task_line = (
                    "아래 CSV_1m 구간의 가격 흐름을 참고하여, 손실 발생의 핵심 원인과 재발 방지를 위한 교훈/체크리스트를 3~5개 불릿으로 제시하세요. 600자 이내.\n"
                    if is_loss
                    else "아래 CSV_1m 구간의 가격 흐름을 참고하여, 수익 발생의 핵심 요인과 재현 방법, 리스크 관리/익절·손절 개선 포인트를 3~5개 불릿으로 제시하세요. 600자 이내.\n"
                )

                prompt = (
                    f"당신은 암호화폐 트레이딩 {role_line}입니다. 한국어로 간결히 답하세요.\n"
                    f"심볼: {contract_symbol} (spot={spot_symbol})\n"
                    f"포지션: {side}, 손익: {pnl} USDT\n"
                    f"기간: {open_ts_str} ~ {close_ts_str}\n"
                    + (f"진입가(추정): {entry_price}\n" if entry_price else "")
                    + task_line
                )

                if is_loss:
                    entry_notes = self._collect_entry_notes(
                        contract_symbol,
                        open_ts=open_ts,
                        close_ts=close_ts,
                    )
                    prompt += (
                        "포지션 진입 판단/액션 기록:\n"
                        + (entry_notes if entry_notes else "(관련 기록 없음)")
                        + "\n"
                    )

                prompt += "[CSV_1m_BETWEEN]\n" + (csv_1m or "(no data)")

                delays = [5, 10, 60]
                max_attempts = len(delays) + 1
                review_text = None

                for attempt in range(1, max_attempts + 1):
                    try:
                        review_text = ai_provider.decide(prompt)
                        break
                    except Exception as exc:
                        LOGGER.error(
                            "Trade review LLM failed (attempt %s/%s): %s",
                            attempt,
                            max_attempts,
                            exc,
                        )
                        if attempt == max_attempts:
                            break
                        wait_seconds = delays[attempt - 1]
                        LOGGER.info(
                            "Trade review LLM 재시도 %s/%s: %s초 후 재시도",
                            attempt + 1,
                            max_attempts,
                            wait_seconds,
                        )
                        time.sleep(wait_seconds)

                if review_text is None:
                    review_text = "저널 조회에 오류"

                try:
                    self.store.record_journal(
                        {
                            "symbol": contract_symbol,
                            "entry_type": "review",
                            "content": review_text or "리뷰 생성 실패",
                            "reason": ("loss_review" if pnl < 0 else "win_review"),
                            "meta": {
                                "closed_ts": pd.Timestamp(close_ts).isoformat(),
                                "opened_ts": pd.Timestamp(open_ts).isoformat(),
                                "pnl": pnl,
                                "side": side,
                                "spot_symbol": spot_symbol,
                                "timeframe": "1m",
                            },
                        }
                    )
                except Exception as exc:
                    LOGGER.error("Journal review write failed: %s", exc)
        except Exception as exc:
            LOGGER.error("review_losing_trades failed: %s", exc)
