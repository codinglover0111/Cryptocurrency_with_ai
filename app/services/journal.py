"""Trade journal helpers."""

# pylint: disable=broad-except
# ruff: noqa: E722, BLE001
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from utils.ai_provider import AIProvider
from utils.storage import TradeStore

from app.core.symbols import contract_to_spot_symbol
from app.services.market_data import ohlcv_csv_between


LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class JournalService:
    """Encapsulates journal and review operations on top of TradeStore."""

    store: TradeStore
    ai_provider: Optional[AIProvider] = None

    def _ensure_ai_provider(self) -> AIProvider:
        """Return a cached AI provider instance, creating one if needed."""

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

    def _load_recent_review_keys(self) -> dict[tuple[str, str], bool]:
        """Collect recently processed reviews to prevent duplicate analysis."""

        recent_reviews: dict[tuple[str, str], bool] = {}
        try:
            review_df = self.store.fetch_journals(
                types=["review"], limit=500, ascending=False
            )
            if review_df is None or getattr(review_df, "empty", True):
                return recent_reviews

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
                closed_ts = meta.get("closed_ts")
                if sym and closed_ts:
                    recent_reviews[(sym, closed_ts)] = True
        except Exception:
            LOGGER.warning("Failed to load recent review keys", exc_info=True)

        return recent_reviews

    def _resolve_open_context(
        self,
        trades_df: pd.DataFrame,
        contract_symbol: str,
        close_ts: pd.Timestamp,
    ) -> tuple[pd.Timestamp, Optional[float]]:
        """Infer the opening timestamp and entry price for a closed position."""

        try:
            opened_df = trades_df[
                (trades_df["symbol"].astype(str) == contract_symbol)
                & (trades_df["status"] == "opened")
                & (trades_df["ts"] <= close_ts)
            ]

            if not getattr(opened_df, "empty", True):
                open_ts = opened_df["ts"].max()
                try:
                    entry_price = float(
                        opened_df.sort_values("ts").iloc[-1].get("price") or 0
                    )
                    if entry_price == 0:
                        entry_price = None
                except Exception:
                    entry_price = None
                return open_ts, entry_price
        except Exception:
            LOGGER.warning("Failed to resolve open context", exc_info=True)

        return pd.Timestamp(close_ts) - pd.Timedelta(hours=2), None

    def _build_review_prompt(
        self,
        *,
        contract_symbol: str,
        spot_symbol: str,
        side: str,
        pnl: float,
        open_ts: pd.Timestamp,
        close_ts: pd.Timestamp,
        entry_price: Optional[float],
        csv_1m: Optional[str],
        csv_after_close: Optional[str] = None,
        is_loss: bool,
        entry_notes: str = "",
    ) -> str:
        """Construct the LLM prompt for reviewing a closed trade."""

        open_ts_str = pd.Timestamp(open_ts).strftime("%Y-%m-%d %H:%M:%S UTC")
        close_ts_str = pd.Timestamp(close_ts).strftime("%Y-%m-%d %H:%M:%S UTC")
        role_line = "손실 원인 분석가" if is_loss else "수익 요인 분석가"
        task_line = (
            "아래 CSV_1m 구간의 가격 흐름을 참고하여, 손실 발생의 핵심 원인과 재발 방지를 위한 교훈/체크리스트를 3~5개 불릿으로 제시하세요. 600자 이내.\n"
            if is_loss
            else "아래 CSV_1m 구간의 가격 흐름을 참고하여, 수익 발생의 핵심 요인과 재현 방법, 리스크 관리/익절·손절 개선 포인트를 3~5개 불릿으로 제시하세요. 600자 이내.\n"
        )
        task_line += (
            "청산 후 1시간 경과한 가격 흐름도 함께 참고해 판단의 적절성을 평가하세요.\n"
        )

        entry_line = f"진입가(추정): {entry_price}\n" if entry_price else ""
        prompt = (
            f"당신은 암호화폐 트레이딩 {role_line}입니다. 한국어로 답하세요.\n"
            """[Explain 출력 양식]
- 심볼
- 진입가,TP가격,SL가격
# 판단 내용
- 임의로 익절 할 수 있었는가?
- 임의로 손절 할 수 있었는가?
- 너무 욕심을 부려서 익절/손절 할 수 없었는가?
- 차트의 시나리오는 내가 예상한 시나리오로 흘러갔는가?
[/Explain 출력 양식]
"""
            f"심볼: {contract_symbol} (spot={spot_symbol})\n"
            f"포지션: {side}, 손익: {pnl} USDT\n"
            f"기간: {open_ts_str} ~ {close_ts_str}\n" + entry_line + task_line
        )

        if is_loss:
            prompt += (
                "포지션 진입 판단/액션 기록:\n"
                + (entry_notes if entry_notes else "(관련 기록 없음)")
                + "\n"
            )

        prompt += "[CSV_1m_BETWEEN]\n" + (csv_1m or "(no data)") + "\n"
        prompt += "[CSV_1m_AFTER_CLOSE]\n" + (csv_after_close or "(no data)")

        return prompt

    def _record_review_entry(
        self,
        *,
        contract_symbol: str,
        review_text: Optional[str],
        pnl: float,
        side: str,
        open_ts: pd.Timestamp,
        close_ts: pd.Timestamp,
        spot_symbol: str,
    ) -> None:
        """Persist the LLM review output into the journal store."""

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

    def list_pending_reviews(
        self,
        *,
        wait_hours: int = 4,
        since_hours: int = 24,
    ) -> list[dict[str, object]]:
        """Return closed-loss trades waiting for review availability."""

        try:
            trades_df = self.store.load_trades()
        except Exception:
            return []

        if trades_df is None or getattr(trades_df, "empty", True):
            return []

        try:
            trades_df["ts"] = pd.to_datetime(trades_df["ts"], errors="coerce", utc=True)
        except Exception:
            trades_df["ts"] = pd.to_datetime(trades_df["ts"], errors="coerce", utc=True)

        now_utc = pd.Timestamp.now(tz="UTC")
        since_ts = now_utc - pd.Timedelta(hours=float(since_hours))
        wait_delta = pd.Timedelta(hours=float(wait_hours))

        closed_df = trades_df[
            (trades_df["status"] == "closed")
            & (trades_df["pnl"].notna())
            & (trades_df["pnl"] <= 0)
            & (trades_df["ts"] >= since_ts)
        ].copy()

        if getattr(closed_df, "empty", True):
            return []

        reviewed_keys: dict[tuple[str, str], bool] = {}
        try:
            review_df = self.store.fetch_journals(
                types=["review"], limit=500, ascending=False
            )
        except Exception:
            review_df = None

        if review_df is not None and not getattr(review_df, "empty", True):
            for _, row in review_df.iterrows():
                symbol = row.get("symbol")
                meta = row.get("meta")
                if isinstance(meta, str):
                    try:
                        meta = json.loads(meta)
                    except Exception:
                        meta = None
                if not isinstance(meta, dict):
                    continue
                closed_ts_val = meta.get("closed_ts")
                if not closed_ts_val:
                    continue
                try:
                    closed_ts_pd = pd.Timestamp(closed_ts_val)
                    if closed_ts_pd.tz is None:
                        closed_ts_pd = closed_ts_pd.tz_localize("UTC")
                    else:
                        closed_ts_pd = closed_ts_pd.tz_convert("UTC")
                except Exception:
                    continue
                key = (str(symbol), closed_ts_pd.isoformat())
                reviewed_keys[key] = True

        pending: list[dict[str, object]] = []

        for _, row in closed_df.sort_values("ts").iterrows():
            symbol_raw = row.get("symbol")
            symbol = str(symbol_raw) if symbol_raw is not None else ""
            if not symbol:
                continue

            close_ts_val = row.get("ts")
            if pd.isna(close_ts_val):
                continue

            close_ts = pd.Timestamp(close_ts_val)
            if close_ts.tz is None:
                close_ts = close_ts.tz_localize("UTC")
            else:
                close_ts = close_ts.tz_convert("UTC")

            key = (symbol, close_ts.isoformat())
            if reviewed_keys.get(key):
                continue

            ready_at = close_ts + wait_delta
            wait_seconds = int(max(0, (ready_at - now_utc).total_seconds()))
            state = "waiting" if ready_at > now_utc else "ready"

            item: dict[str, object] = {
                "symbol": symbol,
                "side": row.get("side"),
                "pnl": float(row.get("pnl") or 0.0),
                "closed_ts": close_ts.isoformat(),
                "ready_at": ready_at.isoformat(),
                "state": state,
                "wait_seconds": wait_seconds,
            }

            order_id = row.get("order_id")
            if order_id is not None and not pd.isna(order_id):
                item["order_id"] = str(order_id)

            pending.append(item)

        return pending

    def review_losing_trades(self, since_minutes: int = 600) -> None:
        """Review recent losing trades and store AI-generated feedback."""
        try:
            trades_df = self.store.load_trades()
            if trades_df is None or getattr(trades_df, "empty", True):
                return

            trades_df["ts"] = pd.to_datetime(trades_df["ts"], errors="coerce", utc=True)

            now_utc = pd.Timestamp.now(tz="UTC")
            since_ts = now_utc - pd.Timedelta(minutes=int(since_minutes))
            review_ready_cutoff = now_utc - pd.Timedelta(hours=48)

            closed_recent = trades_df[
                (trades_df["status"] == "closed")
                & (trades_df["pnl"].notna())
                & (trades_df["pnl"] <= 0)
                & (trades_df["ts"] >= since_ts)
                & (trades_df["ts"] <= review_ready_cutoff)
            ].copy()

            if getattr(closed_recent, "empty", True):
                return

            recent_reviews = self._load_recent_review_keys()

            ai_provider = self._ensure_ai_provider()

            for _, row in closed_recent.sort_values("ts").iterrows():
                try:
                    contract_symbol = str(row.get("symbol"))
                    side = row.get("side")
                    close_ts_raw = row.get("ts")
                    close_ts = pd.Timestamp(close_ts_raw)
                    if close_ts.tz is None:
                        close_ts = close_ts.tz_localize("UTC")
                    else:
                        close_ts = close_ts.tz_convert("UTC")
                    pnl = float(row.get("pnl") or 0.0)

                    closed_key = (contract_symbol, close_ts.isoformat())
                    if recent_reviews.get(closed_key):
                        continue

                    open_ts, entry_price = self._resolve_open_context(
                        trades_df, contract_symbol, close_ts
                    )

                    spot_symbol = contract_to_spot_symbol(contract_symbol)
                    open_ts_utc = pd.Timestamp(open_ts)
                    if open_ts_utc.tz is None:
                        open_ts_utc = open_ts_utc.tz_localize("UTC")
                    else:
                        open_ts_utc = open_ts_utc.tz_convert("UTC")
                    since_ms = int(open_ts_utc.timestamp() * 1000)
                    until_ms = int(close_ts.timestamp() * 1000)
                    csv_1m = ohlcv_csv_between(spot_symbol, "1m", since_ms, until_ms)

                    post_close_until = close_ts + pd.Timedelta(hours=1)
                    post_close_until_ms = int(post_close_until.timestamp() * 1000)
                    csv_after_close = ohlcv_csv_between(
                        spot_symbol, "1m", until_ms, post_close_until_ms
                    )

                    is_loss = pnl < 0
                    entry_notes = (
                        self._collect_entry_notes(
                            contract_symbol,
                            open_ts=open_ts_utc,
                            close_ts=close_ts,
                        )
                        if is_loss
                        else ""
                    )

                    prompt = self._build_review_prompt(
                        contract_symbol=contract_symbol,
                        spot_symbol=spot_symbol,
                        side=side,
                        pnl=pnl,
                        open_ts=open_ts_utc,
                        close_ts=close_ts,
                        entry_price=entry_price,
                        csv_1m=csv_1m,
                        csv_after_close=csv_after_close,
                        is_loss=is_loss,
                        entry_notes=entry_notes,
                    )

                    try:
                        review_text = ai_provider.decide(prompt)
                    except Exception as exc:
                        LOGGER.error("Trade review LLM failed: %s", exc)
                        review_text = None

                    self._record_review_entry(
                        contract_symbol=contract_symbol,
                        review_text=review_text,
                        pnl=pnl,
                        side=side,
                        open_ts=open_ts_utc,
                        close_ts=close_ts,
                        spot_symbol=spot_symbol,
                    )
                except Exception as loop_exc:
                    LOGGER.error("Review loop failed: %s", loop_exc)
        except Exception as exc:
            LOGGER.error("review_losing_trades failed: %s", exc)
