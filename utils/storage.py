from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import pandas as pd
import sqlalchemy as sa


@dataclass
class StorageConfig:
    # DB 전용으로 단순화
    mysql_url: Optional[str] = None  # e.g. mysql+pymysql://user:pwd@host:3306/db
    sqlite_path: Optional[str] = None  # e.g. data/trading.sqlite

    def resolve(self) -> Tuple[Optional[str], bool]:
        """Return (sqlalchemy_url, is_sqlite)."""
        if self.mysql_url:
            return self.mysql_url, False

        # 기본 sqlite 파일 경로 결정
        base_dir = Path(
            os.getenv("APP_BASE_DIR") or Path(__file__).resolve().parents[1]
        )
        default_path = base_dir / "data" / "trading.sqlite"
        target = (
            Path(self.sqlite_path).expanduser() if self.sqlite_path else default_path
        )

        if target.is_dir():
            target = target / "trading.sqlite"

        target.parent.mkdir(parents=True, exist_ok=True)
        return f"sqlite:///{target.resolve().as_posix()}", True


class TradeStore:
    def __init__(self, config: StorageConfig):
        self.config = config
        self._engine = None
        self._db_url, self._is_sqlite = config.resolve()
        if self._db_url:
            try:
                kwargs: Dict[str, Any] = {}
                if self._is_sqlite:
                    kwargs["connect_args"] = {"check_same_thread": False}
                else:
                    kwargs["pool_pre_ping"] = True
                self._engine = sa.create_engine(self._db_url, **kwargs)
            except Exception as e:
                print(f"Warning: failed to init database engine: {e}")

    def record_trade(self, trade: Dict[str, Any]) -> None:
        # DB 기록 전용
        if self._engine is None:
            print("No DB engine configured; trade not persisted")
            return
        if trade.get("ts") is None:
            trade = dict(trade)
            trade["ts"] = datetime.utcnow()
        if trade.get("order_id") is not None:
            trade["order_id"] = str(trade["order_id"])
        try:
            pd.DataFrame([trade]).to_sql(
                "trades",
                self._engine,
                if_exists="append",
                index=False,
                dtype={
                    "ts": sa.DateTime,
                    "symbol": sa.String(64),
                    "side": sa.String(8),
                    "type": sa.String(8),
                    "price": sa.Float,
                    "quantity": sa.Float,
                    "tp": sa.Float,
                    "sl": sa.Float,
                    "leverage": sa.Float,
                    "status": sa.String(16),
                    "order_id": sa.String(128),
                    "pnl": sa.Float,
                },
            )
        except Exception as e:
            print(f"Error writing MySQL: {e}")

    def load_trades(self) -> pd.DataFrame:
        # DB에서만 읽기
        if self._engine is None:
            return pd.DataFrame(
                columns=[
                    "ts",
                    "symbol",
                    "side",
                    "type",
                    "price",
                    "quantity",
                    "tp",
                    "sl",
                    "leverage",
                    "status",
                    "order_id",
                    "pnl",
                ]
            )
        try:
            return pd.read_sql_table("trades", self._engine)
        except Exception:
            return pd.DataFrame(
                columns=[
                    "ts",
                    "symbol",
                    "side",
                    "type",
                    "price",
                    "quantity",
                    "tp",
                    "sl",
                    "leverage",
                    "status",
                    "order_id",
                    "pnl",
                ]
            )

    def compute_stats(self) -> Dict[str, Any]:
        df = self.load_trades()
        if df.empty:
            return {
                "trades": 0,
                "realized_pnl": 0.0,
                "win_rate": 0.0,
                "avg_pnl": 0.0,
            }
        realized_df = df[df["pnl"].notna()]
        trades = len(realized_df)
        realized_pnl = float(realized_df["pnl"].sum()) if trades > 0 else 0.0
        wins = int((realized_df["pnl"] > 0).sum()) if trades > 0 else 0
        win_rate = float(wins / trades) if trades > 0 else 0.0
        avg_pnl = float(realized_df["pnl"].mean()) if trades > 0 else 0.0
        return {
            "trades": trades,
            "realized_pnl": realized_pnl,
            "win_rate": win_rate,
            "avg_pnl": avg_pnl,
        }

    def compute_stats_range(
        self,
        *,
        since_ts: Optional[datetime] = None,
        until_ts: Optional[datetime] = None,
        symbol: Optional[str] = None,
        group: Optional[str] = None,
    ) -> Dict[str, Any]:
        """기간/심볼 필터 기반 실현 손익 통계. DB가 없으면 기본값 반환.

        Args:
            since_ts: 포함 하한(UTC)
            until_ts: 제외 상한(UTC)
            symbol: 심볼 필터
            group: 'day' | 'week' | 'month' 그룹 집계 시리즈
        """
        df = self.load_trades()
        if df is None or getattr(df, "empty", True):
            return {
                "range": {"since": since_ts, "until": until_ts},
                "summary": {
                    "trades": 0,
                    "realized_pnl": 0.0,
                    "wins": 0,
                    "win_rate": 0.0,
                    "avg_pnl": 0.0,
                },
                "by_symbol": [],
                "series": [],
            }

        try:
            if not pd.api.types.is_datetime64_any_dtype(df["ts"]):
                df["ts"] = pd.to_datetime(df["ts"], errors="coerce", utc=True)
        except Exception:
            df["ts"] = pd.to_datetime(df["ts"], errors="coerce", utc=True)

        # 필터: 실현 손익이 있는 행만
        df = df[df["pnl"].notna()].copy()
        if since_ts is not None:
            df = df[df["ts"] >= pd.Timestamp(since_ts, tz="UTC")]
        if until_ts is not None:
            df = df[df["ts"] < pd.Timestamp(until_ts, tz="UTC")]
        if symbol:
            df = df[df["symbol"].astype(str) == str(symbol)]

        if getattr(df, "empty", True):
            base = {
                "trades": 0,
                "realized_pnl": 0.0,
                "wins": 0,
                "win_rate": 0.0,
                "avg_pnl": 0.0,
            }
            return {
                "range": {"since": since_ts, "until": until_ts},
                "summary": base,
                "by_symbol": [],
                "series": [],
            }

        trades = int(len(df))
        realized_pnl = float(df["pnl"].sum())
        wins = int((df["pnl"] > 0).sum())
        win_rate = float(wins / trades) if trades > 0 else 0.0
        avg_pnl = float(df["pnl"].mean()) if trades > 0 else 0.0

        # 심볼별 집계
        by_symbol = []
        try:
            g = df.groupby("symbol", dropna=False)
            for k, sub in g:
                by_symbol.append(
                    {
                        "symbol": k,
                        "trades": int(len(sub)),
                        "realized_pnl": float(sub["pnl"].sum()),
                    }
                )
        except Exception:
            pass

        # 시계열 집계
        series = []
        try:
            if group in ("day", "week", "month"):
                if group == "day":
                    idx = df["ts"].dt.floor("D")
                elif group == "week":
                    # 주의 시작으로 정규화 (월요일)
                    idx = (
                        df["ts"] - pd.to_timedelta(df["ts"].dt.weekday, unit="D")
                    ).dt.floor("D")
                else:
                    idx = df["ts"].dt.to_period("M").dt.to_timestamp()
                gf = df.groupby(idx)
                for t, sub in gf:
                    series.append(
                        {
                            "t": pd.Timestamp(t).tz_convert("UTC").isoformat(),
                            "realized_pnl": float(sub["pnl"].sum()),
                            "trades": int(len(sub)),
                        }
                    )
        except Exception:
            pass

        return {
            "range": {"since": since_ts, "until": until_ts},
            "summary": {
                "trades": trades,
                "realized_pnl": realized_pnl,
                "wins": wins,
                "win_rate": win_rate,
                "avg_pnl": avg_pnl,
            },
            "by_symbol": by_symbol,
            "series": series,
        }

    # -------------------------------
    # Journal (DB only)
    # -------------------------------
    def record_journal(self, entry: Dict[str, Any]) -> None:
        """Persist a journal entry to MySQL if configured.

        Expected fields:
          - ts: datetime (optional; defaults to now)
          - symbol: str (e.g., BTCUSDT)
          - entry_type: str in {"thought", "decision", "action", "review"}
          - content: str (free-form text)
          - reason: Optional[str]
          - meta: Optional[dict]
          - ref_order_id: Optional[str]
        """
        if self._engine is None:
            return
        try:
            data = dict(entry)
            if data.get("ts") is None:
                # Lazy import to avoid global dependency
                from datetime import datetime

                data["ts"] = datetime.utcnow()
            # Normalize
            data.setdefault("symbol", None)
            data.setdefault("entry_type", None)
            data.setdefault("content", None)
            data.setdefault("reason", None)
            data.setdefault("meta", None)
            data.setdefault("ref_order_id", None)

            pd.DataFrame([data]).to_sql(
                "journals",
                self._engine,
                if_exists="append",
                index=False,
                dtype={
                    "ts": sa.DateTime,
                    "symbol": sa.String(64),
                    "entry_type": sa.String(16),
                    "content": sa.Text,
                    "reason": sa.Text,
                    "meta": sa.JSON,
                    "ref_order_id": sa.String(128),
                },
            )
        except Exception as e:
            print(f"Error writing journals: {e}")

    def fetch_journals(
        self,
        symbol: Optional[str] = None,
        types: Optional[list] = None,
        today_only: bool = False,
        since_ts: Optional[datetime] = None,
        until_ts: Optional[datetime] = None,
        limit: int = 20,
        ascending: bool = True,
    ) -> pd.DataFrame:
        """Fetch journal entries with filters. Returns a DataFrame.

        Note: This reads from MySQL only. If engine is not set, returns empty DataFrame.
        """
        if self._engine is None:
            return pd.DataFrame(
                columns=[
                    "ts",
                    "symbol",
                    "entry_type",
                    "content",
                    "reason",
                    "meta",
                    "ref_order_id",
                ]
            )
        try:
            # Build SQL dynamically using SQLAlchemy text for safety
            from sqlalchemy import text

            clauses = []
            params: Dict[str, Any] = {}
            if symbol:
                clauses.append("symbol = :symbol")
                params["symbol"] = symbol
            if types:
                # create IN clause
                in_params = {f"t{i}": t for i, t in enumerate(types)}
                placeholders = ",".join([f":{k}" for k in in_params.keys()])
                clauses.append(f"entry_type IN ({placeholders})")
                params.update(in_params)
            if today_only:
                clauses.append("DATE(ts) = CURRENT_DATE")
            if since_ts is not None:
                clauses.append("ts >= :since_ts")
                params["since_ts"] = since_ts
            if until_ts is not None:
                clauses.append("ts < :until_ts")
                params["until_ts"] = until_ts

            where_sql = (" WHERE " + " AND ".join(clauses)) if clauses else ""
            order_sql = " ORDER BY ts ASC" if ascending else " ORDER BY ts DESC"
            limit_sql = " LIMIT :limit"
            params["limit"] = int(limit)

            sql = text(
                f"SELECT ts, symbol, entry_type, content, reason, meta, ref_order_id FROM journals{where_sql}{order_sql}{limit_sql}"
            )
            with self._engine.connect() as conn:
                rows = conn.execute(sql, params).mappings().all()
            # Convert to DataFrame
            return pd.DataFrame(rows)
        except Exception as e:
            print(f"Error reading journals: {e}")
            return pd.DataFrame(
                columns=[
                    "ts",
                    "symbol",
                    "entry_type",
                    "content",
                    "reason",
                    "meta",
                    "ref_order_id",
                ]
            )
