from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

import pandas as pd
import sqlalchemy as sa


@dataclass
class StorageConfig:
    # XLSX
    xlsx_path: str = "trades.xlsx"
    # MySQL
    mysql_url: Optional[str] = None  # e.g. mysql+pymysql://user:pwd@host:3306/db


class TradeStore:
    def __init__(self, config: StorageConfig):
        self.config = config
        self._engine = None
        if config.mysql_url:
            try:
                self._engine = sa.create_engine(config.mysql_url, pool_pre_ping=True)
            except Exception as e:
                print(f"Warning: failed to init MySQL engine: {e}")

    def _ensure_xlsx(self) -> None:
        if not os.path.exists(self.config.xlsx_path):
            df = pd.DataFrame(
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
            df.to_excel(self.config.xlsx_path, index=False)

    def record_trade(self, trade: Dict[str, Any]) -> None:
        # XLSX 기록
        try:
            self._ensure_xlsx()
            df = pd.read_excel(self.config.xlsx_path)
            df = pd.concat([df, pd.DataFrame([trade])], ignore_index=True)
            df.to_excel(self.config.xlsx_path, index=False)
        except Exception as e:
            print(f"Error writing xlsx: {e}")

        # MySQL 기록
        if self._engine is not None:
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
        try:
            self._ensure_xlsx()
            return pd.read_excel(self.config.xlsx_path)
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
        since_ts: Optional["datetime"] = None,
        until_ts: Optional["datetime"] = None,
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
