# pylint: disable=broad-except
# ruff: noqa: E722, BLE001
from __future__ import annotations

import datetime as dt
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, Union, List

import pandas as pd
import sqlalchemy as sa
from sqlalchemy.engine import make_url

LOGGER = logging.getLogger(__name__)


def ensure_mysql_database(db_url: str) -> None:
    """Ensure the MySQL database exists, creating it when absent.

    Retries connection attempts based on environment configuration.
    """

    if not db_url:
        return

    try:
        url = make_url(db_url)
    except Exception as exc:  # pragma: no cover - defensive, logs for runtime issues
        LOGGER.warning("잘못된 MYSQL_URL: %s", exc)
        raise

    if not url.drivername.startswith("mysql"):
        return

    database = (url.database or "").strip()
    if not database:
        raise ValueError("MYSQL_URL에 데이터베이스 이름이 포함되어야 합니다")

    safe_database = database.replace("`", "")

    attempts = int(os.getenv("MYSQL_BOOTSTRAP_ATTEMPTS", "20"))
    delay = float(os.getenv("MYSQL_BOOTSTRAP_DELAY", "3"))

    connect_url = url.set(database=None)

    last_exc: Optional[Exception] = None

    for attempt in range(1, attempts + 1):
        engine = None
        try:
            engine = sa.create_engine(connect_url)
            with engine.connect() as conn:
                conn.execution_options(isolation_level="AUTOCOMMIT").execute(
                    sa.text(
                        f"CREATE DATABASE IF NOT EXISTS `{safe_database}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
                    )
                )
            LOGGER.info("MySQL 데이터베이스 '%s' 확인 완료", safe_database)
            return
        except Exception as exc:  # pragma: no cover - runtime dependent
            last_exc = exc
            LOGGER.warning(
                "[%s/%s] MySQL 데이터베이스 준비 대기 중: %s",
                attempt,
                attempts,
                exc,
            )
            time.sleep(delay)
        finally:
            if engine is not None:
                engine.dispose()

    if last_exc:
        raise last_exc


@dataclass
class StorageConfig:
    # DB 전용으로 단순화
    mysql_url: Optional[str] = None  # e.g. mysql+pymysql://user:pwd@host:3306/db
    sqlite_path: Optional[str] = None  # e.g. data/trading.sqlite

    def resolve(self) -> Tuple[Optional[str], bool]:
        """Return (sqlalchemy_url, is_sqlite)."""
        force_sqlite = str(os.getenv("FORCE_SQLITE", "0")).lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        if self.mysql_url and not force_sqlite:
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
                    try:
                        ensure_mysql_database(self._db_url)
                        kwargs["pool_pre_ping"] = True
                    except Exception as exc:
                        print(
                            f"Warning: failed to prepare MySQL database ({exc}); falling back to SQLite"
                        )
                        self._db_url, self._is_sqlite = StorageConfig(
                            sqlite_path=self.config.sqlite_path
                        ).resolve()
                        kwargs = {"connect_args": {"check_same_thread": False}}

                self._engine = sa.create_engine(self._db_url, **kwargs)
                # 연결 확인 및 실패 시 SQLite로 폴백
                if not self._is_sqlite:
                    try:
                        with self._engine.connect() as conn:
                            conn.execute(sa.text("SELECT 1"))
                    except Exception as e:
                        print(
                            f"Warning: failed to connect to database ({e}); falling back to SQLite"
                        )
                        # SQLite로 강제 전환
                        self._db_url, self._is_sqlite = StorageConfig(
                            sqlite_path=self.config.sqlite_path
                        ).resolve()
                        kwargs = {"connect_args": {"check_same_thread": False}}
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
            trade["ts"] = dt.datetime.utcnow()
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
            print(f"Error writing database: {e}")

    def update_open_trades_tp_sl(
        self,
        symbol: str,
        *,
        side: Optional[str] = None,
        tp: Optional[float] = None,
        sl: Optional[float] = None,
    ) -> int:
        """Update TP/SL for currently opened trades in storage.

        Returns the number of rows affected. Only updates when TP/SL values
        are provided and the storage engine is configured.
        """

        if self._engine is None:
            return 0

        symbol_key = str(symbol or "").strip()
        if not symbol_key:
            return 0

        set_clauses: List[str] = []
        params: Dict[str, Any] = {"symbol": symbol_key, "status": "opened"}

        if tp is not None:
            try:
                params["tp"] = float(tp)
                set_clauses.append("tp = :tp")
            except Exception:
                pass

        if sl is not None:
            try:
                params["sl"] = float(sl)
                set_clauses.append("sl = :sl")
            except Exception:
                pass

        if not set_clauses:
            return 0

        where_clauses = ["symbol = :symbol", "status = :status"]

        side_values: List[str] = []
        if side is not None:
            try:
                norm = str(side).strip().lower()
            except Exception:
                norm = ""
            if norm in {"long", "buy"}:
                side_values = ["long", "buy"]
            elif norm in {"short", "sell"}:
                side_values = ["short", "sell"]

        if side_values:
            placeholders = ", ".join(
                f":side{i}" for i in range(len(side_values))
            )
            where_clauses.append(f"LOWER(side) IN ({placeholders})")
            for idx, value in enumerate(side_values):
                params[f"side{idx}"] = value

        query = sa.text(
            f"UPDATE trades SET {', '.join(set_clauses)} WHERE {' AND '.join(where_clauses)}"
        )

        try:
            with self._engine.begin() as conn:
                result = conn.execute(query, params)
                return int(getattr(result, "rowcount", 0) or 0)
        except Exception as exc:  # pragma: no cover - runtime logging only
            LOGGER.error("Failed to update trades TP/SL: %s", exc)
            return 0

    @staticmethod
    def _drop_duplicate_orders(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or getattr(df, "empty", True):
            return df
        if "order_id" not in df.columns or "symbol" not in df.columns:
            return df
        try:
            df_sorted = df.sort_values("ts") if "ts" in df.columns else df.copy()
        except Exception:
            df_sorted = df.copy()
        mask = df_sorted["order_id"].notna()
        if not mask.any():
            return df_sorted
        try:
            duplicates = df_sorted.loc[mask].duplicated(
                subset=["symbol", "order_id"], keep="last"
            )
        except Exception:
            return df_sorted
        if not duplicates.any():
            return df_sorted
        drop_index = df_sorted.loc[mask].loc[duplicates].index
        return df_sorted.drop(index=drop_index)

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
        realized_df = self._drop_duplicate_orders(realized_df)
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
        since_ts: Optional[dt.datetime] = None,
        until_ts: Optional[dt.datetime] = None,
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
            # 항상 UTC 타임존을 가진 시계열로 강제 변환 (naive/aware 모두 커버)
            df["ts"] = pd.to_datetime(df["ts"], errors="coerce", utc=True)
        except Exception:
            df["ts"] = pd.to_datetime(df["ts"], errors="coerce", utc=True)

        # 필터: 실현 손익이 있는 행만
        df = df[df["pnl"].notna()].copy()
        df = self._drop_duplicate_orders(df)

        # 입력 경계값(since/until)을 안전하게 UTC로 정규화
        def _ensure_utc(ts_val):
            try:
                t = pd.Timestamp(ts_val)
                if t.tz is None:
                    return t.tz_localize("UTC")
                return t.tz_convert("UTC")
            except Exception:
                return None

        if since_ts is not None:
            _since = _ensure_utc(since_ts)
            if _since is not None:
                df = df[df["ts"] >= _since]
        if until_ts is not None:
            _until = _ensure_utc(until_ts)
            if _until is not None:
                df = df[df["ts"] < _until]
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
                    # month는 tz가 사라질 수 있으므로 이후 UTC 로컬라이즈 처리
                    idx = df["ts"].dt.to_period("M").dt.to_timestamp()

                def _to_utc_iso_any(t):
                    try:
                        tt = pd.Timestamp(t)
                        if tt.tz is None:
                            tt = tt.tz_localize("UTC")
                        else:
                            tt = tt.tz_convert("UTC")
                        return tt.isoformat()
                    except Exception:
                        try:
                            if isinstance(t, dt.datetime):
                                if t.tzinfo is None:
                                    t = t.replace(tzinfo=dt.timezone.utc)
                                else:
                                    t = t.astimezone(dt.timezone.utc)
                                return t.isoformat()
                        except Exception:
                            return str(t)

                gf = df.groupby(idx)
                for t, sub in gf:
                    series.append(
                        {
                            "t": _to_utc_iso_any(t),
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
                data["ts"] = dt.datetime.utcnow()
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
        since_ts: Optional[dt.datetime] = None,
        until_ts: Optional[dt.datetime] = None,
        limit: int = 20,
        ascending: bool = True,
        *,
        offset: int = 0,
        return_total: bool = False,
        limit_choices: Optional[Tuple[int, ...]] = None,
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, int]]:
        """Fetch journal entries with filters.

        Returns:
            - DataFrame: 기본 반환값
            - (DataFrame, total_count): ``return_total=True`` 일 때

        Note: This reads from MySQL only. If engine is not set, returns empty DataFrame.
        """
        empty_df = pd.DataFrame(
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

        if self._engine is None:
            return (empty_df, 0) if return_total else empty_df
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
                # SQLite와 기타 DB의 오늘 날짜 표현을 각각 지원
                if getattr(self, "_is_sqlite", False):
                    clauses.append("DATE(ts) = DATE('now')")
                else:
                    clauses.append("DATE(ts) = CURRENT_DATE")
            if since_ts is not None:
                # Pandas Timestamp 등 datetime 유사 타입을 안전하게 Python datetime으로 변환
                try:
                    since_ts = pd.Timestamp(since_ts).to_pydatetime()
                except Exception:
                    pass
                clauses.append("ts >= :since_ts")
                params["since_ts"] = since_ts
            if until_ts is not None:
                try:
                    until_ts = pd.Timestamp(until_ts).to_pydatetime()
                except Exception:
                    pass
                clauses.append("ts < :until_ts")
                params["until_ts"] = until_ts

            where_sql = (" WHERE " + " AND ".join(clauses)) if clauses else ""
            order_sql = " ORDER BY ts ASC" if ascending else " ORDER BY ts DESC"
            limit_options: Optional[Tuple[int, ...]] = None
            if limit_choices:
                try:
                    limit_options = tuple(int(x) for x in limit_choices if int(x) > 0)
                    if not limit_options:
                        limit_options = None
                except Exception:
                    limit_options = None

            try:
                limit_value = int(limit)
            except Exception:
                limit_value = 0

            if limit_options:
                if limit_value not in limit_options:
                    limit_value = limit_options[0]
            else:
                limit_value = max(1, min(limit_value if limit_value > 0 else 1, 200))

            try:
                offset_value = int(offset)
            except Exception:
                offset_value = 0
            if offset_value < 0:
                offset_value = 0

            limit_sql = f" LIMIT {limit_value}"
            offset_sql = f" OFFSET {offset_value}" if offset_value else ""

            sql = text(
                f"SELECT ts, symbol, entry_type, content, reason, meta, ref_order_id FROM journals{where_sql}{order_sql}{limit_sql}{offset_sql}"
            )

            total_count = 0
            with self._engine.connect() as conn:
                if return_total:
                    count_sql = text(f"SELECT COUNT(*) AS cnt FROM journals{where_sql}")
                    total_raw = conn.execute(count_sql, params).scalar()
                    try:
                        total_count = int(total_raw or 0)
                    except Exception:
                        total_count = 0

                rows = conn.execute(sql, params).mappings().all()

            df = pd.DataFrame(rows)
            return (df, total_count) if return_total else df
        except Exception as e:
            print(f"Error reading journals: {e}")
            return (empty_df, 0) if return_total else empty_df
