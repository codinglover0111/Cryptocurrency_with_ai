# pylint: disable=broad-except
# ruff: noqa: E722, BLE001
from __future__ import annotations

import os
from dataclasses import dataclass
import datetime as dt
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, Union

import pandas as pd
import sqlalchemy as sa


SCHEMA_METADATA = sa.MetaData()


TRADES_TABLE = sa.Table(
    "trades",
    SCHEMA_METADATA,
    sa.Column("ts", sa.DateTime, nullable=True),
    sa.Column("symbol", sa.String(64), nullable=True),
    sa.Column("side", sa.String(8), nullable=True),
    sa.Column("type", sa.String(8), nullable=True),
    sa.Column("price", sa.Float, nullable=True),
    sa.Column("quantity", sa.Float, nullable=True),
    sa.Column("tp", sa.Float, nullable=True),
    sa.Column("sl", sa.Float, nullable=True),
    sa.Column("leverage", sa.Float, nullable=True),
    sa.Column("status", sa.String(16), nullable=True),
    sa.Column("order_id", sa.String(128), nullable=True),
    sa.Column("pnl", sa.Float, nullable=True),
    mysql_engine="InnoDB",
    mysql_charset="utf8mb4",
)


JOURNALS_TABLE = sa.Table(
    "journals",
    SCHEMA_METADATA,
    sa.Column("ts", sa.DateTime, nullable=True),
    sa.Column("symbol", sa.String(64), nullable=True),
    sa.Column("entry_type", sa.String(16), nullable=True),
    sa.Column("content", sa.Text, nullable=True),
    sa.Column("reason", sa.Text, nullable=True),
    sa.Column("meta", sa.JSON, nullable=True),
    sa.Column("ref_order_id", sa.String(128), nullable=True),
    mysql_engine="InnoDB",
    mysql_charset="utf8mb4",
)


SCHEDULER_STATE_TABLE = sa.Table(
    "scheduler_state",
    SCHEMA_METADATA,
    sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
    sa.Column("key", sa.String(64), unique=True, nullable=False),
    sa.Column("value", sa.Text, nullable=True),
    sa.Column("updated_at", sa.DateTime, nullable=True),
    mysql_engine="InnoDB",
    mysql_charset="utf8mb4",
)


SHARED_ANALYSIS_TABLE = sa.Table(
    "shared_analysis",
    SCHEMA_METADATA,
    sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
    sa.Column("symbol", sa.String(64), nullable=True),
    sa.Column("analysis_type", sa.String(32), nullable=True),
    sa.Column("content", sa.Text, nullable=True),
    sa.Column("created_at", sa.DateTime, nullable=True),
    mysql_engine="InnoDB",
    mysql_charset="utf8mb4",
)


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
                    kwargs["pool_pre_ping"] = True
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

        if self._engine is not None:
            self._ensure_schema()

    def _ensure_schema(self) -> None:
        try:
            SCHEMA_METADATA.create_all(
                self._engine,
                tables=[TRADES_TABLE, JOURNALS_TABLE, SCHEDULER_STATE_TABLE, SHARED_ANALYSIS_TABLE],
                checkfirst=True,
            )
        except Exception as e:
            print(f"Warning: failed to ensure database schema: {e}")

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

    # -------------------------------
    # Scheduler State (DB only)
    # -------------------------------
    def set_scheduler_state(self, key: str, value: str) -> bool:
        """스케줄러 상태를 저장합니다 (upsert)."""
        if self._engine is None:
            return False
        try:
            now = dt.datetime.utcnow()
            with self._engine.connect() as conn:
                # 기존 값 확인
                check_sql = sa.text(
                    "SELECT id FROM scheduler_state WHERE `key` = :key"
                )
                result = conn.execute(check_sql, {"key": key}).fetchone()
                
                if result:
                    # UPDATE
                    update_sql = sa.text(
                        "UPDATE scheduler_state SET value = :value, updated_at = :updated_at WHERE `key` = :key"
                    )
                    conn.execute(update_sql, {"key": key, "value": value, "updated_at": now})
                else:
                    # INSERT
                    insert_sql = sa.text(
                        "INSERT INTO scheduler_state (`key`, value, updated_at) VALUES (:key, :value, :updated_at)"
                    )
                    conn.execute(insert_sql, {"key": key, "value": value, "updated_at": now})
                conn.commit()
            return True
        except Exception as e:
            print(f"Error saving scheduler state: {e}")
            return False

    def get_scheduler_state(self, key: str) -> Optional[str]:
        """스케줄러 상태를 가져옵니다."""
        if self._engine is None:
            return None
        try:
            with self._engine.connect() as conn:
                sql = sa.text(
                    "SELECT value FROM scheduler_state WHERE `key` = :key"
                )
                result = conn.execute(sql, {"key": key}).fetchone()
                if result:
                    return result[0]
            return None
        except Exception as e:
            print(f"Error reading scheduler state: {e}")
            return None

    def get_all_scheduler_states(self) -> Dict[str, Any]:
        """모든 스케줄러 상태를 딕셔너리로 반환합니다."""
        if self._engine is None:
            return {}
        try:
            with self._engine.connect() as conn:
                sql = sa.text(
                    "SELECT `key`, value, updated_at FROM scheduler_state"
                )
                rows = conn.execute(sql).fetchall()
                result = {}
                for row in rows:
                    result[row[0]] = {
                        "value": row[1],
                        "updated_at": row[2].isoformat() if row[2] else None,
                    }
                return result
        except Exception as e:
            print(f"Error reading scheduler states: {e}")
            return {}

    # -------------------------------
    # Shared Analysis (BTC 분석 결과 공유)
    # -------------------------------
    def save_shared_analysis(
        self, symbol: str, analysis_type: str, content: str
    ) -> bool:
        """공유 분석 결과를 저장합니다."""
        if self._engine is None:
            return False
        try:
            now = dt.datetime.utcnow()
            with self._engine.connect() as conn:
                # 기존 동일 타입 분석 삭제 (최신 것만 유지)
                delete_sql = sa.text(
                    "DELETE FROM shared_analysis WHERE symbol = :symbol AND analysis_type = :analysis_type"
                )
                conn.execute(delete_sql, {"symbol": symbol, "analysis_type": analysis_type})
                
                # 새 분석 삽입
                insert_sql = sa.text(
                    "INSERT INTO shared_analysis (symbol, analysis_type, content, created_at) "
                    "VALUES (:symbol, :analysis_type, :content, :created_at)"
                )
                conn.execute(insert_sql, {
                    "symbol": symbol,
                    "analysis_type": analysis_type,
                    "content": content,
                    "created_at": now,
                })
                conn.commit()
            return True
        except Exception as e:
            print(f"Error saving shared analysis: {e}")
            return False

    def get_shared_analysis(
        self, symbol: str, analysis_type: Optional[str] = None, max_age_minutes: int = 60
    ) -> Optional[Dict[str, Any]]:
        """공유 분석 결과를 가져옵니다.
        
        Args:
            symbol: 심볼 (예: BTCUSDT:USDT)
            analysis_type: 분석 타입 (예: trend, market_sentiment)
            max_age_minutes: 최대 유효 시간 (분)
        
        Returns:
            분석 결과 딕셔너리 또는 None
        """
        if self._engine is None:
            return None
        try:
            cutoff = dt.datetime.utcnow() - dt.timedelta(minutes=max_age_minutes)
            with self._engine.connect() as conn:
                if analysis_type:
                    sql = sa.text(
                        "SELECT symbol, analysis_type, content, created_at FROM shared_analysis "
                        "WHERE symbol = :symbol AND analysis_type = :analysis_type "
                        "AND created_at >= :cutoff ORDER BY created_at DESC LIMIT 1"
                    )
                    result = conn.execute(sql, {
                        "symbol": symbol,
                        "analysis_type": analysis_type,
                        "cutoff": cutoff,
                    }).fetchone()
                else:
                    sql = sa.text(
                        "SELECT symbol, analysis_type, content, created_at FROM shared_analysis "
                        "WHERE symbol = :symbol AND created_at >= :cutoff "
                        "ORDER BY created_at DESC LIMIT 1"
                    )
                    result = conn.execute(sql, {
                        "symbol": symbol,
                        "cutoff": cutoff,
                    }).fetchone()
                
                if result:
                    return {
                        "symbol": result[0],
                        "analysis_type": result[1],
                        "content": result[2],
                        "created_at": result[3].isoformat() if result[3] else None,
                    }
                return None
        except Exception as e:
            print(f"Error reading shared analysis: {e}")
            return None

    def get_btc_analysis(self, max_age_minutes: int = 60) -> Optional[Dict[str, Any]]:
        """BTC 분석 결과를 가져옵니다 (다른 심볼 분석 시 컨텍스트로 사용)."""
        # BTCUSDT 또는 BTC로 시작하는 심볼 검색
        if self._engine is None:
            return None
        try:
            cutoff = dt.datetime.utcnow() - dt.timedelta(minutes=max_age_minutes)
            with self._engine.connect() as conn:
                sql = sa.text(
                    "SELECT symbol, analysis_type, content, created_at FROM shared_analysis "
                    "WHERE (symbol LIKE 'BTC%' OR symbol LIKE 'btc%') "
                    "AND created_at >= :cutoff ORDER BY created_at DESC LIMIT 1"
                )
                result = conn.execute(sql, {"cutoff": cutoff}).fetchone()
                
                if result:
                    return {
                        "symbol": result[0],
                        "analysis_type": result[1],
                        "content": result[2],
                        "created_at": result[3].isoformat() if result[3] else None,
                    }
                return None
        except Exception as e:
            print(f"Error reading BTC analysis: {e}")
            return None
