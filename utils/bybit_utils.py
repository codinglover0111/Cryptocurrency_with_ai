# pylint: disable=broad-except
# ruff: noqa: E722, BLE001
import os
import ccxt
from dotenv import load_dotenv

from typing import Optional, Dict, Any, Literal, List
from pydantic import BaseModel
import logging
from app.logging_config import setup_logging

setup_logging()
LOGGER = logging.getLogger(__name__)


class Open_Position(BaseModel):
    symbol: str
    side: Literal["buy", "sell"]
    quantity: float
    price: float
    type: Literal["limit", "market"]
    tp: Optional[float]
    sl: Optional[float]
    leverage: Optional[float] = None


dotenv_path = os.path.join(os.path.dirname(__file__), "..", ".env")
load_dotenv(dotenv_path)


class BybitUtils:
    def __init__(self, is_testnet=True):
        try:
            # 사용 환경 결정: demo | testnet | mainnet
            mode_env = (os.getenv("BYBIT_ENV") or "").strip().lower()
            mode = (
                mode_env
                if mode_env in {"demo", "testnet", "mainnet"}
                else ("testnet" if is_testnet else "mainnet")
            )

            # API 키와 시크릿 설정
            if mode == "demo":
                api_key = (
                    os.getenv("BYBIT_DEMO_API_KEY")
                    or os.getenv("DEMO_BYBIT_API_KEY")
                    or os.environ.get("BYBIT_API_KEY")
                )
                api_secret = (
                    os.getenv("BYBIT_DEMO_API_SECRET")
                    or os.getenv("DEMO_BYBIT_API_SECRET")
                    or os.environ.get("BYBIT_API_SECRET")
                )
            elif mode == "testnet":
                api_key = (
                    os.getenv("BYBIT_TESTNET_API_KEY")
                    or os.getenv("TESTNET_BYBIT_API_KEY")
                    or os.getenv("DEMO_BYBIT_API_KEY")
                    or os.environ.get("BYBIT_API_KEY")
                )
                api_secret = (
                    os.getenv("BYBIT_TESTNET_API_SECRET")
                    or os.getenv("TESTNET_BYBIT_API_SECRET")
                    or os.getenv("DEMO_BYBIT_API_SECRET")
                    or os.environ.get("BYBIT_API_SECRET")
                )
            else:
                api_key = os.environ["BYBIT_API_KEY"]
                api_secret = os.environ["BYBIT_API_SECRET"]
            # Bybit 객체 생성
            self.exchange = ccxt.bybit(
                {
                    "apiKey": api_key,
                    "secret": api_secret,
                    "enableRateLimit": True,  # 요청 제한 활성화
                    "options": {
                        "defaultType": "future",  # 선물 거래인 경우
                        # CCXT 권장: 서버-로컬 시간 차 자동 보정
                        "adjustForTimeDifference": True,
                        # Bybit V5 인증 헤더 X-BAPI-RECV-WINDOW (ms)
                        # 환경변수 BYBIT_RECV_WINDOW_MS 로 오버라이드 가능
                        "recvWindow": int(os.getenv("BYBIT_RECV_WINDOW_MS", "15000")),
                        # 일부 CCXT 버전 호환 (옵션 키 소문자 변형)
                        "recvwindow": int(os.getenv("BYBIT_RECV_WINDOW_MS", "15000")),
                    },
                }
            )
            self._markets_loaded = False
            self.positions: Optional[dict] = None
            # 기본 recv_window 헤더 보장
            try:
                recv_ms = int(os.getenv("BYBIT_RECV_WINDOW_MS", "15000"))
                headers = getattr(self.exchange, "headers", {}) or {}
                headers.update({"X-BAPI-RECV-WINDOW": str(recv_ms)})
                self.exchange.headers = headers
            except Exception:
                pass
            # 샌드박스/데모 모드 설정 (모드에 따라 분리)
            try:
                if mode == "demo":
                    # Demo Trading: api-demo.bybit.com 을 사용
                    if hasattr(self.exchange, "enable_demo_trading"):
                        self.exchange.enable_demo_trading(True)
                    if hasattr(self.exchange, "set_sandbox_mode"):
                        self.exchange.set_sandbox_mode(False)
                elif mode == "testnet":
                    # Testnet: api-testnet.bybit.com 을 사용
                    if hasattr(self.exchange, "set_sandbox_mode"):
                        self.exchange.set_sandbox_mode(True)
                    if hasattr(self.exchange, "enable_demo_trading"):
                        self.exchange.enable_demo_trading(False)
                else:
                    # Mainnet 기본값
                    if hasattr(self.exchange, "set_sandbox_mode"):
                        self.exchange.set_sandbox_mode(False)
                    if hasattr(self.exchange, "enable_demo_trading"):
                        self.exchange.enable_demo_trading(False)
            except Exception:
                pass

            # 서버-로컬 시간 차 동기화(초기 1회) 및 보수적 안전 마진 적용
            self._sync_time_with_bybit(mode)
        except Exception as e:
            print(f"Initialization error: {e}")

    def _get_recv_window_ms(self) -> int:
        try:
            return int(os.getenv("BYBIT_RECV_WINDOW_MS", "15000"))
        except Exception:
            return 15000

    def _default_params(self) -> Dict[str, Any]:
        rw = self._get_recv_window_ms()
        # Bybit는 recv_window(스네이크) 혹은 X-BAPI-RECV-WINDOW 헤더를 허용
        return {"recvWindow": rw, "recv_window": rw}

    def _merge_params(self, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        base = self._default_params()
        if params:
            base.update(params)
        return base

    def _sync_time_with_bybit(self, mode: str) -> None:
        try:
            # 서버 시간 조회 후 timeDifference를 직접 설정(보수적 안전 마진 적용)
            # 일부 환경에서 load_time_difference가 동작하지 않는 경우 대비
            server_time = None
            for _ in range(3):
                try:
                    server_time = self.exchange.fetch_time()
                    if server_time:
                        break
                except Exception:
                    LOGGER.error("Failed to fetch server time")

                # 폴백: ccxt의 자동 보정 시도
                try:
                    self.exchange.load_time_difference()
                except Exception:
                    LOGGER.error("Failed to load time difference")
            LOGGER.info("time synced with Bybit mode: %s", mode)
        except Exception:
            LOGGER.error("Failed to sync time with Bybit")

    def _ensure_markets(self) -> None:
        if self._markets_loaded:
            return
        try:
            self.exchange.load_markets()
            self._markets_loaded = True
        except Exception:
            LOGGER.error("Failed to ensure markets")

    def _symbol_to_market_id(self, symbol: Optional[str]) -> Optional[str]:
        if not symbol:
            return symbol
        try:
            self._ensure_markets()
            market = self.exchange.market(symbol)
            if market:
                return market.get("id") or symbol
        except Exception:
            pass
        return str(symbol).replace("/", "").replace(":", "")

    def _market_id_to_symbol(self, market_id: Optional[str]) -> Optional[str]:
        if not market_id:
            return market_id
        try:
            self._ensure_markets()
            market = self.exchange.markets_by_id.get(market_id)  # type: ignore[attr-defined]
            if market:
                return market.get("symbol") or market_id
        except Exception:
            pass
        return market_id

    def _resolve_category(self, symbol: Optional[str] = None) -> str:
        category: Optional[str] = None

        market: Optional[Dict[str, Any]] = None
        if symbol:
            try:
                self._ensure_markets()
                market = self.exchange.market(symbol)
            except Exception:
                market = None

        if market:
            if market.get("option"):
                category = "option"
            elif market.get("linear"):
                category = "linear"
            elif market.get("inverse"):
                category = "inverse"
            else:
                settle = str(market.get("settle") or "").upper()
                if settle in {"USDT", "USDC"}:
                    category = "linear"
                elif settle:
                    category = "inverse"

            if not category:
                type_hint = str(market.get("type") or "").lower()
                if type_hint in {"spot"}:
                    category = "spot"

        if not category:
            default_category = str(
                os.getenv("BYBIT_DEFAULT_CATEGORY")
                or self.exchange.options.get("defaultSubType")
                or self.exchange.options.get("defaultType")
                or ""
            ).lower()
            if default_category in {"linear", "inverse", "option", "spot"}:
                category = default_category

        if not category:
            category = "linear"

        return category

    @staticmethod
    def _float_or_none(raw: Any) -> Optional[float]:
        try:
            if raw is None:
                return None
            if isinstance(raw, str) and raw.strip() == "":
                return None
            return float(raw)
        except Exception:
            return None

    @staticmethod
    def _int_or_none(raw: Any) -> Optional[int]:
        try:
            if raw is None:
                return None
            if isinstance(raw, str) and raw.strip() == "":
                return None
            return int(float(raw))
        except Exception:
            return None

    def _normalize_closed_pnl_entry(
        self,
        entry: Dict[str, Any],
        *,
        symbol_hint: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        if not isinstance(entry, dict):
            return None

        info = entry.get("info") if isinstance(entry.get("info"), dict) else None
        source = info if isinstance(info, dict) else entry

        symbol_id = source.get("symbol") or entry.get("symbol")
        symbol = symbol_hint
        if symbol is None and symbol_id:
            symbol = self._market_id_to_symbol(symbol_id)
        if symbol is None:
            symbol = symbol_id

        order_side = None
        raw_side = source.get("side") or entry.get("side")
        if isinstance(raw_side, str):
            side_lower = raw_side.lower()
            if side_lower in {"buy", "sell"}:
                order_side = side_lower

        position_side = None
        if order_side is not None:
            position_side = "buy" if order_side == "sell" else "sell"
        else:
            alt_side = entry.get("side")
            if isinstance(alt_side, str):
                alt_lower = alt_side.lower()
                if alt_lower in {"long", "short"}:
                    position_side = "buy" if alt_lower == "long" else "sell"

        closed_size = None
        size_candidates = [
            source.get("closedSize"),
            source.get("qty"),
            source.get("closedQty"),
            entry.get("contracts"),
            entry.get("amount"),
            entry.get("size"),
        ]
        for cand in size_candidates:
            cand_value = self._float_or_none(cand)
            if cand_value is not None:
                closed_size = cand_value
                break

        avg_exit_price = self._float_or_none(
            source.get("avgExitPrice") or entry.get("avgExitPrice")
        )
        if avg_exit_price is None:
            cum_exit_value = self._float_or_none(
                source.get("cumExitValue") or entry.get("cumExitValue")
            )
            if cum_exit_value is not None and closed_size not in (None, 0.0):
                avg_exit_price = cum_exit_value / closed_size

        avg_entry_price = self._float_or_none(
            source.get("avgEntryPrice") or entry.get("avgEntryPrice")
        )
        if avg_entry_price is None:
            cum_entry_value = self._float_or_none(
                source.get("cumEntryValue") or entry.get("cumEntryValue")
            )
            qty_val = self._float_or_none(source.get("qty") or entry.get("qty"))
            if cum_entry_value is not None and qty_val not in (None, 0.0):
                avg_entry_price = cum_entry_value / qty_val

        closed_pnl = self._float_or_none(
            source.get("closedPnl")
            or entry.get("closedPnl")
            or source.get("realizedPnl")
            or entry.get("realizedPnl")
        )

        open_fee = self._float_or_none(source.get("openFee") or entry.get("openFee"))
        close_fee = self._float_or_none(source.get("closeFee") or entry.get("closeFee"))
        funding_fee = self._float_or_none(
            source.get("fundingFee")
            or source.get("cumFundingFee")
            or entry.get("fundingFee")
            or entry.get("cumFundingFee")
        )

        created_time = self._int_or_none(
            source.get("createdTime")
            or entry.get("createdTime")
            or entry.get("timestamp")
        )
        updated_time = self._int_or_none(
            source.get("updatedTime")
            or entry.get("updatedTime")
            or entry.get("timestamp")
        )

        order_id = source.get("orderId") or entry.get("orderId") or entry.get("id")
        exec_type = source.get("execType") or entry.get("execType")

        unique_components = [
            str(symbol_id or symbol or ""),
            str(order_id or ""),
            str(updated_time or created_time or ""),
            str(order_side or position_side or ""),
        ]
        unique_id = ":".join(filter(None, unique_components)) or None

        return {
            "symbol": symbol,
            "symbol_id": symbol_id,
            "entry_side": position_side,
            "order_side": order_side,
            "closed_size": closed_size,
            "avg_exit_price": avg_exit_price,
            "avg_entry_price": avg_entry_price,
            "closed_pnl": closed_pnl,
            "open_fee": open_fee,
            "close_fee": close_fee,
            "funding_fee": funding_fee,
            "created_time": created_time,
            "updated_time": updated_time,
            "order_id": order_id,
            "exec_type": exec_type,
            "raw": entry,
            "unique_id": unique_id,
        }

    def get_position_history(
        self,
        symbol: Optional[str] = None,
        since_ms: Optional[int] = None,
        limit: int = 50,
        *,
        category: str = "linear",
        settle_coin: Optional[str] = None,
        max_pages: int = 5,
    ) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        cursor: Optional[str] = None
        pages = 0

        symbol_id = self._symbol_to_market_id(symbol) if symbol else None

        while pages < max_pages:
            request: Dict[str, Any] = {
                "category": category,
            }
            if symbol_id:
                request["symbol"] = symbol_id
            if since_ms is not None:
                request["startTime"] = int(since_ms)
            if limit:
                request["limit"] = int(min(max(limit, 1), 200))
            if settle_coin:
                request["settleCoin"] = settle_coin
            if cursor:
                request["cursor"] = cursor

            merged = self._merge_params(
                {k: v for k, v in request.items() if v is not None}
            )

            api_method = getattr(self.exchange, "privateGetV5PositionClosedPnl", None)
            rows: List[Dict[str, Any]] = []
            result: Dict[str, Any] = {}

            if callable(api_method):
                try:
                    raw = api_method(merged)
                    if isinstance(raw, dict):
                        result = raw.get("result", {}) or {}
                        rows = result.get("list", []) or []
                except Exception:
                    rows = []

            if not rows:
                try:
                    fetch_position_history = getattr(
                        self.exchange, "fetch_position_history", None
                    )
                    fetch_positions_history = getattr(
                        self.exchange, "fetch_positions_history", None
                    )
                    if callable(fetch_position_history):
                        rows = (
                            fetch_position_history(symbol, since_ms, limit, merged)
                            or []
                        )
                        for row in rows:
                            normalized = self._normalize_closed_pnl_entry(
                                row,
                                symbol_hint=symbol,
                            )
                            if normalized:
                                records.append(normalized)
                        break
                    if callable(fetch_positions_history):
                        rows = (
                            fetch_positions_history(symbol, since_ms, limit, merged)
                            or []
                        )
                        for row in rows:
                            normalized = self._normalize_closed_pnl_entry(
                                row,
                                symbol_hint=symbol,
                            )
                            if normalized:
                                records.append(normalized)
                        break
                except Exception:
                    break

            if not rows:
                break

            for row in rows:
                normalized = self._normalize_closed_pnl_entry(row, symbol_hint=symbol)
                if normalized:
                    records.append(normalized)

            cursor = None
            if isinstance(result, dict):
                cursor = result.get("nextPageCursor") or result.get("next_page_cursor")

            pages += 1
            if not cursor:
                break

        records.sort(key=lambda r: r.get("updated_time") or r.get("created_time") or 0)
        return records

    def get_balance(self, currency: str = "USDT") -> Optional[Dict[str, Any]]:
        try:
            balance = self.exchange.fetch_balance(self._default_params())
            if balance is None:
                return None
            total = balance.get("total", {}).get(currency)
            free = balance.get("free", {}).get(currency)
            used = balance.get("used", {}).get(currency)
            return {
                "currency": currency,
                "total": total,
                "free": free,
                "used": used,
                "raw": balance,
            }
        except Exception as e:
            print(f"Error fetching balance: {e}")
            return None

    def set_leverage(
        self, symbol: str, leverage: float, margin_mode: str = "cross"
    ) -> Optional[Any]:
        try:
            # ccxt 통합 API 우선 사용
            if hasattr(self.exchange, "set_leverage"):
                return self.exchange.set_leverage(
                    leverage, symbol, self._merge_params({"marginMode": margin_mode})
                )
            print("set_leverage not supported on this ccxt version")
            return None
        except Exception as e:
            msg = str(e)
            # Bybit: 110043 leverage not modified -> 정상/무시
            if "110043" in msg or "leverage not modified" in msg.lower():
                print("Leverage unchanged (110043); continuing")
                return None
            print(f"Error setting leverage: {e}")
            return None

    def update_symbol_tpsl(
        self,
        symbol: str,
        *,
        take_profit: Optional[float] = None,
        stop_loss: Optional[float] = None,
        side: Optional[str] = None,
        position_idx: Optional[int] = None,
    ) -> Dict[str, Any]:
        if take_profit is None and stop_loss is None:
            return {"status": "noop"}

        idx = position_idx
        if idx is None and side is not None:
            try:
                side_norm = str(side).strip().lower()
                if side_norm in {"long", "buy"}:
                    idx = 1
                elif side_norm in {"short", "sell"}:
                    idx = 2
            except Exception:
                idx = position_idx

        params: Dict[str, Any] = {}
        category = self._resolve_category(symbol)
        if category:
            params["category"] = category
        if idx is not None:
            params["positionIdx"] = idx

        last_error: Optional[str] = None

        if hasattr(self.exchange, "set_trading_stop"):
            try:
                merged = self._merge_params(params)
                response = self.exchange.set_trading_stop(
                    symbol,
                    stopLoss=stop_loss,
                    takeProfit=take_profit,
                    params=merged,
                )
                return {"status": "ok", "raw": response}
            except Exception as exc:
                last_error = str(exc)

        private_method = getattr(
            self.exchange, "privatePostV5PositionTradingStop", None
        )
        if callable(private_method):
            try:
                request: Dict[str, Any] = {}
                market_id = self._symbol_to_market_id(symbol)
                if market_id:
                    request["symbol"] = market_id
                if category:
                    request["category"] = category
                if take_profit is not None:
                    request["takeProfit"] = str(take_profit)
                if stop_loss is not None:
                    request["stopLoss"] = str(stop_loss)
                if idx is not None:
                    request["positionIdx"] = idx
                response = private_method(self._merge_params(request))
                return {"status": "ok", "raw": response}
            except Exception as exc:
                last_error = str(exc)

        if last_error:
            print(f"Error updating TP/SL: {last_error}")
            return {"status": "error", "error": last_error}

        print("update_symbol_tpsl not supported on this ccxt version")
        return {"status": "error", "error": "unsupported"}

    def get_last_price(self, symbol: str) -> Optional[float]:
        try:
            ticker = self.exchange.fetch_ticker(symbol, self._default_params())
            return ticker.get("last") if ticker else None
        except Exception as e:
            print(f"Error fetching ticker: {e}")
            return None

    def get_positions(self):
        """포지션 정보들 조회"""
        try:
            # symbols=None 로 전체, recv_window 포함
            self.positions: dict = self.exchange.fetch_positions(
                None, self._default_params()
            )

            if self.positions is None:
                return None

            return self.positions

        except Exception as e:
            print(f"Error fetching positions: {e}")
            return None

    def get_position(self, num=1):
        """하나의 포지션 정보만 조회"""
        try:
            self.positions: dict = self.exchange.fetch_positions(
                None, self._default_params()
            )

            if self.positions is None:
                return None

            return self.positions[num]["info"]

        except Exception as e:
            print(f"Error fetching positions: {e}")
            return None

    def open_position(self, position: Open_Position):
        try:
            # 레버리지 설정 (선택)
            skip_leverage = False
            existing_positions: List[Dict[str, Any]] = []
            if position.leverage is not None:
                try:
                    existing_positions = (
                        self.get_positions_by_symbol(position.symbol) or []
                    )
                    skip_leverage = len(existing_positions) > 0
                except Exception:
                    skip_leverage = False

            if position.leverage is not None and skip_leverage:
                print(
                    f"Skipping leverage change for {position.symbol} because an active position already exists."
                )
            elif position.leverage is not None:
                try:
                    self.set_leverage(position.symbol, position.leverage)
                except Exception as le:
                    print(f"Warning: failed to set leverage: {le}")

            # 주문 생성 함수(공통) 정의
            def _place_with_quantity(qty: float):
                if position.type == "market":
                    if position.side == "sell":
                        return self.exchange.create_market_sell_order(
                            position.symbol,
                            qty,
                            params=self._merge_params(
                                {
                                    "takeProfit": position.tp,
                                    "stopLoss": position.sl,
                                }
                            ),
                        )
                    else:
                        return self.exchange.create_market_buy_order(
                            position.symbol,
                            qty,
                            params=self._merge_params(
                                {
                                    "takeProfit": position.tp,
                                    "stopLoss": position.sl,
                                }
                            ),
                        )
                elif position.type == "limit":
                    if position.side == "sell":
                        return self.exchange.create_limit_sell_order(
                            position.symbol,
                            qty,
                            position.price,
                            params=self._merge_params(
                                {
                                    "takeProfit": position.tp,
                                    "stopLoss": position.sl,
                                }
                            ),
                        )
                    else:
                        return self.exchange.create_limit_buy_order(
                            position.symbol,
                            qty,
                            position.price,
                            params=self._merge_params(
                                {
                                    "takeProfit": position.tp,
                                    "stopLoss": position.sl,
                                }
                            ),
                        )
                else:
                    raise ValueError(f"Unsupported order type: {position.type}")

            # 1차 시도
            try:
                order = _place_with_quantity(position.quantity)
                return order
            except Exception as first_error:
                msg = str(first_error)
                # Bybit 110007: ab not enough for new order (가용 마진 부족)
                if (
                    "110007" in msg
                    or "ab not enough" in msg.lower()
                    or "insufficient" in msg.lower()
                ):
                    try:
                        # 가용 잔고 및 마켓 스펙을 기반으로 수량 자동 조정 후 재시도
                        balance = self.get_balance("USDT") or {}
                        free_usdt = float(balance.get("free") or 0)
                        if free_usdt <= 0:
                            print(
                                "Insufficient free balance for retry after 110007; aborting"
                            )
                            return None

                        leverage = float(position.leverage or 1.0)
                        safety = float(os.getenv("ORDER_SAFETY_MARGIN", "0.96"))
                        # 가격은 전달된 price 사용(시장가일 경우 main에서 현재가를 전달함)
                        effective_price = float(position.price or 0)
                        if effective_price <= 0:
                            # 폴백: 티커에서 조회
                            last_price = self.get_last_price(position.symbol) or 0
                            effective_price = float(last_price)
                        if effective_price <= 0:
                            print(
                                "Cannot determine effective price for retry; aborting"
                            )
                            return None

                        # 최대 가능 수량 계산
                        max_position_value = free_usdt * leverage * safety
                        computed_max_qty = max_position_value / effective_price

                        # 마켓 스펙 로드
                        try:
                            market = self.exchange.market(position.symbol)
                            min_qty = (
                                (market.get("limits", {}) or {})
                                .get("amount", {})
                                .get("min")
                            )
                        except Exception:
                            market = None
                            min_qty = None

                        # 수량 반올림(정밀도/스텝에 맞춤) - 안전하게 소수 절삭 효과를 위해 amount_to_precision 사용
                        def _round_qty(symbol: str, qty: float) -> float:
                            try:
                                return float(
                                    self.exchange.amount_to_precision(symbol, qty)
                                )
                            except Exception:
                                return float(qty)

                        adjusted_qty = _round_qty(
                            position.symbol,
                            max(0.0, min(position.quantity, computed_max_qty)),
                        )

                        # 여전히 너무 크면 단계적으로 축소하며 최대 3회 재시도
                        retry_quantities = [
                            adjusted_qty,
                            _round_qty(position.symbol, adjusted_qty * 0.9),
                            _round_qty(position.symbol, adjusted_qty * 0.75),
                            _round_qty(position.symbol, adjusted_qty * 0.5),
                        ]

                        for idx, rq in enumerate(retry_quantities):
                            if rq is None or rq <= 0:
                                continue
                            if min_qty is not None and rq < float(min_qty):
                                # 최소 수량보다 작으면 스킵
                                continue
                            try:
                                print(
                                    f"Retrying order after 110007 with qty={rq} (attempt {idx + 1})"
                                )
                                order = _place_with_quantity(rq)
                                return order
                            except Exception as re:
                                # 같은 오류가 계속되면 루프 계속
                                if not (
                                    "110007" in str(re).lower()
                                    or "ab not enough" in str(re).lower()
                                    or "insufficient" in str(re).lower()
                                ):
                                    print(f"Retry failed with non-110007 error: {re}")
                                    return None
                        print(
                            "All retries after 110007 exhausted or qty < min; giving up"
                        )
                        return None
                    except Exception as adjust_error:
                        print(
                            f"Error during auto-adjust retry after 110007: {adjust_error}"
                        )
                        return None
                else:
                    # 기타 오류는 상위로 메시지만 남기고 실패 처리
                    print(f"Error opening position: {first_error}")
                    return None

        except Exception as e:
            print(f"Error opening position: {e}")
            return None

    # def edit_tp_sl(self, symbol, order_id, tp=None, sl=None):
    #     try:
    #         # 포지션 수정
    #         order = self.exchange.edit_order(order_id, symbol, params={
    #             'take_profit': tp,  # TP 가격
    #             'stop_loss': sl,  # SL 가격
    #         })
    #         return order

    #     except Exception as e:
    #         print(f"Error editing position: {e}")
    #         return None

    def close_position(self, symbol, order_id):
        try:
            # 포지션 청산
            order = self.exchange.close_position(order_id, symbol)
            return order

        except Exception as e:
            print(f"Error closing position: {e}")
            return None

    def close_all_positions(self):
        try:
            # 통합 API가 있는 경우 우선 사용
            if getattr(self.exchange, "close_all_positions", None):
                return self.exchange.close_all_positions()
            # 폴백: 보유 포지션을 reduceOnly 시장가로 청산
            positions = self.exchange.fetch_positions(None, self._default_params())
            results = []
            for p in positions or []:
                amount = p.get("contracts") or p.get("amount") or p.get("size")
                side = p.get("side")
                symbol = p.get("symbol")
                if not amount or not side or not symbol:
                    continue
                reduce_side = "sell" if side == "long" else "buy"
                try:
                    res = self.exchange.create_order(
                        symbol,
                        "market",
                        reduce_side,
                        abs(float(amount)),
                        None,
                        self._merge_params({"reduceOnly": True}),
                    )
                    results.append(res)
                except Exception as oe:
                    print(f"Error reduce-only close for {symbol}: {oe}")
            return results
        except Exception as e:
            print(f"Error closing positions: {e}")
            return None

    def get_orders(self):
        """주문들 받기"""
        try:
            orders = self.exchange.fetch_open_orders(
                None, None, None, self._default_params()
            )
            if len(orders) == 0:
                return None

            return orders

        except Exception as e:
            print(f"Error fetching orders: {e}")
            return None

    def get_closed_orders(
        self,
        symbol: Optional[str] = None,
        since_ms: Optional[int] = None,
        limit: int = 50,
    ):
        """닫힌 주문 목록 조회 (ccxt.fetch_closed_orders 래핑)"""
        try:
            return self.exchange.fetch_closed_orders(
                symbol, since_ms, limit, self._default_params()
            )
        except Exception as e:
            print(f"Error fetching closed orders: {e}")
            return []

    def get_my_trades(
        self,
        symbol: Optional[str] = None,
        since_ms: Optional[int] = None,
        limit: int = 100,
    ):
        """내 체결 이력 조회 (ccxt.fetch_my_trades 래핑)"""
        try:
            return self.exchange.fetch_my_trades(
                symbol, since_ms, limit, self._default_params()
            )
        except Exception as e:
            print(f"Error fetching my trades: {e}")
            return []

    def sum_pnl_usdt_from_trades(self, trades: list, side: str) -> Optional[float]:
        """체결 리스트로부터 대략적인 실현손익(USDT 기준)을 계산.
        선물 USDT 마진 기준으로 가정: PnL ≈ Σ( (sell_price - buy_price) * qty )
        (정확한 계산은 포지션별 진입/청산 매칭이 필요하여, 여기서는 VWAP 기반 근사)
        """
        try:
            if not trades:
                return None
            # 가격*수량 총액 비교로 근사 PnL 계산 (reduce side 대비 increase side)
            buy_value = 0.0
            sell_value = 0.0
            qty = 0.0
            for t in trades:
                price = t.get("price") or (t.get("info", {}) or {}).get("price")
                amount = t.get("amount")
                s = (t.get("side") or "").lower()
                if price is None or amount is None:
                    continue
                p = float(price)
                a = float(amount)
                if p <= 0 or a <= 0:
                    continue
                qty += a
                if s == "buy":
                    buy_value += p * a
                elif s == "sell":
                    sell_value += p * a
            if qty <= 0:
                return 0.0
            # 롱 기준: sell_value - buy_value, 숏 기준 반대
            if (side or "").lower() == "buy":
                return sell_value - buy_value
            else:
                return buy_value - sell_value
        except Exception as e:
            print(f"Error summing pnl from trades: {e}")
            return None

    def get_positions_by_symbol(self, symbol: str):
        """지정 심볼의 *활성* 포지션만 반환 (수량이 0인 항목은 제외)."""
        try:
            positions = (
                self.exchange.fetch_positions(None, self._default_params()) or []
            )
            active_positions = []
            for p in positions:
                if p.get("symbol") != symbol:
                    continue

                size_candidates = [
                    p.get("contracts"),
                    p.get("amount"),
                    p.get("size"),
                ]

                info = p.get("info") or {}
                if isinstance(info, dict):
                    size_candidates.extend(
                        [
                            info.get("contracts"),
                            info.get("amount"),
                            info.get("size"),
                            info.get("positionAmt"),
                        ]
                    )

                max_size = 0.0
                for cand in size_candidates:
                    if cand is None:
                        continue
                    try:
                        max_size = max(max_size, abs(float(cand)))
                    except Exception:
                        continue

                if max_size > 1e-8:
                    active_positions.append(p)

            return active_positions
        except Exception as e:
            print(f"Error fetching positions by symbol: {e}")
            return []

    def close_symbol_positions(self, symbol: str):
        """특정 심볼의 모든 포지션을 reduceOnly 시장가로 청산"""
        try:
            positions = (
                self.exchange.fetch_positions(None, self._default_params()) or []
            )
            results = []
            for p in positions:
                if p.get("symbol") != symbol:
                    continue
                amount = p.get("contracts") or p.get("amount") or p.get("size")
                side = p.get("side")
                if not amount or not side:
                    continue
                reduce_side = "sell" if side == "long" else "buy"
                try:
                    res = self.exchange.create_order(
                        symbol,
                        "market",
                        reduce_side,
                        abs(float(amount)),
                        None,
                        self._merge_params({"reduceOnly": True}),
                    )
                    results.append(res)
                except Exception as oe:
                    print(f"Error reduce-only close for {symbol}: {oe}")
            return results
        except Exception as e:
            print(f"Error closing symbol positions: {e}")
            return None

    def reduce_symbol_positions_percent(self, symbol: str, percent: float):
        """특정 심볼 포지션을 퍼센트만큼 부분 청산(reduceOnly 시장가).
        percent: 0~100 범위 권장. 100은 전량.
        """
        try:
            pct = float(percent)
            if pct <= 0:
                return []
            if pct > 100:
                pct = 100.0
            positions = (
                self.exchange.fetch_positions(None, self._default_params()) or []
            )
            results = []
            for p in positions:
                if p.get("symbol") != symbol:
                    continue
                amount = p.get("contracts") or p.get("amount") or p.get("size")
                side = p.get("side")
                if not amount or not side:
                    continue
                reduce_side = "sell" if side == "long" else "buy"
                qty = abs(float(amount)) * (pct / 100.0)
                if qty <= 0:
                    continue
                try:
                    res = self.exchange.create_order(
                        symbol,
                        "market",
                        reduce_side,
                        qty,
                        None,
                        self._merge_params({"reduceOnly": True}),
                    )
                    results.append(res)
                except Exception as oe:
                    print(f"Error partial reduce-only close for {symbol}: {oe}")
            return results
        except Exception as e:
            print(f"Error reducing symbol positions: {e}")
            return None

    def cancle_orders(self):
        try:
            orders = self.exchange.cancel_all_orders(None, self._default_params())
            if len(orders) == 0:
                return None

            return orders

        except Exception as e:
            print(f"Error fetching orders: {e}")
            return None

    def get_account_overview(self) -> Dict[str, Any]:
        """웹 UI에서 보여줄 간단 상태 요약"""
        try:
            balance = self.get_balance("USDT")
            positions = self.get_positions() or []
            return {
                "balance": balance,
                "positions": positions,
                "openOrders": None,
            }
        except Exception as e:
            print(f"Error building account overview: {e}")
            return {
                "balance": None,
                "positions": None,
                "openOrders": None,
            }


if __name__ == "__main__":
    bybit = BybitUtils(is_testnet=True)  # 테스트넷 사용
    value = bybit.get_positions()

    print(value)

    # order = bybit.open_position(
    #     Open_Position(
    #         symbol="XRP/USDT:USDT",
    #         side="buy",
    #         price=2.8,
    #         quantity=1000,
    #         tp=3.3,
    #         sl=2.7,
    #         type="market"
    #     )
    # )
    # print(order)
