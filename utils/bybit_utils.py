import os
import ccxt
from dotenv import load_dotenv

from typing import Optional, Dict, Any, Literal
from pydantic import BaseModel


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
                    pass
            if server_time is not None:
                local_time = self.exchange.milliseconds()
                # 안전 마진(밀리초). 로컬 시계가 서버보다 앞서는 경우를 더 엄격히 보정
                safety_ms = int(os.getenv("BYBIT_TIME_SAFETY_MS", "500"))
                # timeDifference는 ccxt의 nonce 계산에 더해짐
                # 서버 시간 - 로컬 시간 - safety => 요청 타임스탬프가 서버 시간보다 약간 작게
                offset = int(server_time) - int(local_time) - safety_ms
                self.exchange.options["timeDifference"] = offset
            else:
                # 폴백: ccxt의 자동 보정 시도
                try:
                    self.exchange.load_time_difference()
                except Exception:
                    pass
            td = self.exchange.options.get("timeDifference")
            rw = self.exchange.options.get("recvWindow") or self.exchange.options.get(
                "recvwindow"
            )
            print(f"Bybit env: {mode}, timeDifference: {td} ms, recvWindow: {rw} ms")
        except Exception:
            pass

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
            print(f"Error setting leverage: {e}")
            return None

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
            if position.leverage is not None:
                try:
                    self.set_leverage(position.symbol, position.leverage)
                except Exception as le:
                    print(f"Warning: failed to set leverage: {le}")
            # 포지션 열기
            order = None
            if position.type == "market":
                # 시장가 주문
                if position.side == "sell":
                    order = self.exchange.create_market_sell_order(
                        position.symbol,
                        position.quantity,
                        params=self._merge_params(
                            {
                                "takeProfit": position.tp,  # TP 가격
                                "stopLoss": position.sl,  # SL 가격
                            }
                        ),
                    )
                else:
                    order = self.exchange.create_market_buy_order(
                        position.symbol,
                        position.quantity,
                        params=self._merge_params(
                            {
                                "takeProfit": position.tp,  # TP 가격
                                "stopLoss": position.sl,  # SL 가격
                            }
                        ),
                    )
            elif position.type == "limit":
                # 지정가 주문
                if position.side == "sell":
                    order = self.exchange.create_limit_sell_order(
                        position.symbol,
                        position.quantity,
                        position.price,
                        params=self._merge_params(
                            {
                                "takeProfit": position.tp,  # TP 가격kwargs
                                "stopLoss": position.sl,  # SL 가격
                            }
                        ),
                    )
                else:
                    order = self.exchange.create_limit_buy_order(
                        position.symbol,
                        position.quantity,
                        position.price,
                        params=self._merge_params(
                            {
                                "takeProfit": position.tp,  # TP 가격
                                "stopLoss": position.sl,  # SL 가격
                            }
                        ),
                    )
            if order is None:
                raise ValueError(f"Unsupported order type: {position.type}")
            return order

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

    # TODO: OrderID 저장?

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
        """지정 심볼의 포지션만 필터링해서 반환"""
        try:
            positions = (
                self.exchange.fetch_positions(None, self._default_params()) or []
            )
            return [p for p in positions if (p.get("symbol") == symbol)]
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
            orders = self.get_orders() or []
            return {
                "balance": balance,
                "positions": positions,
                "openOrders": orders,
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
