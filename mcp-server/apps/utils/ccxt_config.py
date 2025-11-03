import ccxt
import os
import time
from functools import wraps

from dotenv import load_dotenv
from uvicorn.main import logger

load_dotenv()


class CCXTConfig:
    """CCXT 설정을 관리합니다."""

    def __init__(self):
        self.env = os.getenv("ENV")
        self.api_key = os.getenv("BYBIT_API_KEY")
        self.api_secret = os.getenv("BYBIT_API_SECRET")
        self.recv_window_ms = os.getenv("BYBIT_RECV_WINDOW_MS")
        self.max_position_ratio = float(os.getenv("MAX_POSITION_RATIO"))

        if not self.max_position_ratio:
            self.max_position_ratio = 0.2
            logger.warning("MAX_POSITION_RATIO is not set, using default value: 0.2")

        if not self.api_key or not self.api_secret:
            raise ValueError("API Key or Secret is not set")

        self.exchange = ccxt.bybit(
            {
                "apiKey": self.api_key,
                "secret": self.api_secret,
                "enableRateLimit": True,  # 요청 제한 활성화
                "options": {
                    "defaultType": "future",  # 선물 거래인 경우
                    # CCXT 권장: 서버-로컬 시간 차 자동 보정
                    "adjustForTimeDifference": True,
                    # Bybit V5 인증 헤더 X-BAPI-RECV-WINDOW (ms)
                    # 환경변수 BYBIT_RECV_WINDOW_MS 로 오버라이드 가능
                    "recvWindow": int(self.recv_window_ms or "15000"),
                    # 일부 CCXT 버전 호환 (옵션 키 소문자 변형)
                    "recvwindow": int(self.recv_window_ms or "15000"),
                },
            }
        )

        if self.env == "demo":
            self.exchange.enable_demo_trading(enable=True)

    def sync_time(self):
        """서버 시간 동기화"""
        self.exchange.load_time_difference()

    def get_positions(self):
        """가지고 있는 모든 포지션 조회"""
        positions = self.exchange.fetch_positions()
        return positions

    def get_balance(self):
        """잔고 조회"""
        balance = self.exchange.fetch_balance()
        return balance

    # TODO: 심볼의 원래 가격을 가져오는 함수 제작
    def get_symbol_price(self, symbol: str):
        """심볼의 원래 가격 조회"""
        symbol += ":USDT"
        ticker_struct = self.exchange.fetch_ticker(symbol)
        price = ticker_struct["last"]
        return price

    def get_position_size(self, symbol: str, leverage: int):
        """
        가용 가능 포지션 크기 조회
        return:
        """
        # 포지션 크기 = (잔고 * 레버리지)*1회당 최대 포지션 비율(20%) / 심볼 가격
        price = self.get_symbol_price(symbol)
        balance = self.get_balance()["USDT"]["total"]
        position_size = (balance * leverage) * self.max_position_ratio / price
        return position_size


if __name__ == "__main__":
    ccxt_config = CCXTConfig()
    # print(ccxt_config.get_positions())
    # print(ccxt_config.get_balance())
    print(ccxt_config.get_position_size("BTC/USDT", 100))
