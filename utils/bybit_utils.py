import os
import ccxt
from dotenv import load_dotenv

from typing import Optional,Dict,Any
from enum import Enum,auto

from pydantic import BaseModel

from typing import Optional,Literal
from pydantic import BaseModel
from enum import Enum,auto

class Open_Position(BaseModel):
    symbol:str
    side:Literal["buy","sell"]
    quantity:float
    price:float
    type:Literal['limit','market']
    tp:Optional[float]
    sl:Optional[float]
    


dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(dotenv_path)

    
    
class BybitUtils:
    def __init__(self, is_testnet=True):
        try:
            # API 키와 시크릿 설정
            api_key = os.environ['DEMO_BYBIT_API_KEY'] if is_testnet else os.environ['BYBIT_API_KEY']
            api_secret = os.environ['DEMO_BYBIT_API_SECRET'] if is_testnet else os.environ['BYBIT_API_SECRET']
            # Bybit 객체 생성
            self.exchange = ccxt.bybit({
                'apiKey': api_key,
                'secret': api_secret,
                'enableRateLimit': True,  # 요청 제한 활성화
                'options': {
                    'defaultType': 'future',  # 선물 거래인 경우
                },
            })
            self.exchange.enable_demo_trading(is_testnet)  # 데모 트레이딩 활성화
        except Exception as e:
            print(f"Initialization error: {e}")
 
    def get_positions(self):
        """포지션 정보들 조회"""
        try:
            self.positions: dict  = self.exchange.fetch_positions()
            
            if self.positions is None:
                return None
            
            return self.positions
        
        except Exception as e:
            print(f"Error fetching positions: {e}")
            return None
        
    def get_position(self, num=1):
        """하나의 포지션 정보만 조회"""
        try:
            self.positions: dict  = self.exchange.fetch_positions()
            
            if self.positions is None:
                return None
            
            return self.positions[num]['info']
        
        except Exception as e:
            print(f"Error fetching positions: {e}")
            return None
    
    def open_position(self,
                      position: Open_Position):
        try:
            # 포지션 열기
            if position.type == 'market':
                # 시장가 주문
                if position.side == 'sell':
                    order = self.exchange.create_market_sell_order(position.symbol, position.quantity,params={
                        'takeProfit': position.tp,  # TP 가격
                        'stopLoss': position.sl,  # SL 가격
                    })
                else:
                    order = self.exchange.create_market_buy_order(position.symbol, position.quantity,params={
                        'takeProfit': position.tp,  # TP 가격
                        'stopLoss': position.sl,  # SL 가격
                    })
            elif position.type=="limit":
                # 지정가 주문
                if position.side == 'sell':
                    order = self.exchange.create_limit_sell_order(position.symbol, position.quantity, position.price,params={
                        'takeProfit': position.tp,  # TP 가격kwargs
                        'stopLoss': position.sl,  # SL 가격
                    })
                else:
                    order = self.exchange.create_limit_buy_order(position.symbol, position.quantity, position.price,params={
                        'takeProfit': position.tp,  # TP 가격
                        'stopLoss': position.sl,  # SL 가격
                    })            
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
            order = self.exchange.cancel_order(order_id, symbol)
            return order
        
        except Exception as e:
            print(f"Error closing position: {e}")
            return None
        
        
    def get_orders(self):
        """주문들 받기"""
        try:
            orders = self.exchange.fetch_open_orders()
            if len(orders) == 0:
                return None
            
            return orders
        
        except Exception as e:
            print(f"Error fetching orders: {e}")
            return None
        
    def cancle_orders(self):
        try:
            orders = self.exchange.cancel_all_orders()
            if len(orders) == 0:
                return None
            
            return orders
        
        except Exception as e:
            print(f"Error fetching orders: {e}")
            return None
        
        
if __name__ == '__main__':
    bybit = BybitUtils(is_testnet=True)  # 테스트넷 사용
    value = bybit.get_orders()
    
    print(value)
    
    order = bybit.open_position(
        Open_Position(
            symbol="XRP/USDT:USDT",
            side="buy",
            price=2.8,
            quantity=1000,
            tp=3.3,
            sl=2.7,
            type="market"
        )
    )
    # print(order)
    
    