import ccxt
import pandas as pd
import matplotlib.pyplot as plt
import ta
import mplfinance as mpf


class bybit_utils:
    def __init__(self,symbol = 'XRP/USDT', timeframe = '4h', limit = 100):
        self.exchange = ccxt.bybit()
        self.symbol = symbol
        self.timeframe = timeframe
        self.limit = limit
        pass
    
    def set_timeframe(self, timeframe):
        self.timeframe = timeframe
        pass
    
    def get_ohlcv(self):
        ohlcv = self.exchange.fetch_ohlcv(self.symbol, self.timeframe, limit=self.limit)
        self.df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'], unit='ms')
        self.df.set_index('timestamp', inplace=True)
        return self.df
    
    def get_current_price(self):
        current_price = self.df['close'].iloc[-1]
        return current_price
    
    def get_bollinger_band(self):
        self.df['bb_upper'], self.df['bb_middle'], self.df['bb_lower'] = ta.volatility.bollinger_hband(self.df['close']), ta.volatility.bollinger_mavg(self.df['close']), ta.volatility.bollinger_lband(self.df['close'])
        return self.df
    
    def get_rsi(self):
        self.df['rsi'] = ta.momentum.rsi(self.df['close'])
        return self.df
    
    def plot_candlestick(self):
        self.get_ohlcv()
        self.get_ohlcv()
        self.get_current_price()
        self.get_bollinger_band()
        self.get_rsi()
        apds = [
            mpf.make_addplot(self.df['bb_upper'], color='red', linestyle='--'),  # 볼린저 밴드 상단
            mpf.make_addplot(self.df['bb_middle'], color='green', linestyle='--'),  # 볼린저 밴드 중간
            mpf.make_addplot(self.df['bb_lower'], color='red', linestyle='--'),  # 볼린저 밴드 하단
        ]
        file_name = self.symbol.replace('/','_')
        mpf.plot(self.df, type='candle', addplot=apds, style='charles', title=f'{self.symbol} Price with Bollinger Bands', volume=True, savefig=f'{file_name}_{self.timeframe}_candlestick.png')
        return self.df

if __name__ == '__main__':
    bybit = bybit_utils()
    bybit.get_ohlcv()
    bybit.get_current_price()
    bybit.get_bollinger_band()
    bybit.get_rsi()
    bybit.plot_candlestick()
    bybit = bybit_utils(timeframe='1h')
    bybit.get_ohlcv()
    bybit.get_current_price()
    bybit.get_bollinger_band()
    bybit.get_rsi()
    bybit.plot_candlestick()

# CCXT를 사용하여 거래소 데이터 가져오기
# exchange = ccxt.bybit()  # 바이낸스 사용
# symbol = 'XRP/USDT'
# timeframe = '4h'
# limit = 100

# # 캔들 데이터 가져오기
# ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)

# # 데이터프레임으로 변환
# df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
# df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
# df.set_index('timestamp', inplace=True)

# current_price = df['close'].iloc[-1]
# print(f'4시간 봉 Current Price: {current_price}')

# # 볼린저 밴드 계산
# df['bb_upper'], df['bb_middle'], df['bb_lower'] = ta.volatility.bollinger_hband(df['close']), ta.volatility.bollinger_mavg(df['close']), ta.volatility.bollinger_lband(df['close'])

# # RSI 계산
# df['rsi'] = ta.momentum.rsi(df['close'])

# # 캔들스틱 차트 그리기
# apds = [
#     mpf.make_addplot(df['bb_upper'], color='red', linestyle='--'),  # 볼린저 밴드 상단
#     mpf.make_addplot(df['bb_middle'], color='green', linestyle='--'),  # 볼린저 밴드 중간
#     mpf.make_addplot(df['bb_lower'], color='red', linestyle='--'),  # 볼린저 밴드 하단
# ]

# # 캔들스틱 차트
# mpf.plot(df, type='candle', addplot=apds, style='charles', title=f'{symbol} Price with Bollinger Bands', volume=True, savefig='crypto_candlestick.png')

# # RSI 차트 그리기
# plt.figure(figsize=(14, 5))
# plt.plot(df.index, df['rsi'], label='RSI', color='purple')
# plt.axhline(70, color='red', linestyle='--')
# plt.axhline(30, color='green', linestyle='--')
# plt.title('RSI')
# plt.legend()
# plt.savefig('crypto_rsi.png')
# plt.close()