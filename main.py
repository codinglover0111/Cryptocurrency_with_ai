import os
import google.generativeai as genai
from utils import bybit_utils, BybitUtils, Open_Position, make_to_object
from dotenv import load_dotenv
import schedule
import time
from datetime import datetime
import pytz
import logging

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading.log'),
        logging.StreamHandler()
    ]
)

load_dotenv()

def round_price(price):
    """XRP/USDT 가격을 최소 정밀도(0.0001)에 맞게 반올림"""
    return round(float(price), 4)

def automation():
    try:
        bybit = BybitUtils(True)
        
        # 현재 포지션 체크
        current_position = bybit.get_positions()
        if len(current_position)>0:
            logging.info("Active position exists, skipping trading cycle")
            return
        
        current_orders = bybit.get_orders()
        if current_orders is not None and len(current_position)==0:
        # 포지션이 없으며 현재 오더가 있으면 동작
          logging.info("Active position exists, cancle current orders")
          bybit.cancle_orders()
          

        # Gemini API 설정
        genai.configure(api_key=os.environ["GEMINI_API_KEY"])

        def upload_to_gemini(path, mime_type=None):
            """Uploads the given file to Gemini.

            See https://ai.google.dev/gemini-api/docs/prompting_with_media
            """
            file = genai.upload_file(path, mime_type=mime_type)
            print(f"Uploaded file '{file.display_name}' as: {file.uri}")
            return file

        generation_config = {
            "temperature": 0.2,
            "top_p": 0.95,
            "top_k": 64,
            "max_output_tokens": 65536,
            "response_mime_type": "text/plain",
        }

        model = genai.GenerativeModel(
            model_name="gemini-2.0-flash-thinking-exp-01-21",
            generation_config=generation_config,
            system_instruction="You are a great crypto trader\nI am fully aware of the risks of crypto and know the risks of leverage\nI will send you a picture of the chart and you will tell me if I should short or buy\nI need to set a clear direction at this time\nI need to choose between watch, short, buy, and position user\nI need to give me a clear buy price (market or limit), stop loss, and take profit price.",
        )

        # 차트 생성 및 가격 정보 획득
        price_utils = bybit_utils("XRP/USDT", "1h", 100)
        price_utils.plot_candlestick()
        price_utils.set_timeframe("4h")
        price_utils.plot_candlestick()
        price_utils.set_timeframe("15m")
        price_utils.plot_candlestick()
        current_price = price_utils.get_current_price()

        files = [
            upload_to_gemini("XRP_USDT_15m_candlestick.png", mime_type="image/png"),
            upload_to_gemini("XRP_USDT_1h_candlestick.png", mime_type="image/png"),
            upload_to_gemini("XRP_USDT_4h_candlestick.png", mime_type="image/png"),
        ]

        print(f"4시간 봉 Current Price: {current_price}")
        chat_session = model.start_chat(
            history=[
                {
                    "role": "model",
                    "parts": [
                        files[0],
                        "15분 봉"
                    ],
                },
                {
                    "role": "model",
                    "parts": [
                        files[1],
                        "1시간 봉"
                    ],
                }
            ]
        )

        response = chat_session.send_message(content={
            "role": "user",
            "parts": [
                files[1],
                f"4시간 봉 Current Price: {current_price}",
            ],
        })
        response = response.text
        print(response)

        object = make_to_object()
        value = object.make_it_object(response)
        print(value)
        
        # 트레이딩 실행
        if value['Status'] in ["buy", "sell"]:
            side = value['Status']
              
            if value['buy_now'] == True:
              typevalue = 'market'
            else:
              typevalue = 'limit'
              
            position_params = Open_Position(
                symbol="XRP/USDT:USDT",
                type=typevalue,
                price=value['price'],
                side=side,
                tp=value['tp'],
                sl=value['sl'],
                quantity=1
            )
            bybit.open_position(position_params)
            logging.info(f"Position opened: {position_params}")
        else:
            logging.info("No trading signal generated")

    except Exception as e:
        logging.error(f"Error in automation: {str(e)}")

def run_scheduler():
    # 서울 시간대 설정
    seoul_tz = pytz.timezone('Asia/Seoul')
    current_time = datetime.now(seoul_tz)
    logging.info(f"Scheduler started at {current_time}")

    # 4시간마다 실행
    schedule.every(4).hours.do(automation)
    
    # 초기 실행
    automation()
    
    while True:
        try:
            schedule.run_pending()
            time.sleep(1)
        except Exception as e:
            logging.error(f"Scheduler error: {str(e)}")
            time.sleep(60)  # 오류 발생시 1분 대기

if __name__ == '__main__':
    run_scheduler()