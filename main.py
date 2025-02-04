import os
import google.generativeai as genai
from utils import bybit_utils, BybitUtils, Open_Position, make_to_object
from dotenv import load_dotenv
import schedule
import time
from datetime import datetime
import pytz
import logging

# TODO: 추후 AI 관련 부분을 클래스로 묶어 리팩토링해야함

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
        # TODO 리팩토링 진행
        # if len(current_position)>0:
        #     logging.info("Active position exists, skipping trading cycle")
        #     return
        
        current_orders = bybit.get_orders()
        # if current_orders is not None and len(current_position)==0:
        # # 포지션이 없으며 현재 오더가 있으면 동작
        #   logging.info("Active position exists, cancle current orders")
        #   bybit.cancle_orders()
          

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
            system_instruction="You are a good crypto trader\nYou are fully aware of the risks of cryptocurrencies and know the dangers of leverage\nSend us a picture of a chart and tell us if you should sell or buy\nYou need to set a clear direction at this point\nYou need to choose between watch, sell, buy, and position user\nYou need to clearly tell us your purchase price (market or limit), stop loss, and take profit price. \nIf you have a contingent position, you can only choose one of two options: watch and wait or close with a hold or stop.",
        )

        # 차트 생성 및 가격 정보 획득
        # TODO 비동기적으로 진행하여 빠르게 이미지를 업로드 할 수 있게 해야함
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

        print(f"Current Price: {current_price}")
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
                },
                {
                    "role": "user",
                    "parts": [
                        files[1],
                        f"4시간 봉",
                    ],
                }
            ]
        )
        if len(current_position)>0:
            response = chat_session.send_message(content={
                "role": "user",
                "parts": [
                    files[1],
                    f"""포지션 사이드: {current_position[0]['info']['side']}
                    진입 가격:{current_position[0]['entryPrice']}
                    T/P:{current_position[0]['info']['takeProfit']}
                    S/L:{current_position[0]['info']['stopLoss']}
                    Current Price: {current_price}""",
                ],
            })
        else:
            response = chat_session.send_message(content={
                "role": "user",
                "parts": [
                    files[1],
                    f"""현재 포지션: 없음
                    Current Price: {current_price}""",
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
            elif value.get('buy_now') is None or value['buy_now'] == False:
              typevalue = 'limit'
              
            position_params = Open_Position(
                symbol="XRP/USDT:USDT",
                type=typevalue,
                price=value['price'],
                side=side,
                tp=value['tp'],
                sl=value['sl'],
                quantity=100
            )
            bybit.open_position(position_params)
            logging.info(f"Position opened: {position_params}")
            
        elif value['Status'] == "hold":
            logging.info("No trading signal generated")
            
        elif value['Status'] == "stop":
            logging.info("stop position(close_all)")
            bybit.close_all_positions()

    except Exception as e:
        logging.error(f"Error in automation: {str(e)}")

def run_scheduler():
    # 서울 시간대 설정
    seoul_tz = pytz.timezone('Asia/Seoul')
    current_time = datetime.now(seoul_tz)
    logging.info(f"Scheduler started at {current_time}")

    # 매 30분 마다 실행
    schedule.every().hour.at(":30").do(automation)
    
    # 매 15분마다 실행
    # schedule.every(15).minutes.do(automation)
    
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