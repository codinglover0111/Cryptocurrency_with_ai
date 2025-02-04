from google.ai.generativelanguage_v1beta.types import content
import google.generativeai as genai
import json

class make_to_object():
    def __init__(self):
        generation_config = {
        "temperature": 0,
        "top_p": 0.0,
        "top_k": 40,
        "max_output_tokens": 400,
        "response_schema": content.Schema(
            type = content.Type.OBJECT,
            required = ["Status"],
            properties = {
            "Status": content.Schema(
                type = content.Type.STRING,
                # TODO: 추후 포지션 청산 기능 추가
                enum = ["hold", "sell", "buy"]
            ),
            "tp": content.Schema(
                type = content.Type.NUMBER,
            ),
            "sl": content.Schema(
                type = content.Type.NUMBER,
            ),
            "price": content.Schema(
                type = content.Type.NUMBER,
            ),
            "buy_now": content.Schema(
                type = content.Type.BOOLEAN,
            ),
            },
        ),
        "response_mime_type": "application/json",
        }
        self.model = genai.GenerativeModel(
          model_name="gemini-2.0-flash-exp",
          generation_config=generation_config,
          system_instruction="입력된 값을 object로 변환하시오.\n시장가의 경우 market buy를 true로 합니다.",
        )
        self.chat_session = self.model.start_chat(
          history=[
          ]
        )
        pass
    
    def make_it_object(self,inputs:str):
        response = self.chat_session.send_message(inputs)
        dict = json.loads(response.text)
        return dict
    
    
if __name__ == '__main__':
    test = make_to_object()
    value = test.make_it_object("""**Summary:**

* **Action:** **SHORT** XRP/USDT
* **Entry:** **Market Order** around **2.6943**
* **Stop Loss:** **2.85**
* **Take Profit:** **2.40**""")
    print(value)