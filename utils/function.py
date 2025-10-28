from google.ai.generativelanguage_v1beta.types import content
import google.generativeai as genai
import json


class make_to_object:
    def __init__(self):
        generation_config = {
            "temperature": 0,
            "top_p": 0.0,
            "top_k": 40,
            "max_output_tokens": 400,
            "response_schema": content.Schema(
                type=content.Type.OBJECT,
                required=["Status"],
                properties={
                    "Status": content.Schema(
                        type=content.Type.STRING, enum=["hold", "short", "long", "stop"]
                    ),
                    "tp": content.Schema(
                        type=content.Type.NUMBER,
                    ),
                    "sl": content.Schema(
                        type=content.Type.NUMBER,
                    ),
                    "price": content.Schema(
                        type=content.Type.NUMBER,
                    ),
                    "buy_now": content.Schema(
                        type=content.Type.BOOLEAN,
                    ),
                    "stop_order": content.Schema(
                        type=content.Type.BOOLEAN,
                    ),
                    "leverage": content.Schema(
                        type=content.Type.NUMBER,
                    ),
                    "close_now": content.Schema(
                        type=content.Type.BOOLEAN,
                    ),
                    "close_percent": content.Schema(
                        type=content.Type.NUMBER,
                    ),
                    "reduce_only": content.Schema(
                        type=content.Type.BOOLEAN,
                    ),
                },
            ),
            "response_mime_type": "application/json",
        }
        self.model = genai.GenerativeModel(
            model_name="gemini-2.0-flash-exp",
            generation_config=generation_config,
            system_instruction=(
                "입력된 트레이딩 결론을 JSON 오브젝트로 변환하시오.\n"
                "필수: Status in [hold,short,long,stop]. 선택: price, sl, tp, buy_now, leverage.\n"
                "시장가의 경우 buy_now를 true로 설정합니다.\n"
                "레버리지를 숫자로 제안할 수 있습니다(예: 3, 5, 10)."
            ),
        )
        self.chat_session = self.model.start_chat(history=[])

    def make_it_object(self, inputs: str):
        response = self.chat_session.send_message(inputs)
        obj = json.loads(response.text)
        return obj


if __name__ == "__main__":
    test = make_to_object()
    value = test.make_it_object("""**Summary:**

* **Action:** **SHORT** XRP/USDT
* **Entry:** **Market Order** around **2.6943**
* **Stop Loss:** **2.85**
* **Take Profit:** **2.40**""")
    print(value)
