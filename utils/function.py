import json
import os
from typing import Any, Dict

from openai import OpenAI  # type: ignore


class make_to_object:
    """OpenAI 모델을 사용해 요약 텍스트를 구조화 JSON으로 파싱."""

    def __init__(self) -> None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY 환경변수가 필요합니다")

        base_url = os.environ.get("OPENAI_BASE_URL")
        client_kwargs: Dict[str, str] = {"api_key": f"{api_key}"}
        if base_url:
            client_kwargs["base_url"] = base_url
        self._client = OpenAI(**client_kwargs)
        self._model = os.environ.get("OPENAI_MODEL", "deepseek-reasoner")

    def make_it_object(self, inputs: str) -> Dict[str, Any]:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "입력된 트레이딩 결론을 JSON 오브젝트로 변환하시오.\n"
                        "필수: Status in [hold,short,long,stop]. 선택: price, sl, tp, buy_now, leverage.\n"
                        "시장가의 경우 buy_now를 true로 설정합니다.\n"
                        "레버리지를 숫자로 제안할 수 있습니다(예: 3, 5, 10).\n"
                        "기존 포지션의 TP/SL만 수정하려면 update_existing를 true로 설정하고 leverage는 비우십시오."
                    ),
                },
                {"role": "user", "content": inputs},
            ],
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content or "{}"
        return json.loads(content)


if __name__ == "__main__":
    test = make_to_object()
    value = test.make_it_object("""**Summary:**

* **Action:** **SHORT** XRP/USDT
* **Entry:** **Market Order** around **2.6943**
* **Stop Loss:** **2.85**
* **Take Profit:** **2.40**""")
    print(value)
