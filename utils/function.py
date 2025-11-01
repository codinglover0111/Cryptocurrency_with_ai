from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

from openai import OpenAI  # type: ignore


class make_to_object:
    """LLM 응답을 JSON 오브젝트로 강제 변환하기 위한 헬퍼."""

    def __init__(
        self, client: Optional[OpenAI] = None, *, model: Optional[str] = None
    ) -> None:
        if client is None:
            base_url = os.environ.get("OPENAI_BASE_URL")
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY가 필요합니다")
            if base_url:
                self._client = OpenAI(base_url=base_url, api_key=api_key)
            else:
                self._client = OpenAI(api_key=api_key)
        else:
            self._client = client

        self._model = model or os.environ.get("OPENAI_MODEL", "deepseek-reasoner")
        self._system_prompt = (
            "입력된 트레이딩 결론을 JSON 오브젝트로 변환하시오.\n"
            "필수: Status in [hold, short, long, stop]. 선택: price, sl, tp, buy_now, stop_order, leverage, close_now, close_percent, reduce_only, explain.\n"
            "숫자 필드는 숫자 타입, 불리언 필드는 불리언 타입으로 반환하십시오."
        )

    def make_it_object(self, inputs: str) -> Dict[str, Any]:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": inputs},
            ],
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content
        if not content:
            raise RuntimeError("OpenAI JSON 변환 응답이 비었습니다.")
        return json.loads(content)


if __name__ == "__main__":
    test = make_to_object()
    value = test.make_it_object(
        """**Summary:**

* **Action:** **SHORT** XRP/USDT
* **Entry:** **Market Order** around **2.6943**
* **Stop Loss:** **2.85**
* **Take Profit:** **2.40**"""
    )
    print(value)
