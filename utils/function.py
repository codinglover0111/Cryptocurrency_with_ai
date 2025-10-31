"""JSON 파싱 헬퍼 - LLM 텍스트 응답을 구조화된 객체로 변환."""

import json
from typing import Any, Dict


class make_to_object:
    """LLM의 텍스트 응답을 JSON 객체로 변환하는 폴백 파서.
    
    OpenAI의 JSON 모드가 실패할 경우 사용되는 간단한 파서입니다.
    """

    def __init__(self):
        """초기화 - 현재는 빈 클래스입니다."""
        pass

    def make_it_object(self, inputs: str) -> Dict[str, Any]:
        """텍스트 입력을 JSON 객체로 변환.
        
        Args:
            inputs: LLM의 텍스트 응답 (JSON 형식일 것으로 예상)
            
        Returns:
            파싱된 JSON 딕셔너리
            
        Raises:
            json.JSONDecodeError: 입력이 유효한 JSON이 아닌 경우
        """
        # JSON 코드 블록에서 JSON 추출 시도
        if "```json" in inputs:
            start = inputs.find("```json") + 7
            end = inputs.find("```", start)
            if end > start:
                inputs = inputs[start:end].strip()
        elif "```" in inputs:
            start = inputs.find("```") + 3
            end = inputs.find("```", start)
            if end > start:
                inputs = inputs[start:end].strip()
        
        # JSON 객체 추출 시도
        if "{" in inputs:
            start = inputs.find("{")
            end = inputs.rfind("}") + 1
            if end > start:
                inputs = inputs[start:end].strip()
        
        obj = json.loads(inputs)
        return obj


if __name__ == "__main__":
    test = make_to_object()
    value = test.make_it_object("""**Summary:**

* **Action:** **SHORT** XRP/USDT
* **Entry:** **Market Order** around **2.6943**
* **Stop Loss:** **2.85**
* **Take Profit:** **2.40**""")
    print(value)
