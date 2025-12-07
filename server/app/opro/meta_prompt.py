"""Meta Prompt Manager."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple


@dataclass(slots=True)
class PromptCandidate:
    prompt: str
    score: float


class MetaPromptManager:
    """이전 프롬프트 궤적을 기반으로 메타 프롬프트 생성."""

    def build_meta_prompt(
        self,
        trajectory: Iterable[Tuple[str, float]],
        market_regime: str,
        feedback: Dict[str, float],
    ) -> str:
        """Adaptive-OPRO에서 사용할 메타 프롬프트 생성."""

        lines: List[str] = [
            "# Adaptive-OPRO Meta Prompt",
            f"- Market regime: {market_regime}",
            f"- Recent ROI: {feedback.get('roi', 0):.2f}%",
            f"- Sharpe: {feedback.get('sharpe', 0):.2f}",
            f"- Max Drawdown: {feedback.get('max_dd', 0):.2f}%",
            "## Prompt Trajectory (top 5)",
        ]

        for idx, (prompt, score) in enumerate(list(trajectory)[:5], start=1):
            lines.append(f"{idx}. score={score:.2f} → {prompt[:300]}...")

        lines.append("## Optimization Goals")
        if market_regime == "sideways":
            lines.append("- Favor range-bound strategies and tighter TP/SL (1-2%).")
            lines.append("- Highlight fake breakout risks and volume divergence.")
        elif market_regime == "bullish":
            lines.append("- Allow breakout trades with trailing stops.")
        elif market_regime == "bearish":
            lines.append("- Prioritize short setups and capital preservation.")
        else:
            lines.append("- Regime unknown: default to neutral risk stance.")

        lines.append("## Output Requirements")
        lines.append("- Respond in Korean.")
        lines.append("- Provide bullet point reasoning.")

        return "\n".join(lines)

