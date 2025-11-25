"""Adaptive-OPRO 최적화 루프."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Tuple

from .meta_prompt import MetaPromptManager
from .scorer import PerformanceScorer


class AdaptiveOPRO:
    """시장 레짐과 성과 데이터를 이용해 메타 프롬프트를 생성."""

    def __init__(
        self,
        *,
        meta_prompt_manager: MetaPromptManager,
        scorer: PerformanceScorer,
        optimizer_model: str,
        performance_window: int,
        min_trades: int,
    ) -> None:
        self.meta_prompt_manager = meta_prompt_manager
        self.scorer = scorer
        self.optimizer_model = optimizer_model
        self.performance_window = performance_window
        self.min_trades = min_trades

    def _trajectory(self, prompt_trace: List[Dict[str, Any]]) -> List[Tuple[str, float]]:
        pairs: List[Tuple[str, float]] = []
        for entry in prompt_trace[-10:]:
            prompt = entry.get("prompt")
            score = entry.get("score", 0.0)
            if isinstance(prompt, str):
                try:
                    score_val = float(score)
                except Exception:
                    score_val = 0.0
                pairs.append((prompt, score_val))
        return pairs

    def generate_meta_prompt(self, state: Dict[str, Any]) -> Dict[str, Any]:
        prompt_trace = list(state.get("prompt_trace") or [])
        regime = state.get("regime", "unknown")
        performance = state.get("performance_snapshot") or {}

        trajectory = self._trajectory(prompt_trace)
        meta_prompt = self.meta_prompt_manager.build_meta_prompt(
            trajectory=trajectory,
            market_regime=regime,
            feedback={
                "roi": float(performance.get("roi", 0.0)),
                "sharpe": float(performance.get("sharpe", 0.0)),
                "max_dd": float(performance.get("max_drawdown", 0.0)),
            },
        )

        prompt_trace.append(
            {
                "ts": datetime.now(timezone.utc).isoformat(),
                "prompt": meta_prompt,
                "regime": regime,
                "score": performance.get("roi", 0.0),
            }
        )
        prompt_trace = prompt_trace[-20:]

        return {
            "meta_prompt": meta_prompt,
            "prompt_trace": prompt_trace,
        }

