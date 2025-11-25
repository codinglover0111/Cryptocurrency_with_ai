"""멀티 에이전트/Adaptive-OPRO 기본 설정."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping


CONFIG_DIR = Path(__file__).resolve().parent
RUNTIME_CONFIG_PATH = CONFIG_DIR / "runtime_config.json"


AGENT_CONFIG: Dict[str, Dict[str, Any]] = {
    "indicator_agent": {
        "provider": "openai",
        "model": "gpt-4o-mini",
        "temperature": 0.1,
    },
    "pattern_agent": {
        "provider": "gemini",
        "model": "gemini-2.0-flash-exp",
        "temperature": 0.1,
    },
    "trend_agent": {
        "provider": "gemini",
        "model": "gemini-2.0-flash-exp",
        "temperature": 0.1,
    },
    "decision_agent": {
        "provider": "openrouter",
        "model": "deepseek/deepseek-chat",
        "temperature": 0.1,
    },
}

SCHEDULER_CONFIG: Dict[str, Any] = {
    "automation_minutes": 15,
    "loss_review_minutes": 60,
    "cold_start": False,
}

RISK_CONFIG: Dict[str, Any] = {
    "default_leverage": 5,
    "max_loss_percent": 40,  # 레버리지 적용 후 기준
    "position_allocation_percent": 20,  # 포지션당 최대 할당 % (초기 잔고 기준)
}

ADAPTIVE_OPRO_CONFIG: Dict[str, Any] = {
    "performance_window": 20,
    "min_trades_for_update": 5,
    "optimizer_model": "openai:gpt-4o-mini",
    "scorer_model": "openai:gpt-4o-mini",
    "sideways_threshold": 25.0,
}


def _runtime_defaults() -> Dict[str, Any]:
    return {
        "agents": deepcopy(AGENT_CONFIG),
        "scheduler": deepcopy(SCHEDULER_CONFIG),
        "adaptive_opro": deepcopy(ADAPTIVE_OPRO_CONFIG),
        "risk": deepcopy(RISK_CONFIG),
    }


def load_runtime_config() -> Dict[str, Any]:
    """런타임 설정(JSON)을 로드하고 기본값과 병합."""

    defaults = _runtime_defaults()
    if not RUNTIME_CONFIG_PATH.exists():
        return defaults

    try:
        data = json.loads(RUNTIME_CONFIG_PATH.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return defaults
    except Exception:
        return defaults

    merged = deepcopy(defaults)
    for section, values in data.items():
        if isinstance(values, Mapping) and section in merged:
            merged_section = merged[section]
            if isinstance(merged_section, MutableMapping):
                merged_section.update(values)
            else:
                merged[section] = values
        else:
            merged[section] = values
    return merged


def save_runtime_config(config: Mapping[str, Any]) -> None:
    """설정을 디스크에 저장."""

    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with RUNTIME_CONFIG_PATH.open("w", encoding="utf-8") as fp:
        json.dump(config, fp, indent=2, ensure_ascii=False)


def update_runtime_config(section: str, payload: Mapping[str, Any]) -> Dict[str, Any]:
    """특정 섹션을 갱신하고 결과를 반환."""

    runtime = load_runtime_config()
    current_section = runtime.setdefault(section, {})
    if isinstance(current_section, MutableMapping):
        current_section.update(payload)
    else:
        runtime[section] = dict(payload)
        current_section = runtime[section]
    save_runtime_config(runtime)
    return dict(current_section)

