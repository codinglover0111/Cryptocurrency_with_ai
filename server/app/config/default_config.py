"""멀티 에이전트/Adaptive-OPRO 기본 설정."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Optional


CONFIG_DIR = Path(__file__).resolve().parent
RUNTIME_CONFIG_PATH = CONFIG_DIR / "runtime_config.json"


AGENT_CONFIG: Dict[str, Dict[str, Any]] = {
    "indicator_agent": {
        "provider": "openrouter",
        "model": "openai/gpt-5.1",
        "temperature": 0.7,
    },
    "pattern_agent": {
        "provider": "openrouter",
        "model": "google/gemini-3-pro-preview",
        "temperature": 1,
    },
    "trend_agent": {
        "provider": "openrouter",
        "model": "google/gemini-3-pro-preview",
        "temperature": 1,
    },
    "decision_agent": {
        "provider": "openrouter",
        "model": "deepseek/deepseek-chat-v3.1",
        "temperature": 0.8,
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


def _get_trade_store():
    """TradeStore 싱글톤 인스턴스를 가져옵니다. 순환 참조 방지를 위해 함수 내 임포트."""
    try:
        from utils.storage import get_trade_store

        return get_trade_store()
    except Exception as e:
        print(f"Warning: Failed to get TradeStore: {e}")
        return None


def _load_from_db() -> Optional[Dict[str, Any]]:
    """DB에서 런타임 설정을 로드합니다."""
    try:
        store = _get_trade_store()
        if store is None:
            return None

        all_configs = store.get_all_runtime_configs()
        if not all_configs:
            return None

        result = {}
        for section, data in all_configs.items():
            config_data = data.get("config_data")
            if config_data:
                try:
                    result[section] = json.loads(config_data)
                except (json.JSONDecodeError, TypeError):
                    pass

        return result if result else None
    except Exception:
        return None


def _save_to_db(config: Mapping[str, Any]) -> bool:
    """런타임 설정을 DB에 일괄 저장합니다."""
    try:
        store = _get_trade_store()
        if store is None:
            return False

        # 모든 섹션을 JSON 문자열로 변환
        configs_json = {}
        for section, data in config.items():
            try:
                configs_json[section] = json.dumps(data, ensure_ascii=False)
            except Exception:
                return False

        # 한 트랜잭션에서 일괄 저장
        return store.save_runtime_configs_bulk(configs_json)
    except Exception:
        return False


def _load_from_json() -> Optional[Dict[str, Any]]:
    """JSON 파일에서 런타임 설정을 로드합니다 (폴백용)."""
    if not RUNTIME_CONFIG_PATH.exists():
        return None

    try:
        data = json.loads(RUNTIME_CONFIG_PATH.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    return None


def load_runtime_config() -> Dict[str, Any]:
    """런타임 설정을 로드하고 기본값과 병합.

    우선순위:
    1. DB에서 로드
    2. JSON 파일에서 로드 (폴백)
    3. 기본값 반환
    """
    defaults = _runtime_defaults()

    # 1. DB에서 로드 시도
    data = _load_from_db()

    # 2. DB에 없으면 JSON 파일에서 로드 (마이그레이션 지원)
    if data is None:
        data = _load_from_json()
        # JSON에서 로드한 경우 DB로 마이그레이션
        if data is not None:
            _migrate_json_to_db(data, defaults)

    # 3. 둘 다 없으면 기본값 반환
    if data is None:
        return defaults

    # 기본값과 병합
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


def _migrate_json_to_db(json_data: Dict[str, Any], defaults: Dict[str, Any]) -> None:
    """JSON 설정을 DB로 마이그레이션합니다."""
    try:
        # 기본값과 병합
        merged = deepcopy(defaults)
        for section, values in json_data.items():
            if isinstance(values, Mapping) and section in merged:
                merged_section = merged[section]
                if isinstance(merged_section, MutableMapping):
                    merged_section.update(values)
                else:
                    merged[section] = values
            else:
                merged[section] = values

        # DB에 저장
        if _save_to_db(merged):
            print("Runtime config migrated from JSON to DB successfully")
    except Exception as e:
        print(f"Warning: Failed to migrate config from JSON to DB: {e}")


def save_runtime_config(config: Mapping[str, Any]) -> None:
    """설정을 DB에 저장. 실패 시 JSON 파일에 폴백 저장."""

    # 1. DB에 저장 시도
    if _save_to_db(config):
        return

    # 2. DB 저장 실패 시 JSON 파일에 폴백
    print("Warning: Failed to save config to DB, falling back to JSON file")
    try:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        with RUNTIME_CONFIG_PATH.open("w", encoding="utf-8") as fp:
            json.dump(dict(config), fp, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Error: Failed to save config to JSON file: {e}")


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
