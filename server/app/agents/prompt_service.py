"""프롬프트 관리 서비스.

DB에서 프롬프트를 우선 로드하고, 없으면 기본값을 사용합니다.
"""

from __future__ import annotations

from typing import Dict, Optional

from utils.storage import get_trade_store
from app.services import supabase_repo

from .prompts import (
    INDICATOR_TEMPLATE,
    PATTERN_TEMPLATE,
    TREND_TEMPLATE,
    DECISION_TEMPLATE,
)


# 에이전트 타입별 기본 프롬프트 매핑
DEFAULT_PROMPTS: Dict[str, str] = {
    "indicator": INDICATOR_TEMPLATE,
    "pattern": PATTERN_TEMPLATE,
    "trend": TREND_TEMPLATE,
    "decision": DECISION_TEMPLATE,
}

# 에이전트 타입별 한글 이름
AGENT_TYPE_LABELS: Dict[str, str] = {
    "indicator": "Indicator Agent (기술적 지표 분석)",
    "pattern": "Pattern Agent (차트 패턴 인식)",
    "trend": "Trend Agent (추세/지지저항 분석)",
    "decision": "Decision Agent (최종 매매 결정)",
}


def get_prompt(agent_type: str) -> str:
    """에이전트 프롬프트를 가져옵니다.

    DB에 저장된 프롬프트가 있으면 우선 사용하고, 없으면 기본값을 반환합니다.

    Args:
        agent_type: 에이전트 타입 ('indicator', 'pattern', 'trend', 'decision')

    Returns:
        프롬프트 템플릿 문자열
    """
    # Supabase 우선 조회
    try:
        db_prompt = supabase_repo.get_agent_prompt(agent_type)
        if db_prompt is not None and str(db_prompt).strip():
            return str(db_prompt)
    except Exception as e:
        print(f\"Warning: failed to load prompt from Supabase for {agent_type}: {e}\")

    # DB에서 프롬프트 조회 시도
    try:
        store = get_trade_store()
        db_prompt = store.get_agent_prompt(agent_type)
        if db_prompt is not None and db_prompt.strip():
            return db_prompt
    except Exception as e:
        print(f\"Warning: failed to load prompt from DB for {agent_type}: {e}\")

    # 기본값 반환
    return DEFAULT_PROMPTS.get(agent_type, "")


def get_all_prompts() -> Dict[str, Dict[str, str]]:
    """모든 에이전트 프롬프트를 가져옵니다.

    Returns:
        {
            agent_type: {
                "prompt": str,  # 현재 사용 중인 프롬프트
                "default": str,  # 기본 프롬프트
                "source": str,  # "db" 또는 "default"
                "label": str,  # 한글 라벨
                "updated_at": str | None,  # DB 저장 시간
            },
            ...
        }
    """
    result = {}

    # Supabase -> DB 순서로 조회
    db_prompts = {}
    try:
        db_prompts = supabase_repo.get_all_agent_prompts() or {}
    except Exception as e:
        print(f\"Warning: failed to load prompts from Supabase: {e}\")
    if not db_prompts:
        try:
            store = get_trade_store()
            db_prompts = store.get_all_agent_prompts()
        except Exception as e:
            print(f\"Warning: failed to load prompts from DB: {e}\")

    # 각 에이전트 타입별로 결과 구성
    for agent_type, default_prompt in DEFAULT_PROMPTS.items():
        db_data = db_prompts.get(agent_type, {})
        db_prompt = db_data.get("prompt_template")
        has_db_prompt = db_prompt is not None and db_prompt.strip()

        result[agent_type] = {
            "prompt": db_prompt if has_db_prompt else default_prompt,
            "default": default_prompt,
            "source": "db" if has_db_prompt else "default",
            "label": AGENT_TYPE_LABELS.get(agent_type, agent_type),
            "updated_at": db_data.get("updated_at") if has_db_prompt else None,
        }

    return result


def save_prompt(agent_type: str, prompt_template: str) -> bool:
    """에이전트 프롬프트를 DB에 저장합니다.

    Args:
        agent_type: 에이전트 타입
        prompt_template: 프롬프트 템플릿 텍스트

    Returns:
        성공 여부
    """
    if agent_type not in DEFAULT_PROMPTS:
        print(f"Warning: unknown agent type: {agent_type}")
        return False

    # Supabase 우선 저장
    try:
        if supabase_repo.upsert_agent_prompt(agent_type, prompt_template):
            return True
    except Exception as e:
        print(f\"Error saving prompt to Supabase for {agent_type}: {e}\")

    try:
        store = get_trade_store()
        return store.save_agent_prompt(agent_type, prompt_template)
    except Exception as e:
        print(f\"Error saving prompt for {agent_type}: {e}\")
        return False


def save_prompts_bulk(prompts: Dict[str, str]) -> bool:
    """여러 에이전트 프롬프트를 일괄 저장합니다.

    Args:
        prompts: {agent_type: prompt_template, ...}

    Returns:
        성공 여부
    """
    # 유효한 에이전트 타입만 필터링
    valid_prompts = {
        k: v for k, v in prompts.items()
        if k in DEFAULT_PROMPTS and v is not None
    }

    if not valid_prompts:
        return False

    # Supabase 우선
    try:
        if supabase_repo.upsert_agent_prompts_bulk(valid_prompts):
            return True
    except Exception as e:
        print(f"Error saving prompts bulk to Supabase: {e}")

    try:
        store = get_trade_store()
        return store.save_agent_prompts_bulk(valid_prompts)
    except Exception as e:
        print(f"Error saving prompts bulk: {e}")
        return False


def reset_prompt(agent_type: str) -> bool:
    """에이전트 프롬프트를 기본값으로 초기화합니다 (DB에서 삭제).

    Args:
        agent_type: 에이전트 타입

    Returns:
        성공 여부
    """
    if agent_type not in DEFAULT_PROMPTS:
        print(f"Warning: unknown agent type: {agent_type}")
        return False

    # Supabase 삭제 시도
    try:
        supabase_repo.delete_agent_prompt(agent_type)
    except Exception as e:
        print(f"Warning: failed to delete prompt from Supabase for {agent_type}: {e}")

    try:
        store = get_trade_store()
        return store.delete_agent_prompt(agent_type)
    except Exception as e:
        print(f"Error resetting prompt for {agent_type}: {e}")
        return False


def reset_all_prompts() -> bool:
    """모든 에이전트 프롬프트를 기본값으로 초기화합니다.

    Returns:
        성공 여부
    """
    success = True
    for agent_type in DEFAULT_PROMPTS:
        if not reset_prompt(agent_type):
            success = False
    return success


def get_prompt_variables(agent_type: str) -> list[str]:
    """프롬프트에서 사용 가능한 변수 목록을 반환합니다.

    Args:
        agent_type: 에이전트 타입

    Returns:
        변수 이름 목록 (예: ['symbol', 'regime', ...])
    """
    variables = {
        "indicator": ["symbol", "regime", "position_summary", "indicator_block"],
        "pattern": ["symbol", "regime", "indicator_summary"],
        "trend": ["symbol", "regime", "pattern_summary"],
        "decision": [
            "indicator_summary",
            "pattern_summary",
            "trend_summary",
            "regime",
            "meta_prompt",
        ],
    }
    return variables.get(agent_type, [])

