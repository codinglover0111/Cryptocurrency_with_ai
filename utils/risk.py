from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class RiskConfig:
    risk_percent: float = 1.0  # 계정 대비 1% 리스크
    max_allocation_percent: float = 20.0  # 계정 대비 포지션 최대 노출(레버리지 포함)
    default_leverage: float = 5.0
    min_quantity: float = 1.0


def calculate_position_size(
    *,
    balance_usdt: float,
    entry_price: float,
    stop_price: float,
    risk_percent: float,
    max_allocation_percent: float,
    leverage: float,
    min_quantity: float = 1.0,
) -> float:
    """리스크 기반 포지션 사이즈 계산(선물/USDT 선물 가정)

    - 위험금액 = balance * (risk_percent/100)
    - 1단위 손실 = |entry - stop|
    - 기본 수량 = 위험금액 / (1단위 손실)
    - 노출 상한 = balance * (max_allocation_percent/100) * leverage
      → 수량*진입가가 노출 상한을 넘으면 축소
    """
    if balance_usdt is None or balance_usdt <= 0:
        return min_quantity
    price_distance = abs(entry_price - stop_price)
    if price_distance <= 0:
        return min_quantity

    risk_amount = balance_usdt * (risk_percent / 100.0)
    base_quantity = risk_amount / price_distance

    # 레버리지 포함 노출 상한
    max_notional = balance_usdt * (max_allocation_percent / 100.0) * max(1.0, leverage)
    notional = base_quantity * entry_price
    if notional > max_notional and entry_price > 0:
        scale = max_notional / notional
        base_quantity *= max(0.0, scale)

    # 최소 수량 보정
    quantity = max(min_quantity, base_quantity)
    return float(quantity)


def enforce_max_loss_sl(
    *,
    entry_price: float,
    proposed_sl: Optional[float],
    position: str,
    max_loss_percent: float = 80.0,
) -> Optional[float]:
    """제안된 SL이 최대 손실 한도(%)를 초과할 경우 강제로 제한된 SL을 반환.

    - position: "long" 또는 "short" (AI Status 기준)
    - long: 손실% = (entry - sl)/entry*100 → sl는 최소 entry*(1 - max_loss%) 이상
    - short: 손실% = (sl - entry)/entry*100 → sl는 최대 entry*(1 + max_loss%) 이하

    proposed_sl 가 None 이면 그대로 None 반환.
    entry_price 가 비정상(<=0)이면 proposed_sl 그대로 반환.
    """
    try:
        if proposed_sl is None:
            return None
        e = float(entry_price)
        if e <= 0:
            return float(proposed_sl)
        sl = float(proposed_sl)
        pos = (position or "").lower()
        limit = max(0.0, float(max_loss_percent)) / 100.0
        if pos == "long":
            min_allowed = e * (1.0 - limit)
            return float(sl) if sl >= min_allowed else float(min_allowed)
        elif pos == "short":
            max_allowed = e * (1.0 + limit)
            return float(sl) if sl <= max_allowed else float(max_allowed)
        else:
            return float(sl)
    except Exception:
        return proposed_sl
