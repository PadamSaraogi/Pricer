"""
swaps.py
Plain-vanilla fixed-for-floating interest rate swap utilities using a YieldCurve.

Assumptions:
- No notional exchange
- Fixed leg pays s/freq at times t_i (i=1..N)
- Float leg PV for a par-start swap (reset at 0) is N * (1 - DF(T_N))
- Day count via year fractions implied by payment frequency (1/freq)

Functions:
- swap_annuity(curve, mats, freq) -> sum DF(t_i)/freq
- par_swap_rate(curve, T, freq) -> s* such that PV=0
- swap_pv(curve, T, fixed_rate, notional, freq) -> PV (fixed - float)
- dv01(curve, T, fixed_rate, notional, freq) -> bump-1bp PV sensitivity
"""

from __future__ import annotations
from typing import List
from yield_curve import YieldCurve


def _payment_schedule(T: float, freq: int) -> List[float]:
    n = int(round(T * freq))
    return [i / freq for i in range(1, n + 1)]


def swap_annuity(curve: YieldCurve, T: float, freq: int) -> float:
    """Annuity = sum DF(t_i)/freq over fixed coupons."""
    times = _payment_schedule(T, freq)
    return sum(curve.get_df(t) for t in times) / freq


def par_swap_rate(curve: YieldCurve, T: float, freq: int) -> float:
    """Par fixed rate s* that sets PV to zero."""
    times = _payment_schedule(T, freq)
    df_T = curve.get_df(T)
    denom = sum(curve.get_df(t) for t in times) / freq
    if denom <= 0:
        return 0.0
    # For par-start float leg: PV_float = 1 - DF(T)
    return (1.0 - df_T) / denom


def swap_pv(curve: YieldCurve, T: float, fixed_rate: float, notional: float, freq: int) -> float:
    """PV (Fixed - Float), using no-notional-exchange formulation."""
    times = _payment_schedule(T, freq)
    ann = sum(curve.get_df(t) for t in times) / freq
    pv_fixed = notional * fixed_rate * ann
    pv_float = notional * (1.0 - curve.get_df(T))
    return pv_fixed - pv_float


def dv01(curve: YieldCurve, T: float, fixed_rate: float, notional: float, freq: int, bp: float = 1.0) -> float:
    """Numerical DV01: PV shift for +bp parallel bump (1bp=0.0001)."""
    base = swap_pv(curve, T, fixed_rate, notional, freq)
    bumped = swap_pv(curve.bumped(bp), T, fixed_rate, notional, freq)
    # Return per 1bp
    return (bumped - base)
