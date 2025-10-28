"""
dividends.py
Utilities for discrete **cash** dividends.

We use a simple present value approach:
  S_eff = S0 - sum_i PV(D_i),  where PV(D_i) = D_i * exp(-r * T_i)

Assumptions:
- r is **continuous-compounded** (consistent with your options tabs).
- T_i computed with a chosen day-count (e.g., ACT/365F) from valuation_date to each dividend_date.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, List, Tuple
import math
from datetime import date

from daycount import year_fraction, DayCount


@dataclass(frozen=True)
class CashDividend:
    pay_date: date
    amount: float  # in same currency as S0


def pv_cash_dividends(
    dividends: Iterable[CashDividend],
    r_cont: float,
    valuation_date: date,
    dc: DayCount = "ACT/365F",
) -> float:
    """Present value (sum) of cash dividends using continuous comp r."""
    pv = 0.0
    for d in dividends:
        if d.amount <= 0:
            continue
        # If dividend already on/before valuation_date, ignore it
        if d.pay_date <= valuation_date:
            continue
        T = year_fraction(valuation_date, d.pay_date, dc)
        if T <= 0:
            continue
        pv += d.amount * math.exp(-float(r_cont) * float(T))
    return pv


def spot_adjusted_for_dividends(
    S0: float,
    dividends: Iterable[CashDividend],
    r_cont: float,
    valuation_date: date,
    dc: DayCount = "ACT/365F",
) -> float:
    """Return S_eff = S0 - PV(dividends). Never below tiny positive floor."""
    pv = pv_cash_dividends(dividends, r_cont, valuation_date, dc)
    S_eff = float(S0) - float(pv)
    # Prevent non-positive effective spot in corner cases
    return max(S_eff, 1e-9)
