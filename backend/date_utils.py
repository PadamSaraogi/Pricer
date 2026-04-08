"""
date_utils.py
Lightweight date helpers for fixed income.

- parse_date("YYYY-MM-DD") -> date
- add_months(d, n)         -> date (month roll with end-of-month handling)
- generate_coupon_schedule(issue_date, maturity_date, freq, biz_conv="Following")
  -> list of adjusted coupon dates (excludes issue date, includes maturity)
"""

from __future__ import annotations
from datetime import date, timedelta
from typing import List
from calendar import monthrange

from daycount import adjust_business_day, BizConv


def parse_date(s: str) -> date:
    y, m, d = map(int, s.split("-"))
    return date(y, m, d)


def add_months(d: date, n: int) -> date:
    y = d.year + (d.month - 1 + n) // 12
    m = (d.month - 1 + n) % 12 + 1
    day = min(d.day, monthrange(y, m)[1])
    return date(y, m, day)


def generate_coupon_schedule(
    issue_date: date,
    maturity_date: date,
    freq: int,
    biz_conv: BizConv = "Following",
) -> List[date]:
    """
    Return adjusted coupon dates strictly after issue_date up to and including maturity_date.
    """
    if freq <= 0:
        raise ValueError("freq must be positive")
    step = 12 // freq
    # start from first period after issue
    dates: List[date] = []
    d = issue_date
    while True:
        d = add_months(d, step)
        if d >= maturity_date:
            dates.append(adjust_business_day(maturity_date, biz_conv))
            break
        dates.append(adjust_business_day(d, biz_conv))
    return dates
