"""
bonds.py
Bond pricing with both numeric-T and date-aware APIs.

Numeric API (legacy):
- price_bond(face, coupon_rate, ytm, T_years, freq)
- ytm_from_price(price_target, face, coupon_rate, T_years, freq)
- macaulay_duration(...), modified_duration(...), convexity_numeric(...)

Date-aware API (new):
- price_bond_dates(face, coupon_rate, ytm, settlement_date, maturity_date, freq, day_count="ACT/365F", biz_conv="Following")
- ytm_from_price_dates(price_target, face, coupon_rate, settlement_date, maturity_date, freq, day_count="ACT/365F", biz_conv="Following")
- durations via dates: macaulay_duration_dates(...), modified_duration_dates(...)
"""

from __future__ import annotations
from typing import List, Tuple, Optional
from datetime import date

from daycount import year_fraction, DayCount, BizConv
from date_utils import generate_coupon_schedule

# ---------------------------
# Numeric (legacy) functions
# ---------------------------
def _n_periods(T_years: float, freq: int) -> int:
    if T_years <= 0 or freq <= 0:
        raise ValueError("T_years and freq must be positive")
    return int(round(T_years * freq))


def price_bond(face: float, coupon_rate: float, ytm: float, T_years: float, freq: int = 2) -> float:
    n = _n_periods(T_years, freq)
    c = face * coupon_rate / freq
    y_per = ytm / freq
    pv = 0.0
    disc = 1.0 + y_per
    for k in range(1, n + 1):
        pv += c / (disc ** k)
    pv += face / (disc ** n)
    return pv


def ytm_from_price(
    price_target: float,
    face: float,
    coupon_rate: float,
    T_years: float,
    freq: int = 2,
    lower: float = -0.99,
    upper: float = 1.00,
    tol: float = 1e-8,
    max_iter: int = 200,
) -> Optional[float]:
    if price_target <= 0:
        return None

    def f(y: float) -> float:
        return price_bond(face, coupon_rate, y, T_years, freq) - price_target

    a, b = lower, upper
    fa, fb = f(a), f(b)
    tries = 0
    while fa * fb > 0 and tries < 20:
        a = a - 0.5 * (b - a)
        b = b + 0.5 * (b - a)
        fa, fb = f(a), f(b)
        tries += 1
    if fa * fb > 0:
        return None

    x0, x1, f0, f1 = a, b, fa, fb
    for _ in range(max_iter):
        x2 = x1 - f1 * (x1 - x0) / (f1 - f0) if abs(f1 - f0) > 1e-14 else 0.5 * (x0 + x1)
        if not (min(x0, x1) <= x2 <= max(x0, x1)):
            x2 = 0.5 * (x0 + x1)
        f2 = f(x2)
        if f0 * f2 <= 0:
            x1, f1 = x2, f2
        else:
            x0, f0 = x2, f2
        if abs(f2) < tol or abs(x1 - x0) < tol:
            return x2
    return None


def macaulay_duration(face: float, coupon_rate: float, ytm: float, T_years: float, freq: int = 2) -> float:
    n = _n_periods(T_years, freq)
    c = face * coupon_rate / freq
    y_per = ytm / freq
    disc = 1.0 + y_per
    price = price_bond(face, coupon_rate, ytm, T_years, freq)
    acc = 0.0
    for k in range(1, n + 1):
        t = k / freq
        cf = c if k < n else (c + face)
        acc += t * cf / (disc ** k)
    return acc / price


def modified_duration(face: float, coupon_rate: float, ytm: float, T_years: float, freq: int = 2) -> float:
    mac = macaulay_duration(face, coupon_rate, ytm, T_years, freq)
    return mac / (1.0 + ytm / freq)


def convexity_numeric(face: float, coupon_rate: float, ytm: float, T_years: float, freq: int = 2, bump: float = 1e-4) -> float:
    p0 = price_bond(face, coupon_rate, ytm, T_years, freq)
    p_up = price_bond(face, coupon_rate, ytm + bump, T_years, freq)
    p_dn = price_bond(face, coupon_rate, ytm - bump, T_years, freq)
    return (p_up + p_dn - 2.0 * p0) / (p0 * (bump ** 2))


# ---------------------------
# Date-aware functions
# ---------------------------
def _cashflow_times_from_dates(
    settlement: date,
    maturity: date,
    freq: int,
    day_count: DayCount,
    biz_conv: BizConv,
    issue_date: Optional[date] = None,
) -> list[float]:
    """
    Compute year fractions from settlement to each future coupon date (including maturity).
    If issue_date provided, schedule starts from issue_date; otherwise approximate backwards from maturity.
    """
    if issue_date is None:
        # approximate last X coupons by stepping backwards until < settlement
        from date_utils import add_months, adjust_business_day  # local import to avoid cycles
        step = 12 // freq
        dates = []
        d = maturity
        while True:
            d_adj = adjust_business_day(d, biz_conv)
            if d_adj > settlement:
                dates.append(d_adj)
            d = add_months(d, -step)
            if d <= settlement or len(dates) > 1000:
                break
        dates.sort()
    else:
        dates = generate_coupon_schedule(issue_date, maturity, freq, biz_conv)
        dates = [d for d in dates if d > settlement]

    return [year_fraction(settlement, d, day_count) for d in dates]


def price_bond_dates(
    face: float,
    coupon_rate: float,
    ytm: float,
    settlement_date: date,
    maturity_date: date,
    freq: int = 2,
    day_count: DayCount = "ACT/365F",
    biz_conv: BizConv = "Following",
    issue_date: Optional[date] = None,
) -> float:
    c = face * coupon_rate / freq
    times = _cashflow_times_from_dates(settlement_date, maturity_date, freq, day_count, biz_conv, issue_date)
    if not times:
        return 0.0
    n = len(times)
    pv = 0.0
    for i, t in enumerate(times, start=1):
        df = 1.0 / (1.0 + ytm / freq) ** (t * freq)
        cf = c if i < n else (c + face)
        pv += cf * df
    return pv


def ytm_from_price_dates(
    price_target: float,
    face: float,
    coupon_rate: float,
    settlement_date: date,
    maturity_date: date,
    freq: int = 2,
    day_count: DayCount = "ACT/365F",
    biz_conv: BizConv = "Following",
    issue_date: Optional[date] = None,
    lower: float = -0.99,
    upper: float = 1.00,
    tol: float = 1e-8,
    max_iter: int = 200,
) -> Optional[float]:
    if price_target <= 0:
        return None

    def f(y: float) -> float:
        return price_bond_dates(face, coupon_rate, y, settlement_date, maturity_date, freq, day_count, biz_conv, issue_date) - price_target

    a, b = lower, upper
    fa, fb = f(a), f(b)
    tries = 0
    while fa * fb > 0 and tries < 20:
        a = a - 0.5 * (b - a)
        b = b + 0.5 * (b - a)
        fa, fb = f(a), f(b)
        tries += 1
    if fa * fb > 0:
        return None

    x0, x1, f0, f1 = a, b, fa, fb
    for _ in range(max_iter):
        x2 = x1 - f1 * (x1 - x0) / (f1 - f0) if abs(f1 - f0) > 1e-14 else 0.5 * (x0 + x1)
        if not (min(x0, x1) <= x2 <= max(x0, x1)):
            x2 = 0.5 * (x0 + x1)
        f2 = f(x2)
        if f0 * f2 <= 0:
            x1, f1 = x2, f2
        else:
            x0, f0 = x2, f2
        if abs(f2) < tol or abs(x1 - x0) < tol:
            return x2
    return None


def macaulay_duration_dates(
    face: float,
    coupon_rate: float,
    ytm: float,
    settlement_date: date,
    maturity_date: date,
    freq: int = 2,
    day_count: DayCount = "ACT/365F",
    biz_conv: BizConv = "Following",
    issue_date: Optional[date] = None,
) -> float:
    c = face * coupon_rate / freq
    times = _cashflow_times_from_dates(settlement_date, maturity_date, freq, day_count, biz_conv, issue_date)
    if not times:
        return 0.0
    n = len(times)
    price = price_bond_dates(face, coupon_rate, ytm, settlement_date, maturity_date, freq, day_count, biz_conv, issue_date)
    acc = 0.0
    for i, t in enumerate(times, start=1):
        df = 1.0 / (1.0 + ytm / freq) ** (t * freq)
        cf = c if i < n else (c + face)
        acc += t * cf * df
    return acc / price if price > 0 else 0.0


def modified_duration_dates(
    face: float,
    coupon_rate: float,
    ytm: float,
    settlement_date: date,
    maturity_date: date,
    freq: int = 2,
    day_count: DayCount = "ACT/365F",
    biz_conv: BizConv = "Following",
    issue_date: Optional[date] = None,
) -> float:
    mac = macaulay_duration_dates(face, coupon_rate, ytm, settlement_date, maturity_date, freq, day_count, biz_conv, issue_date)
    return mac / (1.0 + ytm / freq)
