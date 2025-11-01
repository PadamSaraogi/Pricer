"""
bonds.py
----------
Fixed-income bond pricing and yield calculations with flexible coupon frequencies
(annual, semiannual, quarterly, monthly).

Supports:
- Clean and dirty price calculation
- Yield-to-Maturity (YTM) solving
- Accrued interest computation
- Full and partial coupon periods
- Integration with yield_curve.py for bootstrapping

Usage examples:
---------------
>>> price_bond(face=100, coupon_rate=0.06, ytm=0.065, T=5, freq=2)
>>> yield_to_maturity(price=98.5, face=100, coupon_rate=0.07, T=10, freq=12)
"""

from __future__ import annotations
from math import pow
from typing import Optional
from datetime import date

# ---------------------------------------------------------------------
# Core bond pricing utilities
# ---------------------------------------------------------------------

def price_bond(
    face: float,
    coupon_rate: float,
    ytm: float,
    T: float,
    freq: int = 2,
    *,
    clean_price: bool = True,
    accrual: float = 0.0
) -> float:
    """
    Compute bond price given yield, coupon rate, and time to maturity.

    Parameters
    ----------
    face : float
        Face value of the bond (usually 100).
    coupon_rate : float
        Annual coupon rate (as decimal, e.g., 0.07 for 7%).
    ytm : float
        Annual yield-to-maturity (as decimal).
    T : float
        Time to maturity in years.
    freq : int, default=2
        Coupon payments per year (1=annual, 2=semiannual, 4=quarterly, 12=monthly).
    clean_price : bool, default=True
        If True, return clean price (ex-accrued). If False, return dirty price.
    accrual : float, default=0.0
        Accrued interest (if already known), otherwise assumed 0.

    Returns
    -------
    float
        Price of the bond (clean or dirty depending on flag).
    """
    if T <= 0:
        return face

    c = face * coupon_rate / freq
    n = int(round(T * freq))

    pv_coupons = sum(c / pow(1 + ytm / freq, k) for k in range(1, n + 1))
    pv_face = face / pow(1 + ytm / freq, n)
    dirty = pv_coupons + pv_face

    if clean_price:
        return dirty - accrual
    return dirty


def yield_to_maturity(
    price: float,
    face: float,
    coupon_rate: float,
    T: float,
    freq: int = 2,
    guess: float = 0.05,
    tol: float = 1e-8,
    max_iter: int = 100
) -> float:
    """
    Compute YTM given price and coupon structure via Newton–Raphson iteration.

    Parameters
    ----------
    price : float
        Observed market price of the bond.
    face : float
        Face value (usually 100).
    coupon_rate : float
        Annual coupon rate (decimal).
    T : float
        Time to maturity (years).
    freq : int, default=2
        Coupon frequency per year (1, 2, 4, 12 supported).
    guess : float
        Initial YTM guess (decimal).
    tol : float
        Convergence tolerance.
    max_iter : int
        Maximum iterations.

    Returns
    -------
    float
        Yield-to-maturity as a decimal.
    """
    if T <= 0:
        return 0.0

    c = face * coupon_rate / freq
    n = int(round(T * freq))
    ytm = guess

    for _ in range(max_iter):
        pv = sum(c / pow(1 + ytm / freq, k) for k in range(1, n + 1)) + face / pow(1 + ytm / freq, n)
        diff = price - pv
        if abs(diff) < tol:
            return ytm
        # derivative wrt ytm
        d_pv = sum(-k * c / freq / pow(1 + ytm / freq, k + 1) for k in range(1, n + 1)) - \
               n * face / freq / pow(1 + ytm / freq, n + 1)
        ytm -= diff / d_pv if d_pv != 0 else 0

    return ytm


def accrued_interest(
    face: float,
    coupon_rate: float,
    freq: int,
    last_coupon_date: date,
    settlement_date: date,
    next_coupon_date: date,
    day_count_basis: str = "ACT/365F"
) -> float:
    """
    Compute accrued interest between last coupon and settlement date.

    Parameters
    ----------
    face : float
        Face value.
    coupon_rate : float
        Annual coupon rate (decimal).
    freq : int
        Coupon frequency (1, 2, 4, 12).
    last_coupon_date : date
        Last coupon payment date.
    settlement_date : date
        Current date or settlement date.
    next_coupon_date : date
        Next coupon payment date.
    day_count_basis : str
        Day count convention (e.g. "ACT/365F", "30/360").

    Returns
    -------
    float
        Accrued interest.
    """
    from daycount import year_fraction
    yf_total = year_fraction(last_coupon_date, next_coupon_date, day_count_basis)
    yf_elapsed = year_fraction(last_coupon_date, settlement_date, day_count_basis)
    coupon_amt = face * coupon_rate / freq
    return coupon_amt * (yf_elapsed / yf_total)


def price_bond_dates(
    face: float,
    coupon_rate: float,
    ytm: float,
    issue_date: date,
    maturity_date: date,
    freq: int = 2,
    valuation_date: Optional[date] = None,
    day_count_basis: str = "ACT/365F"
) -> float:
    """
    Price a bond given actual dates (issue, maturity, valuation).

    Uses date-based schedule via date_utils.generate_coupon_schedule().

    Parameters
    ----------
    face : float
        Face value.
    coupon_rate : float
        Annual coupon rate (decimal).
    ytm : float
        Annual YTM (decimal).
    issue_date : date
        Bond issue/start date.
    maturity_date : date
        Bond maturity date.
    freq : int
        Coupon frequency per year (1, 2, 4, 12).
    valuation_date : date
        Valuation/settlement date.
    day_count_basis : str
        Day-count convention.

    Returns
    -------
    float
        Bond price (clean).
    """
    from date_utils import generate_coupon_schedule
    from daycount import year_fraction

    if valuation_date is None:
        valuation_date = issue_date

    schedule = generate_coupon_schedule(issue_date, maturity_date, freq)
    future_coupons = [d for d in schedule if d > valuation_date]
    if not future_coupons:
        return face

    price = 0.0
    for d in future_coupons:
        t = year_fraction(valuation_date, d, day_count_basis)
        c = face * coupon_rate / freq
        price += c / pow(1 + ytm / freq, t * freq)
    price += face / pow(1 + ytm / freq, future_coupons and (t * freq) or 1)
    return price
