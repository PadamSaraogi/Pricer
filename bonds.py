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

from daycount import adjust_business_day, BizConv


# -------------------------------------------------------------------------
# Core bond pricing utilities
# -------------------------------------------------------------------------

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


def ytm_from_price(
    price: float,
    face: float,
    coupon_rate: float,
    T: float,
    freq: int = 2,
    guess: float = 0.05,
    tol: float = 1e-6,
    max_iter: int = 100
) -> Optional[float]:
    """
    Calculate Yield to Maturity (YTM) given bond price, coupon rate, and time to maturity.

    Uses Newton-Raphson iteration to solve for YTM.

    Parameters
    ----------
    price : float
        Market price of the bond.
    face : float
        Face value of the bond (typically 100).
    coupon_rate : float
        Annual coupon rate as a decimal (e.g., 0.06 for 6%).
    T : float
        Time to maturity in years.
    freq : int, default=2
        Number of coupon payments per year (1 = annual, 2 = semiannual, 4 = quarterly).
    guess : float
        Initial guess for the YTM (default 5%).
    tol : float
        Convergence tolerance (default 1e-6).
    max_iter : int
        Maximum number of iterations for Newton-Raphson method.

    Returns
    -------
    float or None
        The calculated YTM (as a decimal), or None if it could not be solved.
    """
    c = face * coupon_rate / freq
    n = int(round(T * freq))
    ytm = guess

    for _ in range(max_iter):
        # Price formula for the bond
        bond_price = sum(c / pow(1 + ytm / freq, k) for k in range(1, n + 1)) + face / pow(1 + ytm / freq, n)
        diff = price - bond_price

        if abs(diff) < tol:
            return ytm

        # Derivative of bond price with respect to YTM (using finite difference method)
        d_price = sum(-k * c / freq / pow(1 + ytm / freq, k + 1) for k in range(1, n + 1)) - \
                  n * face / freq / pow(1 + ytm / freq, n + 1)

        # Update YTM using Newton's method
        if d_price == 0:
            break  # Prevent division by zero
        ytm -= diff / d_price

    return None  # If it couldn't converge


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


def macaulay_duration(
    face: float,
    coupon_rate: float,
    ytm: float,
    T: float,
    freq: int = 2
) -> float:
    """
    Calculate the Macaulay duration of a bond given the price, yield, coupon rate, and time to maturity.

    Parameters
    ----------
    face : float
        Face value of the bond (typically 100).
    coupon_rate : float
        Annual coupon rate as a decimal (e.g., 0.06 for 6%).
    ytm : float
        Yield to maturity as a decimal (e.g., 0.06 for 6%).
    T : float
        Time to maturity in years.
    freq : int, default=2
        Number of coupon payments per year (1 = annual, 2 = semiannual, 4 = quarterly).

    Returns
    -------
    float
        The Macaulay duration of the bond in years.
    """
    c = face * coupon_rate / freq  # coupon payment per period
    n = int(round(T * freq))  # total number of coupon periods
    t = [k / freq for k in range(1, n + 1)]  # time periods in years
    discount_factors = [1 / (1 + ytm / freq) ** k for k in range(1, n + 1)]  # DF for each period

    # Calculate the weighted average time to receive each cash flow
    weighted_sum = sum(c * t[i] * discount_factors[i] for i in range(n)) + face * t[-1] * discount_factors[-1]
    price = sum(c * discount_factors[i] for i in range(n)) + face * discount_factors[-1]
    
    return weighted_sum / price


def modified_duration(
    face: float,
    coupon_rate: float,
    ytm: float,
    T: float,
    freq: int = 2
) -> float:
    """
    Calculate the modified duration of a bond given the Macaulay duration and yield.

    Parameters
    ----------
    face : float
        Face value of the bond (typically 100).
    coupon_rate : float
        Annual coupon rate as a decimal (e.g., 0.06 for 6%).
    ytm : float
        Yield to maturity as a decimal (e.g., 0.06 for 6%).
    T : float
        Time to maturity in years.
    freq : int, default=2
        Number of coupon payments per year (1 = annual, 2 = semiannual, 4 = quarterly).

    Returns
    -------
    float
        The modified duration of the bond in years.
    """
    macaulay_dur = macaulay_duration(face, coupon_rate, ytm, T, freq)
    return macaulay_dur / (1 + ytm / freq)


def convexity_numeric(
    face: float,
    coupon_rate: float,
    ytm: float,
    T: float,
    freq: int = 2
) -> float:
    """
    Calculate the bond convexity using numerical methods (second derivative of price).

    Parameters
    ----------
    face : float
        Face value of the bond (typically 100).
    coupon_rate : float
        Annual coupon rate as a decimal (e.g., 0.06 for 6%).
    ytm : float
        Yield to maturity as a decimal (e.g., 0.06 for 6%).
    T : float
        Time to maturity in years.
    freq : int, default=2
        Number of coupon payments per year (1 = annual, 2 = semiannual, 4 = quarterly).

    Returns
    -------
    float
        The bond convexity (in years^2).
    """
    c = face * coupon_rate / freq
    n = int(round(T * freq))
    discount_factors = [1 / (1 + ytm / freq) ** k for k in range(1, n + 1)]

    # First derivative (duration)
    first_derivative = sum(c * (k / freq) * discount_factors[k-1] for k in range(1, n + 1)) + face * (n / freq) * discount_factors[-1]

    # Second derivative (convexity)
    second_derivative = sum(c * (k / freq) ** 2 * discount_factors[k-1] for k in range(1, n + 1)) + face * (n / freq) ** 2 * discount_factors[-1]

    # Convexity is the second derivative of price divided by price
    price = sum(c * discount_factors[k-1] for k in range(1, n + 1)) + face * discount_factors[-1]
    return second_derivative / price


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

