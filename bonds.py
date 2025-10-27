# bonds.py
"""
Bond pricing utilities: clean price, yield-to-maturity (YTM), duration & convexity.
- Discrete compounding with payment frequency `freq` per year (1, 2, 4, etc.).
- Duration (Macaulay & modified) analytic; convexity via robust finite-difference.
"""
from __future__ import annotations
import math


def _n_periods(T_years: float, freq: int) -> int:
    if T_years <= 0 or freq <= 0:
        raise ValueError("T_years and freq must be positive")
    return int(round(T_years * freq))


def price_bond(face: float, coupon_rate: float, ytm: float, T_years: float, freq: int = 2) -> float:
    """
    Price a fixed-coupon bond.
    face: redemption value (e.g., 100 or 1000)
    coupon_rate: annual coupon rate (e.g., 0.06 for 6%)
    ytm: annual yield to maturity (nominal, compounded `freq` times)
    T_years: time to maturity in years
    freq: coupon payments per year
    """
    n = _n_periods(T_years, freq)
    c = face * coupon_rate / freq
    y_per = ytm / freq
    disc = 1.0 + y_per
    pv = 0.0
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
) -> float | None:
    """
    Solve nominal annual YTM such that model price == price_target using a bisection–secant hybrid.
    Allows negative yields down to -99% and up to 100%.
    Returns None if no root found in the bracket.
    """
    if price_target <= 0:
        return None

    def f(y: float) -> float:
        return price_bond(face, coupon_rate, y, T_years, freq) - price_target

    a, b = lower, upper
    fa, fb = f(a), f(b)

    # Expand bracket if needed
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
        # Secant step
        if abs(f1 - f0) > 1e-14:
            x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
        else:
            x2 = 0.5 * (x0 + x1)
        # Keep inside bracket
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
    """Macaulay duration (in years)."""
    n = _n_periods(T_years, freq)
    c = face * coupon_rate / freq
    y_per = ytm / freq
    disc = 1.0 + y_per
    price = price_bond(face, coupon_rate, ytm, T_years, freq)
    acc = 0.0
    for k in range(1, n + 1):
        t_years = k / freq
        cf = c if k < n else (c + face)
        acc += t_years * cf / (disc ** k)
    return acc / price


def modified_duration(face: float, coupon_rate: float, ytm: float, T_years: float, freq: int = 2) -> float:
    """Modified duration (in years)."""
    mac = macaulay_duration(face, coupon_rate, ytm, T_years, freq)
    return mac / (1.0 + ytm / freq)


def convexity_numeric(
    face: float,
    coupon_rate: float,
    ytm: float,
    T_years: float,
    freq: int = 2,
    bump: float = 1e-4,
) -> float:
    """
    Numerical convexity using central difference on yield.
    bump: yield bump in absolute terms (1e-4 = 1bp).
    Returns convexity in 1/(yield^2) units.
    """
    p0 = price_bond(face, coupon_rate, ytm, T_years, freq)
    p_up = price_bond(face, coupon_rate, ytm + bump, T_years, freq)
    p_dn = price_bond(face, coupon_rate, ytm - bump, T_years, freq)
    return (p_up + p_dn - 2.0 * p0) / (p0 * (bump ** 2))
