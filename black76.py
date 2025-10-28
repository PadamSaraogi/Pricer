"""
black76.py
Options on futures (Black–76): prices, Greeks, and IV solver.

Inputs:
- F0 : futures/forward price at t=0
- K  : strike
- r  : risk-free (annual, continuously-compounded)
- T  : time to expiry in years (>= 0)
- sigma : volatility (annual)

Formulas:
- d1 = [ln(F0/K) + 0.5*sigma^2*T] / (sigma*sqrt(T))
- d2 = d1 - sigma*sqrt(T)
- Call = D * (F0*N(d1) - K*N(d2)),   D = exp(-r*T)
- Put  = D * (K*N(-d2) - F0*N(-d1))
Greeks (with respect to F, not S): 
- Delta_fut(call) = D * N(d1),  Delta_fut(put) = -D * N(-d1)
- Gamma_fut = D * n(d1) / (F0 * sigma * sqrt(T))
- Vega = D * F0 * n(d1) * sqrt(T)
- Theta, Rho -> computed numerically for robustness unless analytic requested

IV Solver:
- implied_vol_b76(target_px, F0, K, r, T, otype) using a robust bisection/secant hybrid on [1e-6, 5].
"""

from __future__ import annotations
import math
from typing import Literal, Tuple, Optional

OptionType = Literal["call", "put"]

SQRT2PI = math.sqrt(2.0 * math.pi)


def _n(x: float) -> float:
    """Standard normal PDF."""
    return math.exp(-0.5 * x * x) / SQRT2PI


def _N(x: float) -> float:
    """Standard normal CDF (erf-based)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _d1_d2(F0: float, K: float, sigma: float, T: float) -> Tuple[float, float]:
    if T <= 0 or sigma <= 0:
        # limit cases handled by caller; avoid div by zero
        return float("inf"), float("inf")
    vsqrt = sigma * math.sqrt(T)
    d1 = (math.log(F0 / K) + 0.5 * sigma * sigma * T) / vsqrt
    d2 = d1 - vsqrt
    return d1, d2


def black76_price(F0: float, K: float, r: float, T: float, sigma: float, otype: OptionType) -> float:
    """Black–76 price for a call/put on futures."""
    if T <= 0:
        # option expires now -> intrinsic on futures, discounted
        D = math.exp(-r * 0.0)  # 1.0; keep explicit
        intrinsic = max(F0 - K, 0.0) if otype == "call" else max(K - F0, 0.0)
        return D * intrinsic
    if sigma <= 0:
        D = math.exp(-r * T)
        intrinsic = max(F0 - K, 0.0) if otype == "call" else max(K - F0, 0.0)
        return D * intrinsic

    D = math.exp(-r * T)
    d1, d2 = _d1_d2(F0, K, sigma, T)
    if otype == "call":
        return D * (F0 * _N(d1) - K * _N(d2))
    else:
        return D * (K * _N(-d2) - F0 * _N(-d1))


def black76_greeks(F0: float, K: float, r: float, T: float, sigma: float) -> dict:
    """Greeks with respect to the **futures** (not the spot)."""
    D = math.exp(-r * T)
    if T <= 0 or sigma <= 0 or F0 <= 0:
        # handle degenerates numerically to avoid divide-by-zero in gamma
        eps = 1e-6
        base = {t: black76_price(F0, K, r, T, sigma, t) for t in ("call", "put")}
        # numeric vega/theta/rho fallbacks
        vega = (black76_price(F0, K, r, T, sigma + 1e-4, "call") - base["call"]) / 1e-4
        theta = (black76_price(F0, K, r, max(T - 1/365, 0.0), sigma, "call") - base["call"]) / (-1/365)
        rho = (black76_price(F0, K, r + 1e-4, T, sigma, "call") - base["call"]) / 1e-4
        return {
            "delta_fut": {"call": 1.0 if F0 > K else 0.0, "put": - (0.0 if F0 > K else 1.0)},
            "gamma_fut": 0.0,
            "vega": vega,
            "theta_per_day": theta / 365.0,
            "theta_per_year": theta,
            "rho": rho,
            "disc": D,
        }

    d1, d2 = _d1_d2(F0, K, sigma, T)
    Nd1 = _N(d1)
    Nd1m = _N(-d1)
    nd1 = _n(d1)
    vsqrt = sigma * math.sqrt(T)

    delta_c = D * Nd1
    delta_p = -D * Nd1m
    gamma = D * nd1 / (F0 * vsqrt)
    vega = D * F0 * nd1 * math.sqrt(T)

    # numeric theta (per year) and rho (per 1.0 change in r), robust and simple
    h = 1/365.0
    theta_num = (black76_price(F0, K, r, max(T - h, 0.0), sigma, "call") - black76_price(F0, K, r, T, sigma, "call")) / (-h)
    rho_num = (black76_price(F0, K, r + 1e-4, T, sigma, "call") - black76_price(F0, K, r, T, sigma, "call")) / 1e-4

    return {
        "delta_fut": {"call": delta_c, "put": delta_p},
        "gamma_fut": gamma,
        "vega": vega,                       # per 1.0 vol (not %)
        "theta_per_day": theta_num / 365.0,
        "theta_per_year": theta_num,
        "rho": rho_num,
        "disc": D,
        "d1": d1,
        "d2": d2,
    }


def implied_vol_b76(
    target_price: float,
    F0: float,
    K: float,
    r: float,
    T: float,
    otype: OptionType,
    low: float = 1e-6,
    high: float = 5.0,
    tol: float = 1e-8,
    max_iter: int = 100,
) -> Optional[float]:
    """Solve sigma such that Black–76 price = target_price."""
    if target_price <= 0:
        return 0.0 if max(F0 - K, 0.0) == 0 else None

    f_low = black76_price(F0, K, r, T, low, otype) - target_price
    f_high = black76_price(F0, K, r, T, high, otype) - target_price
    # expand if not bracketing
    tries = 0
    while f_low * f_high > 0 and tries < 15:
        high *= 1.5
        f_high = black76_price(F0, K, r, T, high, otype) - target_price
        tries += 1
    if f_low * f_high > 0:
        return None

    x0, x1, f0, f1 = low, high, f_low, f_high
    for _ in range(max_iter):
        # secant update
        if abs(f1 - f0) > 1e-14:
            x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
        else:
            x2 = 0.5 * (x0 + x1)
        # keep in bracket
        if not (min(x0, x1) <= x2 <= max(x0, x1)):
            x2 = 0.5 * (x0 + x1)
        f2 = black76_price(F0, K, r, T, x2, otype) - target_price
        if f0 * f2 <= 0:
            x1, f1 = x2, f2
        else:
            x0, f0 = x2, f2
        if abs(f2) < tol or abs(x1 - x0) < tol:
            return x2
    return None
