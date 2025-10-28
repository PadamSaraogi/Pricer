"""
black76.py
Options on futures (Black–76): prices, Greeks, and IV solver.
Now supports **DF-aware** pricing so you can discount off a bootstrapped curve.

Flat-r interface (legacy):
- black76_price(F0, K, r, T, sigma, otype)

DF-aware interface (preferred with curves):
- black76_price_df(F0, K, DF_T, T, sigma, otype)
- black76_greeks_df(F0, K, DF_T, T, sigma)   # Greeks w.r.t. futures; D = DF_T
- implied_vol_b76(..., DF_T=...)  # if DF_T is supplied, r is ignored
"""

from __future__ import annotations
import math
from typing import Literal, Tuple, Optional

OptionType = Literal["call", "put"]
SQRT2 = math.sqrt(2.0)
SQRT2PI = math.sqrt(2.0 * math.pi)


def _n(x: float) -> float:
    return math.exp(-0.5 * x * x) / SQRT2PI


def _N(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / SQRT2))


def _d1_d2(F0: float, K: float, sigma: float, T: float) -> Tuple[float, float]:
    if T <= 0 or sigma <= 0:
        return float("inf"), float("inf")
    vsqrt = sigma * math.sqrt(T)
    d1 = (math.log(F0 / K) + 0.5 * sigma * sigma * T) / vsqrt
    d2 = d1 - vsqrt
    return d1, d2


# ---------- Flat-r (legacy) ----------
def black76_price(F0: float, K: float, r: float, T: float, sigma: float, otype: OptionType) -> float:
    if T <= 0 or sigma <= 0:
        D = math.exp(-r * max(T, 0.0))
        intrinsic = max(F0 - K, 0.0) if otype == "call" else max(K - F0, 0.0)
        return D * intrinsic
    D = math.exp(-r * T)
    d1, d2 = _d1_d2(F0, K, sigma, T)
    if otype == "call":
        return D * (F0 * _N(d1) - K * _N(d2))
    else:
        return D * (K * _N(-d2) - F0 * _N(-d1))


# ---------- DF-aware (curve) ----------
def black76_price_df(F0: float, K: float, DF_T: float, T: float, sigma: float, otype: OptionType) -> float:
    D = float(DF_T)
    if T <= 0 or sigma <= 0:
        intrinsic = max(F0 - K, 0.0) if otype == "call" else max(K - F0, 0.0)
        return D * intrinsic
    d1, d2 = _d1_d2(F0, K, sigma, T)
    if otype == "call":
        return D * (F0 * _N(d1) - K * _N(d2))
    else:
        return D * (K * _N(-d2) - F0 * _N(-d1))


def black76_greeks_df(F0: float, K: float, DF_T: float, T: float, sigma: float) -> dict:
    D = float(DF_T)
    if T <= 0 or sigma <= 0 or F0 <= 0:
        # Numeric fallbacks
        base = {t: black76_price_df(F0, K, D, T, sigma, t) for t in ("call", "put")}
        vega = (black76_price_df(F0, K, D, T, sigma + 1e-4, "call") - base["call"]) / 1e-4
        theta = (black76_price_df(F0, K, D, max(T - 1/365, 0.0), sigma, "call") - base["call"]) / (-1/365)
        rhoD = (black76_price_df(F0, K, D * 1.0001, T, sigma, "call") - base["call"]) / (D * 1e-4)  # sensitivity to DF
        return {
            "delta_fut": {"call": 1.0 if F0 > K else 0.0, "put": - (0.0 if F0 > K else 1.0)},
            "gamma_fut": 0.0,
            "vega": vega,
            "theta_per_day": theta / 365.0,
            "theta_per_year": theta,
            "rho_df": rhoD,
            "disc": D,
        }

    d1, d2 = _d1_d2(F0, K, sigma, T)
    Nd1 = _N(d1); Nd1m = _N(-d1); nd1 = _n(d1)
    vsqrt = sigma * math.sqrt(T)

    delta_c = D * Nd1
    delta_p = -D * Nd1m
    gamma = D * nd1 / (F0 * vsqrt)
    vega = D * F0 * nd1 * math.sqrt(T)

    # numeric theta and DF sensitivity (rho_df)
    h = 1/365.0
    theta_num = (black76_price_df(F0, K, D, max(T - h, 0.0), sigma, "call") - black76_price_df(F0, K, D, T, sigma, "call")) / (-h)
    rho_df = (black76_price_df(F0, K, D * 1.0001, T, sigma, "call") - black76_price_df(F0, K, D, T, sigma, "call")) / (D * 1e-4)

    return {
        "delta_fut": {"call": delta_c, "put": delta_p},
        "gamma_fut": gamma,
        "vega": vega,
        "theta_per_day": theta_num / 365.0,
        "theta_per_year": theta_num,
        "rho_df": rho_df,   # sensitivity to DF(T)
        "disc": D,
        "d1": d1, "d2": d2,
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
    DF_T: Optional[float] = None,
) -> Optional[float]:
    """
    Solve sigma such that price = target_price.
    If DF_T is provided, r is ignored and DF-aware pricing is used.
    """
    if target_price <= 0:
        intrinsic = max(F0 - K, 0.0) if otype == "call" else max(K - F0, 0.0)
        return 0.0 if intrinsic == 0 else None

    def px(sig: float) -> float:
        if DF_T is None:
            return black76_price(F0, K, r, T, sig, otype)
        else:
            return black76_price_df(F0, K, DF_T, T, sig, otype)

    f_low = px(low) - target_price
    f_high = px(high) - target_price

    tries = 0
    while f_low * f_high > 0 and tries < 15:
        high *= 1.5
        f_high = px(high) - target_price
        tries += 1
    if f_low * f_high > 0:
        return None

    x0, x1, f0, f1 = low, high, f_low, f_high
    for _ in range(max_iter):
        x2 = x1 - f1 * (x1 - x0) / (f1 - f0) if abs(f1 - f0) > 1e-14 else 0.5 * (x0 + x1)
        if not (min(x0, x1) <= x2 <= max(x0, x1)):
            x2 = 0.5 * (x0 + x1)
        f2 = px(x2) - target_price
        if f0 * f2 <= 0:
            x1, f1 = x2, f2
        else:
            x0, f0 = x2, f2
        if abs(f2) < tol or abs(x1 - x0) < tol:
            return x2
    return None
