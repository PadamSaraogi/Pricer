"""
forwards.py
Forward & futures helpers (cost-of-carry).

Conventions:
- r and q in the "flat" functions are **continuously-compounded** (for parity with options tabs).
- Curve-based functions use the repo's YieldCurve (annual-compounded zeros under the hood).
  We consume the curve via its discount factor DF(T). If you want a continuous-rate
  equivalent, use r_cont = -ln(DF(T))/T.

Core:
- forward_price(S0, r, q, T)                        -> F0 = S0 * exp((r - q)*T)
- fair_value_from_forward(F0, r, T)                 -> PV = exp(-r*T) * F0
- ladder_forward_prices(S0, r, q, T, grid)         -> vectorized F(S) across spot grid

Curve-aware:
- forward_price_from_curve(S0, curve, T, q_cont=0) -> F0 = S0 * exp(-q_cont*T) / DF_r(T)
- discount_factor_from_curve(curve, T)             -> DF_r(T)
"""

from __future__ import annotations
import numpy as np
from typing import Optional
from yield_curve import YieldCurve


def forward_price(S0: float, r: float, q: float, T: float) -> float:
    """Theoretical forward/futures price under cost-of-carry with cont. comp."""
    if T < 0:
        raise ValueError("T must be non-negative")
    return float(S0) * float(np.exp((float(r) - float(q)) * float(T)))


def fair_value_from_forward(F0: float, r: float, T: float) -> float:
    """Present value (discounted) of the forward level, continuous comp r."""
    if T < 0:
        raise ValueError("T must be non-negative")
    return float(F0) * float(np.exp(-float(r) * float(T)))


def ladder_forward_prices(S0: float, r: float, q: float, T: float, S_grid: np.ndarray) -> np.ndarray:
    """Vectorized forward prices across a grid of spot values."""
    carry = np.exp((float(r) - float(q)) * float(T))
    return np.asarray(S_grid, dtype=float) * carry


# -------- Curve-aware helpers --------
def discount_factor_from_curve(curve: YieldCurve, T: float) -> float:
    """DF(T) from a repo YieldCurve."""
    return float(curve.get_df(float(T)))


def forward_price_from_curve(S0: float, curve: YieldCurve, T: float, q_cont: float = 0.0) -> float:
    """
    Forward using funding from curve:
        F0 = S0 * exp(-q_cont*T) / DF_r(T)
    where DF_r(T) is funding discount factor from the curve and q_cont is a flat
    **continuous** dividend yield (or convenience yield).
    """
    if T < 0:
        raise ValueError("T must be non-negative")
    df = discount_factor_from_curve(curve, T)
    if df <= 0:
        raise ValueError("Invalid discount factor from curve")
    return float(S0) * float(np.exp(-float(q_cont) * float(T))) / float(df)
