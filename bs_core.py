"""
bs_core.py
Core Black–Scholes logic for pricing, Greeks, implied volatility, and moneyness tags.
No external deps beyond the Python standard library.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import math


# -----------------------------
# Normal pdf/cdf
# -----------------------------
def _phi(x: float) -> float:
    """Standard normal PDF."""
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _Phi(x: float) -> float:
    """Standard normal CDF using error function (no SciPy required)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


# -----------------------------
# Inputs container
# -----------------------------
@dataclass
class OptionInput:
    S0: float      # Spot
    K: float       # Strike
    r: float       # Risk-free rate (annual, continuous compounding)
    sigma: float   # Volatility (annualized)
    T: float       # Time to expiry (years)
    q: float = 0.0 # Dividend yield / foreign rate (annual, continuous)


# -----------------------------
# Black–Scholes core
# -----------------------------
def d1_d2(inp: OptionInput) -> Tuple[float, float]:
    """
    Compute d1, d2 for Black–Scholes with continuous dividend yield q.
    """
    S0, K, r, sigma, T, q = inp.S0, inp.K, inp.r, inp.sigma, inp.T, inp.q
    if min(S0, K, sigma, T) <= 0:
        raise ValueError("S0, K, sigma, and T must be positive.")
    d1 = (math.log(S0 / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return d1, d2


def bs_prices(inp: OptionInput) -> Tuple[float, float, float, float]:
    """
    Return (call, put, d1, d2) under Black–Scholes with continuous q.
    """
    d1, d2 = d1_d2(inp)
    S0, K, r, T, q = inp.S0, inp.K, inp.r, inp.T, inp.q
    disc_r = math.exp(-r * T)
    disc_q = math.exp(-q * T)
    call = S0 * disc_q * _Phi(d1) - K * disc_r * _Phi(d2)
    put = K * disc_r * _Phi(-d2) - S0 * disc_q * _Phi(-d1)
    return call, put, d1, d2


def bs_greeks(inp: OptionInput) -> Dict:
    """
    Return a dict of Greeks (call/put) including:
      - delta (call/put)
      - gamma
      - vega_per_1  (per 1.00 = 100% change in sigma)
      - vega_per_1pct (per 0.01 = 1% change in sigma)
      - theta_per_year (call/put)
      - theta_per_day  (call/put)
      - rho (call/put)
      - d1, d2 (for convenience)
    """
    S0, K, r, sigma, T, q = inp.S0, inp.K, inp.r, inp.sigma, inp.T, inp.q
    d1, d2 = d1_d2(inp)
    nd1 = _phi(d1)
    Nd1 = _Phi(d1)
    Nd2 = _Phi(d2)
    disc_r = math.exp(-r * T)
    disc_q = math.exp(-q * T)

    delta_call = disc_q * Nd1
    delta_put = disc_q * (Nd1 - 1.0)
    gamma = disc_q * nd1 / (S0 * sigma * math.sqrt(T))
    vega = S0 * disc_q * nd1 * math.sqrt(T)

    theta_call = (
        -(S0 * disc_q * nd1 * sigma) / (2.0 * math.sqrt(T))
        - r * K * disc_r * Nd2
        + q * S0 * disc_q * Nd1
    )
    theta_put = (
        -(S0 * disc_q * nd1 * sigma) / (2.0 * math.sqrt(T))
        + r * K * disc_r * _Phi(-d2)
        - q * S0 * disc_q * _Phi(-d1)
    )

    rho_call = K * T * disc_r * Nd2
    rho_put = -K * T * disc_r * _Phi(-d2)

    return {
        "delta": {"call": delta_call, "put": delta_put},
        "gamma": gamma,
        "vega_per_1": vega,
        "vega_per_1pct": vega / 100.0,
        "theta_per_year": {"call": theta_call, "put": theta_put},
        "theta_per_day": {"call": theta_call / 365.0, "put": theta_put / 365.0},
        "rho": {"call": rho_call, "put": rho_put},
        "d1": d1,
        "d2": d2,
    }


# -----------------------------
# Implied volatility (robust)
# -----------------------------
def implied_vol(
    target_price: float,
    inp: OptionInput,
    opt_type: str = "call",
    lower: float = 1e-6,
    upper: float = 5.0,
    tol: float = 1e-8,
    max_iter: int = 100,
) -> Optional[float]:
    """
    Solve for sigma such that BS price == target_price using a safe
    bracketed method (secant + bisection). Returns None if no solution found.
    """
    if target_price <= 0:
        return None

    otype = opt_type.lower()

    def f(sig: float) -> float:
        _in = OptionInput(inp.S0, inp.K, inp.r, sig, inp.T, inp.q)
        c, p, *_ = bs_prices(_in)
        return (c if otype == "call" else p) - target_price

    a, b = lower, upper
    fa, fb = f(a), f(b)

    # Try to expand bounds if not bracketing
    expand_tries = 0
    while fa * fb > 0.0 and expand_tries < 10:
        a *= 0.5
        b *= 1.5
        fa, fb = f(a), f(b)
        expand_tries += 1

    if fa * fb > 0.0:
        return None  # No bracket

    x0, x1 = a, b
    f0, f1 = fa, fb
    for _ in range(max_iter):
        # Secant step (fallback to bisection)
        if abs(f1 - f0) > 1e-14:
            x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
        else:
            x2 = 0.5 * (x0 + x1)

        # Ensure we stay within the bracket
        if not (min(x0, x1) <= x2 <= max(x0, x1)):
            x2 = 0.5 * (x0 + x1)

        f2 = f(x2)

        # Update bracket
        if f0 * f2 <= 0:
            x1, f1 = x2, f2
        else:
            x0, f0 = x2, f2

        if abs(f2) < tol or abs(x1 - x0) < tol:
            return max(x2, 0.0)

    return None


# -----------------------------
# Moneyness tag helper
# -----------------------------
def moneyness_tags(S0: float, K: float, d1: float, threshold: float = 0.01) -> Dict[str, str]:
    """
    Basic moneyness labeling using S/K ± threshold for ATM and a helpful summary.
    Returns a dict with S_over_K, log_moneyness, d1, tag.
    """
    ratio = S0 / K
    log_m = math.log(ratio)
    if abs(ratio - 1.0) <= threshold:
        tag = "ATM"
    elif ratio > 1.0:
        tag = "ITM (for Call) / OTM (for Put)"
    else:
        tag = "OTM (for Call) / ITM (for Put)"
    return {
        "S_over_K": f"{ratio:.4f}",
        "log_moneyness": f"{log_m:.4f}",
        "d1": f"{d1:.4f}",
        "tag": tag,
    }


__all__ = [
    "OptionInput",
    "bs_prices",
    "bs_greeks",
    "implied_vol",
    "moneyness_tags",
    "d1_d2",
]
