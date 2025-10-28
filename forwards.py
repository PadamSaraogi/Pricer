"""
forwards.py
Forward & futures helpers (cost-of-carry).

Conventions:
- Rates r and q are annual continuously-compounded unless you explicitly
  pass a discount factor yourself (rare). This matches Black–76.

Core:
- forward_price(S0, r, q, T)                 -> F0 = S0 * exp((r - q)*T)
- fair_value_from_forward(F0, r, T)          -> PV = exp(-r*T) * F0   (for convenience)
- ladder_forward_prices(S0, r, q, T, grid)   -> vectorized F(S) across spot grid
"""

from __future__ import annotations
import numpy as np


def forward_price(S0: float, r: float, q: float, T: float) -> float:
    """Theoretical forward/futures price under cost-of-carry with cont. comp."""
    if T < 0:
        raise ValueError("T must be non-negative")
    return float(S0) * float(np.exp((float(r) - float(q)) * float(T)))


def fair_value_from_forward(F0: float, r: float, T: float) -> float:
    """Present value (discounted) of the forward level. Useful for Black–76 checks."""
    if T < 0:
        raise ValueError("T must be non-negative")
    return float(F0) * float(np.exp(-float(r) * float(T)))


def ladder_forward_prices(S0: float, r: float, q: float, T: float, S_grid: np.ndarray) -> np.ndarray:
    """Vectorized forward prices across a grid of spot values."""
    carry = np.exp((float(r) - float(q)) * float(T))
    return np.asarray(S_grid, dtype=float) * carry
