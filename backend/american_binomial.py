"""
american_binomial.py
Cox–Ross–Rubinstein (CRR) binomial tree pricer for European and American options.
Supports continuous dividend yield q via risk-neutral drift (r - q).
"""

from __future__ import annotations

import math
from typing import Literal

from bs_core import OptionInput


OptionType = Literal["call", "put"]


def _crr_params(inp: OptionInput, steps: int):
    """Compute CRR tree parameters."""
    if steps <= 0:
        raise ValueError("steps must be positive")
    if inp.sigma <= 0 or inp.T <= 0 or inp.S0 <= 0 or inp.K <= 0:
        raise ValueError("All of S0, K, sigma, T must be positive")

    dt = inp.T / steps
    # Standard CRR up/down
    u = math.exp(inp.sigma * math.sqrt(dt))
    d = 1.0 / u
    # Risk-neutral growth with continuous dividend yield
    growth = math.exp((inp.r - inp.q) * dt)
    p = (growth - d) / (u - d)
    disc = math.exp(-inp.r * dt)
    return dt, u, d, p, disc


def _payoff(opt_type: OptionType, S: float, K: float) -> float:
    if opt_type == "call":
        return max(S - K, 0.0)
    else:
        return max(K - S, 0.0)


def crr_price_european(inp: OptionInput, opt_type: OptionType, steps: int = 200) -> float:
    """
    European option price via CRR tree.
    """
    opt_type = opt_type.lower()  # type: ignore
    dt, u, d, p, disc = _crr_params(inp, steps)

    # Guard against pathological probabilities (very small sigma/T)
    if not (0.0 <= p <= 1.0):
        # Fall back: clamp p to [0, 1] to avoid NaNs; note this may bias slightly.
        p = min(max(p, 0.0), 1.0)

    # Terminal stock prices and payoffs
    # S_T(j) = S0 * u^j * d^(N-j), j=0..N
    V = [0.0] * (steps + 1)
    for j in range(steps + 1):
        # compute S at node (j ups, N-j downs)
        S_T = inp.S0 * (u ** j) * (d ** (steps - j))
        V[j] = _payoff(opt_type, S_T, inp.K)

    # Backward induction (European: no early exercise)
    for n in range(steps - 1, -1, -1):
        for j in range(n + 1):
            V[j] = disc * (p * V[j + 1] + (1.0 - p) * V[j])

    return V[0]


def crr_price_american(inp: OptionInput, opt_type: OptionType, steps: int = 200) -> float:
    """
    American option price via CRR tree with early exercise at each node.
    """
    opt_type = opt_type.lower()  # type: ignore
    dt, u, d, p, disc = _crr_params(inp, steps)

    if not (0.0 <= p <= 1.0):
        p = min(max(p, 0.0), 1.0)

    # Terminal payoffs
    V = [0.0] * (steps + 1)
    S_nodes = [0.0] * (steps + 1)
    for j in range(steps + 1):
        S_T = inp.S0 * (u ** j) * (d ** (steps - j))
        S_nodes[j] = S_T
        V[j] = _payoff(opt_type, S_T, inp.K)

    # Backward with early exercise
    for n in range(steps - 1, -1, -1):
        for j in range(n + 1):
            # Stock at node (n, j) = S0 * u^j * d^(n-j)
            S_nj = inp.S0 * (u ** j) * (d ** (n - j))
            cont = disc * (p * V[j + 1] + (1.0 - p) * V[j])
            exer = _payoff(opt_type, S_nj, inp.K)
            V[j] = max(exer, cont)

    return V[0]
