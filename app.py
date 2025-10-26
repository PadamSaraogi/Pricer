import math
from dataclasses import dataclass
from typing import Tuple, Dict


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st


# -----------------------------
# Core Black–Scholes functions
# -----------------------------


def _phi(x: float) -> float:
"""Standard normal PDF."""
return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)




def _Phi(x: float) -> float:
"""Standard normal CDF using error function (no external deps)."""
return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))




@dataclass
class OptionInput:
S0: float # Spot price
K: float # Strike
r: float # Risk-free rate (annual, cont. comp.)
sigma: float # Volatility (annualized)
T: float # Time to expiry (years)
q: float = 0.0 # Dividend yield (annual, cont. comp.)




def _d1_d2(inp: OptionInput) -> Tuple[float, float]:
S0, K, r, sigma, T, q = inp.S0, inp.K, inp.r, inp.sigma, inp.T, inp.q
if min(S0, K, sigma, T) <= 0:
raise ValueError("S0, K, sigma, and T must be positive.")
d1 = (math.log(S0 / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
d2 = d1 - sigma * math.sqrt(T)
return d1, d2




def bs_prices(inp: OptionInput) -> Tuple[float, float, float, float]:
d1, d2 = _d1_d2(inp)
S0, K, r, T, q = inp.S0, inp.K, inp.r, inp.T, inp.q
disc_r = math.exp(-r * T)
disc_q = math.exp(-q * T)
call = S0 * disc_q * _Phi(d1) - K * disc_r * _Phi(d2)
put = K * disc_r * _Phi(-d2) - S0 * disc_q * _Phi(-d1)
return call, put, d1, d2




def bs_greeks(inp: OptionInput) -> Dict:
S0, K, r, sigma, T, q = inp.S0, inp.K, inp.r, inp.sigma, inp.T, inp.q
d1, d2 = _d1_d2(inp)
nd1 = _phi(d1)
Nd1 = _Phi(d1)
Nd2 = _Phi(d2)


disc_r = math.exp(-r * T)
