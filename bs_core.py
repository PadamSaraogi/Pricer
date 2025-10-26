from dataclasses import dataclass
import math
from typing import Tuple, Dict, Optional


# ---- Normal pdf/cdf ----
def _phi(x: float) -> float:
  return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _Phi(x: float) -> float:
return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

@dataclass
class OptionInput:
  S0: float
  K: float
  r: float
  sigma: float
  T: float
  q: float = 0.0


def d1_d2(inp: OptionInput) -> Tuple[float, float]:
  S0, K, r, sigma, T, q = inp.S0, inp.K, inp.r, inp.sigma, inp.T, inp.q
  if min(S0, K, sigma, T) <= 0:
    raise ValueError("S0, K, sigma, and T must be positive.")
  d1 = (math.log(S0 / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
  d2 = d1 - sigma * math.sqrt(T)
  return d1, d2

def bs_prices(inp: OptionInput) -> Tuple[float, float, float, float]:
  d1, d2 = d1_d2(inp)
  S0, K, r, T, q = inp.S0, inp.K, inp.r, inp.T, inp.q
  disc_r = math.exp(-r * T)
  disc_q = math.exp(-q * T)
  call = S0 * disc_q * _Phi(d1) - K * disc_r * _Phi(d2)
  put = K * disc_r * _Phi(-d2) - S0 * disc_q * _Phi(-d1)
  return call, put, d1, d2


def bs_greeks(inp: OptionInput) -> Dict:
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
  theta_call = -(S0 * disc_q * nd1 * sigma) / (2.0 * math.sqrt(T)) - r * K * disc_r * Nd2 + q * S0 * disc_q * Nd1
  theta_put = -(S0 * disc_q * nd1 * sigma) / (2.0 * math.sqrt(T)) + r * K * disc_r * _Phi(-d2) - q * S0 * disc_q * _Phi(-d1)
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
    "d1": d1, "d2": d2
  }

# ---- IV Solver (robust bracketed bisection/secant hybrid, no SciPy needed) ----
def implied_vol(target_price: float, inp: OptionInput, opt_type: str = "call",
  lower: float = 1e-6, upper: float = 5.0,
  tol: float = 1e-8, max_iter: int = 100) -> Optional[float]:
  """Return sigma such that BS price == target_price, or None if not found.
  Uses a sign-checked bracket and bisection; falls back to secant steps.
  """
  if target_price <= 0:
    return None

  def f(sig: float) -> float:
    _in = OptionInput(inp.S0, inp.K, inp.r, sig, inp.T, inp.q)
    c, p, *_ = bs_prices(_in)
    return (c if opt_type.lower() == "call" else p) - target_price
  
  a, b = lower, upper
  fa, fb = f(a), f(b)
  
  
  # Try to expand bounds if same sign
  expand_tries = 0
  while fa * fb > 0 and expand_tries < 10:
    a *= 0.5
    b *= 1.5
    fa, fb = f(a), f(b)
    expand_tries += 1
  if fa * fb > 0:
    return None # no bracket
  
  x0, x1 = a, b
  f0, f1 = fa, fb
  for _ in range(max_iter):
    # Secant step
    if abs(f1 - f0) > 1e-14:
      x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
    else:
      x2 = 0.5 * (x0 + x1)
    # If out of bracket, do bisection
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

# ---- Moneyness classification ----
def moneyness_tags(S0: float, K: float, d1: float, threshold: float = 0.01) -> Dict[str, str]:
  ratio = S0 / K
  log_m = math.log(ratio)
  # ATM if within ±1% by default
  if abs(ratio - 1.0) <= threshold:
    tag = "ATM"
  elif ratio > 1.0:
    tag = "ITM (for Call) / OTM (for Put)"
  else:
    tag = "OTM (for Call) / ITM (for Put)"
  return {"S_over_K": f"{ratio:.4f}", "log_moneyness": f"{log_m:.4f}", "d1": f"{d1:.4f}", "tag": tag}
