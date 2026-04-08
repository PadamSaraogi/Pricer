# tests/test_bs_core.py
import math

from bs_core import OptionInput, bs_prices, bs_greeks, implied_vol


def approx(a, b, tol=1e-6):
    return abs(a - b) < tol


def test_put_call_parity():
    inp = OptionInput(100, 105, 0.05, 0.20, 0.5, 0.0)
    c, p, *_ = bs_prices(inp)
    lhs = c - p
    rhs = inp.S0 * math.exp(-inp.q * inp.T) - inp.K * math.exp(-inp.r * inp.T)
    assert approx(lhs, rhs, 1e-6)


def test_greeks_signs_and_ranges():
    inp = OptionInput(100, 105, 0.05, 0.20, 0.5, 0.0)
    G = bs_greeks(inp)

    # Basic sanity checks
    assert G["gamma"] > 0.0
    assert G["vega_per_1"] > 0.0
    assert -1.0 <= G["delta"]["put"] <= 0.0
    assert 0.0 <= G["delta"]["call"] <= 1.0

    # Theta (per day) should typically be negative for calls/puts (time decay),
    # though edge cases exist; we just ensure numbers are finite.
    assert math.isfinite(G["theta_per_day"]["call"])
    assert math.isfinite(G["theta_per_day"]["put"])


def test_implied_vol_recovers_sigma_call_and_put():
    inp = OptionInput(100, 105, 0.05, 0.20, 0.5, 0.0)
    c, p, *_ = bs_prices(inp)

    iv_c = implied_vol(c, inp, "call")
    iv_p = implied_vol(p, inp, "put")

    assert iv_c is not None and abs(iv_c - inp.sigma) < 1e-4
    assert iv_p is not None and abs(iv_p - inp.sigma) < 1e-4
