# tests/test_american_binomial.py

import math

from bs_core import OptionInput, bs_prices
from american_binomial import crr_price_european, crr_price_american


def approx(a, b, tol=1e-2):
    """Loose approx for tree convergence tests (tighter with more steps)."""
    return abs(a - b) < tol


def test_crr_european_converges_to_bs_call_put():
    """CRR (European) should converge to Black–Scholes as steps → large."""
    inp = OptionInput(S0=100, K=100, r=0.05, sigma=0.20, T=1.0, q=0.0)

    bs_call, bs_put, *_ = bs_prices(inp)
    steps = 1000  # high steps for tight convergence

    crr_call = crr_price_european(inp, "call", steps)
    crr_put = crr_price_european(inp, "put", steps)

    assert approx(crr_call, bs_call, tol=1e-2)
    assert approx(crr_put, bs_put, tol=1e-2)


def test_american_put_has_premium_over_european():
    """
    American put should be >= European (BS) due to early exercise value.
    Use ITM put to make premium clearer.
    """
    inp = OptionInput(S0=100, K=110, r=0.05, sigma=0.20, T=0.5, q=0.0)
    _, bs_put, *_ = bs_prices(inp)

    steps = 1000
    amer_put = crr_price_american(inp, "put", steps)

    assert amer_put >= bs_put
    # usually positive premium in this setup; allow small tolerance
    assert amer_put - bs_put >= -1e-3


def test_american_call_equals_european_when_no_dividends():
    """
    With q=0, American call equals European (no incentive to early exercise).
    """
    inp = OptionInput(S0=100, K=100, r=0.05, sigma=0.20, T=1.0, q=0.0)
    bs_call, _, *_ = bs_prices(inp)

    steps = 1000
    amer_call = crr_price_american(inp, "call", steps)
    euro_call = crr_price_european(inp, "call", steps)

    # American ≈ European ≈ BS call
    assert approx(amer_call, euro_call, tol=1e-3)
    assert approx(amer_call, bs_call, tol=1e-2)


def test_american_call_with_dividends_ge_european():
    """
    With dividends (q>0), early exercise on call can have value; American ≥ European.
    """
    inp = OptionInput(S0=100, K=100, r=0.05, sigma=0.20, T=1.0, q=0.04)
    steps = 800
    amer_call = crr_price_american(inp, "call", steps)
    euro_call = crr_price_european(inp, "call", steps)

    assert amer_call >= euro_call - 1e-4  # allow tiny numerical wiggle
