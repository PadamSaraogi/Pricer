# tests/test_black76.py
import math
from black76 import black76_price, black76_greeks, implied_vol_b76


def test_put_call_parity_black76():
    F0, K, r, T, sigma = 100.0, 95.0, 0.02, 0.75, 0.25
    D = math.exp(-r * T)
    C = black76_price(F0, K, r, T, sigma, "call")
    P = black76_price(F0, K, r, T, sigma, "put")
    # Parity: C - P = D*(F0 - K)
    assert abs((C - P) - D * (F0 - K)) < 1e-8


def test_iv_solver_recovers_sigma():
    F0, K, r, T, sigma_true = 100.0, 100.0, 0.01, 1.0, 0.3
    price = black76_price(F0, K, r, T, sigma_true, "call")
    iv = implied_vol_b76(price, F0, K, r, T, "call")
    assert iv is not None
    assert abs(iv - sigma_true) < 1e-4


def test_greeks_shapes_and_signs():
    F0, K, r, T, sigma = 120.0, 100.0, 0.02, 0.5, 0.25
    G = black76_greeks(F0, K, r, T, sigma)
    # delta on futures should be between 0..D for call; negative for put
    D = math.exp(-r * T)
    assert 0.0 < G["delta_fut"]["call"] < D + 1e-12
    assert -D - 1e-12 < G["delta_fut"]["put"] < 0.0
    # gamma and vega positive
    assert G["gamma_fut"] > 0.0
    assert G["vega"] > 0.0
